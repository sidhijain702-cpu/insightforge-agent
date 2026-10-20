"use client";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { API, LogLine, RunSnapshot } from "@/lib/api";

const mmss = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const STATUS_LABEL: Record<string, string> = {
  PLANNING: "Planning", RESEARCHING: "Researching", SYNTHESIZING: "Synthesizing", WRITING: "Writing",
  FACT_CHECKING: "Fact-checking", COMPLETE: "Complete", FAILED: "Failed", NEEDS_CLARIFICATION: "Needs clarification",
};

export default function ProgressView(props: { runId: string; onClarify: (question: string) => void; onReset: () => void }) {
  const [snap, setSnap] = useState<RunSnapshot | null>(null);
  const [lines, setLines] = useState<LogLine[]>([]);
  const [conn, setConn] = useState("connecting");
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const es = new EventSource(`${API}/research/status/${props.runId}`);
    es.onopen = () => setConn("live");
    es.onerror = () => setConn(es.readyState === EventSource.CLOSED ? "closed" : "reconnecting…");
    es.addEventListener("log", (e) => {
      const l = JSON.parse((e as MessageEvent).data) as LogLine;
      setLines((prev) => (prev.some((p) => p.id === l.id) ? prev : [...prev, l]));
    });
    es.addEventListener("state", (e) => setSnap(JSON.parse((e as MessageEvent).data) as RunSnapshot));
    es.addEventListener("done", (e) => { setSnap(JSON.parse((e as MessageEvent).data) as RunSnapshot); setConn("finished"); es.close(); });
    return () => es.close();
  }, [props.runId]);

  useEffect(() => { logRef.current?.scrollTo({ top: logRef.current.scrollHeight }); }, [lines]);

  const status = snap?.status ?? "PLANNING";
  const q = snap?.clarifying_question;
  return (
    <>
      <div className="panel">
        <div className="row">
          <h2 className="grow" style={{ margin: 0 }}>{STATUS_LABEL[status] ?? status}</h2>
          <span className="muted">elapsed {mmss(snap?.elapsed ?? 0)} · {conn}</span>
        </div>
        <div className="bar" style={{ margin: "12px 0" }} role="progressbar" aria-valuenow={snap?.progress ?? 0} aria-valuemin={0} aria-valuemax={100}>
          <div style={{ width: `${snap?.progress ?? 2}%` }} />
        </div>

        {status === "COMPLETE" && snap?.report_id && (
          <p><Link className="btn" href={`/reports/${snap.report_id}`}>Open report</Link></p>
        )}
        {status === "FAILED" && (
          <p className="err" role="alert">The run failed: {snap?.error}. <button className="btn ghost small" onClick={props.onReset}>Start over</button></p>
        )}
        {status === "NEEDS_CLARIFICATION" && q && (
          <div>
            <p><strong>One question before I start:</strong> {q}</p>
            <button className="btn" onClick={() => props.onClarify(q)}>Add detail to my brief</button>
          </div>
        )}
      </div>

      <div className="panel">
        <h2>Sub-tasks</h2>
        {!snap?.subtasks?.length ? <p className="muted">The planner is decomposing your brief…</p> : (
          <table className="t">
            <thead><tr><th>Sub-task</th><th>Source</th><th>Status</th><th>Detail</th></tr></thead>
            <tbody>
              {snap.subtasks.map((s) => (
                <tr key={s.id}>
                  <td><strong>{s.title}</strong><div className="muted">{s.query}</div></td>
                  <td>{s.source_type.replace("_", " ")}</td>
                  <td><span className={`pill ${s.status}`}>{s.status}</span></td>
                  <td className="muted">{s.detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="panel">
        <h2>Agent activity</h2>
        <div className="log" ref={logRef} aria-live="polite">
          {lines.filter((l) => l.kind !== "state").map((l) => (
            <div key={l.id} className={l.level}><b>{l.agent}:</b> {l.message}</div>
          ))}
        </div>
      </div>
    </>
  );
}
