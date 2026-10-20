"use client";
import { useEffect, useState } from "react";
import { Evidence } from "@/lib/api";

export default function CitationDrawer(props: { citeId: string; load: (id: string) => Promise<Evidence>; onClose: () => void }) {
  const [ev, setEv] = useState<Evidence | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    setEv(null); setErr("");
    props.load(props.citeId).then(setEv).catch((e) => setErr(String(e.message ?? e)));
  }, [props.citeId]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && props.onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [props]);

  return (
    <>
      <div className="drawer-back" onClick={props.onClose} />
      <aside className="drawer" role="dialog" aria-label="Source details">
        <div className="row"><h3 className="grow">Source</h3><button className="btn ghost small" autoFocus onClick={props.onClose}>Close</button></div>
        {err && <p className="err">{err}</p>}
        {!ev && !err && <p className="muted">Loading…</p>}
        {ev && (
          <>
            <p style={{ margin: "4px 0 0" }}><strong>{ev.title || ev.domain}</strong></p>
            <p className="muted" style={{ margin: 0 }}>
              {ev.url.startsWith("http") ? <a href={ev.url} target="_blank" rel="noreferrer">{ev.domain}</a> : ev.domain}
            </p>
            <dl className="kv">
              <dt>Published</dt><dd>{ev.date ?? "date unknown"}</dd>
              <dt>Credibility</dt>
              <dd>{ev.credibility.toFixed(2)}<span className="meter"><div style={{ width: `${ev.credibility * 100}%` }} /></span></dd>
              <dt>Evidence</dt><dd><span className={`badge ${ev.confidence}`}>{ev.confidence}</span>{ev.conflict && " · sources disagree"}</dd>
            </dl>
            <h3 style={{ fontSize: 14 }}>Paragraph this fact was drawn from</h3>
            <blockquote>{ev.paragraph || "No source paragraph was stored for this finding."}</blockquote>
            <h3 style={{ fontSize: 14 }}>Finding as extracted</h3>
            <p style={{ marginTop: 4 }}>{ev.claim}</p>
            {ev.compressed && <p className="muted">This finding condenses several lower-priority items to stay within the evidence token budget.</p>}
          </>
        )}
      </aside>
    </>
  );
}
