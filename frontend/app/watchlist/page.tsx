"use client";
import { useCallback, useEffect, useState } from "react";
import { Alert, Changes, WatchItem, api, fmtTime } from "@/lib/api";

export default function Watchlist() {
  const [items, setItems] = useState<WatchItem[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [topic, setTopic] = useState("");
  const [cadence, setCadence] = useState<"daily" | "weekly">("daily");
  const [webhook, setWebhook] = useState("");
  const [diff, setDiff] = useState<{ topic: string; summary: Changes; created_at?: number } | null>(null);
  const [err, setErr] = useState("");
  const [note, setNote] = useState("");

  const refresh = useCallback(() => {
    api<WatchItem[]>("/watchlist").then(setItems).catch((e) => setErr(e.message));
    api<Alert[]>("/alerts").then(setAlerts).catch(() => undefined);
  }, []);
  useEffect(() => { refresh(); const t = setInterval(refresh, 8000); return () => clearInterval(t); }, [refresh]);

  async function add() {
    setErr("");
    try {
      await api("/watchlist", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topic, cadence, webhook_url: webhook || null }) });
      setTopic(""); setWebhook(""); refresh();
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
  }
  const act = async (path: string, init?: RequestInit, msg?: string) => { await api(path, init); if (msg) setNote(msg); refresh(); };
  async function showDiff(w: WatchItem) {
    const d = await api<{ summary: Changes; created_at?: number }>(`/watchlist/${w.id}/diff`);
    setDiff({ topic: w.topic, summary: d.summary, created_at: d.created_at });
  }
  const empty = (c: Changes) => !c.added_findings.length && !c.changed_findings.length && !c.removed_findings.length;

  return (
    <>
      <h1 className="title">Watchlist</h1>
      <p className="lede">Topics are re-researched on a schedule. You're alerted only when something materially new appears, not when sources reword old facts.</p>

      <div className="panel">
        <h2>Watch a topic</h2>
        <div className="row">
          <input className="grow" type="text" placeholder="e.g. Deel and Rippling payroll expansion in Asia" value={topic} onChange={(e) => setTopic(e.target.value)} aria-label="Topic" />
          <select value={cadence} onChange={(e) => setCadence(e.target.value as "daily" | "weekly")} style={{ width: 120 }} aria-label="Cadence">
            <option value="daily">Daily</option><option value="weekly">Weekly</option>
          </select>
        </div>
        <div className="row" style={{ marginTop: 10 }}>
          <input className="grow" type="url" placeholder="Slack incoming webhook URL (optional)" value={webhook} onChange={(e) => setWebhook(e.target.value)} aria-label="Webhook URL" />
          <button className="btn" disabled={!topic.trim()} onClick={add}>Add to watchlist</button>
        </div>
        {err && <p className="err" role="alert">{err}</p>}
      </div>

      <div className="panel">
        <h2>Monitored topics</h2>
        {note && <p className="muted" role="status">{note}</p>}
        {items.length === 0 ? <p className="muted">Nothing watched yet. Add a topic above.</p> : (
          <table className="t">
            <thead><tr><th>Topic</th><th>Cadence</th><th>Last run</th><th>Next run</th><th>Actions</th></tr></thead>
            <tbody>
              {items.map((w) => (
                <tr key={w.id}>
                  <td>{w.topic}{!w.active && <span className="muted"> (paused)</span>}</td>
                  <td>{w.cadence}</td><td>{fmtTime(w.last_run)}</td><td>{w.active ? fmtTime(w.next_run) : "paused"}</td>
                  <td>
                    <div className="row">
                      <button className="btn ghost small" onClick={() => act(`/watchlist/${w.id}/run`, { method: "POST" }, `Run started for "${w.topic}". It takes a few minutes; alerts appear below.`)}>Run now</button>
                      <button className="btn ghost small" onClick={() => showDiff(w)}>What changed</button>
                      <button className="btn ghost small" onClick={() => act(`/watchlist/${w.id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ active: !w.active }) })}>{w.active ? "Pause" : "Resume"}</button>
                      <button className="btn ghost small danger" onClick={() => act(`/watchlist/${w.id}`, { method: "DELETE" })}>Delete</button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {diff && (
        <div className="panel">
          <div className="row"><h2 className="grow">Latest change: {diff.topic}</h2><button className="btn ghost small" onClick={() => setDiff(null)}>Close</button></div>
          {empty(diff.summary) ? <p className="muted">No material change has been detected yet. The first run only sets the baseline; changes appear from the second run.</p> : <ChangeList c={diff.summary} />}
        </div>
      )}

      <div className="panel">
        <h2>Alert feed</h2>
        {alerts.length === 0 ? <p className="muted">No alerts yet.</p> : alerts.map((a) => (
          <div key={a.id} style={{ borderTop: "1px solid var(--paper)", padding: "12px 0" }}>
            <strong>{a.topic}</strong> <span className="muted">· {fmtTime(a.created_at)}{a.webhook_status && ` · webhook ${a.webhook_status}`}</span>
            <ChangeList c={a.summary} />
          </div>
        ))}
      </div>
    </>
  );
}

function ChangeList({ c }: { c: Changes }) {
  const block = (label: string, xs: string[]) => xs.length > 0 && (
    <div><div className="muted" style={{ marginTop: 6 }}>{label}</div><ul style={{ margin: "4px 0" }}>{xs.map((x, i) => <li key={i}>{x}</li>)}</ul></div>
  );
  return <>{block("New", c.added_findings)}{block("Changed", c.changed_findings)}{block("No longer found", c.removed_findings)}</>;
}
