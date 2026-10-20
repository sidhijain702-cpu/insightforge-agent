"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import { api, fmtTime } from "@/lib/api";

type Row = { id: string; title: string; watch_id: number | null; pass_rate: number | null; created_at: number };

export default function Reports() {
  const [rows, setRows] = useState<Row[] | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => { api<Row[]>("/reports").then(setRows).catch((e) => setErr(e.message)); }, []);

  return (
    <>
      <h1 className="title">Reports</h1>
      <p className="lede">Every finished run is saved here with its fact-check result.</p>
      {err && <p className="err">{err}</p>}
      {rows && rows.length === 0 && <div className="panel">No reports yet. <Link href="/">Run your first brief.</Link></div>}
      {rows && rows.length > 0 && (
        <div className="panel">
          <table className="t">
            <thead><tr><th>Report</th><th>Created</th><th>Verified</th><th>Origin</th></tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.id}>
                  <td><Link href={`/reports/${r.id}`}>{r.title}</Link></td>
                  <td className="muted">{fmtTime(r.created_at)}</td>
                  <td>{r.pass_rate == null ? "n/a" : `${Math.round(r.pass_rate * 100)}%`}</td>
                  <td className="muted">{r.watch_id ? "watchlist" : "manual brief"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
