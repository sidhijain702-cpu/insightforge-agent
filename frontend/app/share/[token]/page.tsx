"use client";
import { useEffect, useState } from "react";
import ReportViewer from "@/components/ReportViewer";
import { Evidence, ReportPayload, api } from "@/lib/api";

export default function SharedReport({ params }: { params: { token: string } }) {
  const { token } = params;
  const [report, setReport] = useState<ReportPayload | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => { api<ReportPayload>(`/share/${token}`).then(setReport).catch((e) => setErr(e.message)); }, [token]);

  if (err) return <div className="panel"><h2>This link can't be opened</h2><p className="err">{err}</p></div>;
  if (!report) return <p className="muted">Loading shared report…</p>;
  return (
    <>
      <p className="muted">Shared read-only view.</p>
      <ReportViewer report={report} loadEvidence={(cid) => api<Evidence>(`/share/${token}/evidence/${cid}`)} />
    </>
  );
}
