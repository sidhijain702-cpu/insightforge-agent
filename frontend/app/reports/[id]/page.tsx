"use client";
import { useEffect, useState } from "react";
import ReportViewer from "@/components/ReportViewer";
import { API, Evidence, ReportPayload, api } from "@/lib/api";

export default function ReportPage({ params }: { params: { id: string } }) {
  const { id } = params;
  const [report, setReport] = useState<ReportPayload | null>(null);
  const [err, setErr] = useState("");
  const [shareUrl, setShareUrl] = useState("");
  const [copied, setCopied] = useState(false);

  useEffect(() => { api<ReportPayload>(`/reports/${id}`).then(setReport).catch((e) => setErr(e.message)); }, [id]);

  async function share() {
    const j = await api<{ url: string; expires_at: string }>(`/reports/${id}/share`, { method: "POST" });
    setShareUrl(j.url);
    try { await navigator.clipboard.writeText(j.url); setCopied(true); } catch { setCopied(false); }
  }

  if (err) return <p className="err">{err}</p>;
  if (!report) return <p className="muted">Loading report…</p>;
  return (
    <>
      <ReportViewer
        report={report}
        loadEvidence={(cid) => api<Evidence>(`/reports/${id}/evidence/${cid}`)}
        actions={
          <div className="row">
            <a className="btn ghost small" href={`${API}/reports/${id}/export?format=pdf`}>Download PDF</a>
            <a className="btn ghost small" href={`${API}/reports/${id}/export?format=md`}>Download .md</a>
            <button className="btn small" onClick={share}>Copy share link</button>
          </div>
        }
      />
      {shareUrl && (
        <p className="muted" role="status">
          {copied ? "Link copied. " : "Copy this link: "}Read-only, expires in 7 days.<br /><code>{shareUrl}</code>
        </p>
      )}
    </>
  );
}
