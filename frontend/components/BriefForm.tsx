"use client";
import { useRef, useState } from "react";
import { API } from "@/lib/api";

const MAX_CHARS = 500, MAX_FILES = 10, MAX_BYTES = 20 * 1024 * 1024;

/** Runtime estimate: sub-task count (3-6) drives it; they run in parallel but share API rate limits. */
export function estimate(chars: number, files: number, prior: boolean): string {
  const n = Math.min(6, 3 + (chars > 220 ? 1 : 0) + (files > 0 ? 1 : 0) + (prior ? 0 : 0));
  const lo = Math.ceil((60 + 20 * n) / 60), hi = Math.ceil((120 + 45 * n) / 60);
  return `about ${lo}-${hi} min (${n} parallel sub-tasks)`;
}

export default function BriefForm(props: { initialBrief?: string; onStarted: (runId: string, brief: string) => void }) {
  const [brief, setBrief] = useState(props.initialBrief ?? "");
  const [files, setFiles] = useState<File[]>([]);
  const [web, setWeb] = useState(true);
  const [docs, setDocs] = useState(true);
  const [prior, setPrior] = useState(true);
  const [over, setOver] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  function addFiles(list: FileList | null) {
    if (!list) return;
    const next = [...files];
    for (const f of Array.from(list)) {
      if (!f.name.toLowerCase().endsWith(".pdf")) { setError(`${f.name}: only PDF files are supported`); continue; }
      if (f.size > MAX_BYTES) { setError(`${f.name} is larger than 20 MB`); continue; }
      if (next.length >= MAX_FILES) { setError(`At most ${MAX_FILES} PDFs per run`); break; }
      next.push(f);
    }
    setFiles(next);
  }

  async function submit() {
    setError(""); setBusy(true);
    const fd = new FormData();
    fd.append("brief", brief.trim());
    fd.append("web", String(web)); fd.append("docs", String(docs)); fd.append("prior", String(prior));
    files.forEach((f) => fd.append("files", f));
    try {
      const res = await fetch(`${API}/research/run`, { method: "POST", body: fd });
      const j = await res.json();
      if (!res.ok) throw new Error(typeof j.detail === "string" ? j.detail : "Could not start the run");
      props.onStarted(j.run_id, brief.trim());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  }

  const tooLong = brief.length > MAX_CHARS;
  const noSource = !web && !prior && !(docs && files.length > 0);
  return (
    <div className="panel">
      <label htmlFor="brief"><h2>Research brief</h2></label>
      <textarea id="brief" value={brief} onChange={(e) => setBrief(e.target.value)}
        placeholder="Summarize the competitive landscape for B2B payroll software in Southeast Asia. Focus on pricing, recent funding, and product gaps." />
      <div className={`counter ${tooLong ? "over" : ""}`}>{brief.length} / {MAX_CHARS}</div>

      <h2 style={{ marginTop: 16 }}>Sources</h2>
      <div className="toggles">
        <label><input type="checkbox" checked={web} onChange={(e) => setWeb(e.target.checked)} /> Web search</label>
        <label><input type="checkbox" checked={docs} onChange={(e) => setDocs(e.target.checked)} /> Uploaded documents</label>
        <label><input type="checkbox" checked={prior} onChange={(e) => setPrior(e.target.checked)} /> Prior reports</label>
      </div>

      <div className={`drop ${over ? "over" : ""}`} style={{ marginTop: 14 }} role="button" tabIndex={0}
        onClick={() => input.current?.click()} onKeyDown={(e) => e.key === "Enter" && input.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); addFiles(e.dataTransfer.files); }}>
        Drop PDFs here or click to choose (up to {MAX_FILES} files, 20 MB each)
        <input ref={input} type="file" accept="application/pdf" multiple hidden onChange={(e) => { addFiles(e.target.files); e.target.value = ""; }} />
      </div>
      {files.length > 0 && (
        <ul className="files">
          {files.map((f, i) => (
            <li key={f.name + i}><span>{f.name} <span className="muted">({(f.size / 1048576).toFixed(1)} MB)</span></span>
              <button className="btn ghost small" onClick={() => setFiles(files.filter((_, k) => k !== i))}>Remove</button></li>
          ))}
        </ul>
      )}

      <div className="row" style={{ marginTop: 18 }}>
        <button className="btn" disabled={busy || !brief.trim() || tooLong || noSource} onClick={submit}>
          {busy ? "Starting…" : "Run research"}
        </button>
        <span className="muted">Estimated runtime: {estimate(brief.length, docs ? files.length : 0, prior)}</span>
      </div>
      {noSource && <p className="muted">Turn on at least one source (uploaded documents need a PDF).</p>}
      {error && <p className="err" role="alert">{error}</p>}
    </div>
  );
}
