"use client";
import { ReactNode, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Evidence, ReportPayload } from "@/lib/api";
import CitationDrawer from "./CitationDrawer";

const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
function textOf(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(textOf).join("");
  return "";
}

/** Turn the backend's plain-markdown markers into links our custom renderer can style. */
function prepare(md: string): string {
  return md
    .replace(/\[UNVERIFIED\]/g, "[UNVERIFIED](flag:unverified)")
    .replace(/\*\*Confidence: (HIGH|MEDIUM|LOW)\*\*/g, "[$1](conf:$1)");
}

export default function ReportViewer(props: { report: ReportPayload; loadEvidence: (id: string) => Promise<Evidence>; actions?: ReactNode }) {
  const { report } = props;
  const [open, setOpen] = useState<string | null>(null);
  const toc = useMemo(() => [...report.markdown.matchAll(/^## (.+)$/gm)].map((m) => m[1]), [report.markdown]);
  const fc = report.factcheck;

  return (
    <>
      <div className="row" style={{ marginBottom: 14 }}>
        <h1 className="title grow" style={{ fontSize: 26 }}>{report.title}</h1>
        {props.actions}
      </div>
      <div className="report-grid">
        <nav className="toc" aria-label="Table of contents">
          {toc.map((t) => <a key={t} href={`#${slug(t)}`}>{t}</a>)}
          <a href="#fact-check">Fact-check summary</a>
        </nav>
        <div>
          <article className="doc">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              urlTransform={(url) => url}
              components={{
                h2: ({ children }) => <h2 id={slug(textOf(children))}>{children}</h2>,
                a: ({ href = "", children }) => {
                  if (href.startsWith("cite:")) {
                    return <button className="cite" onClick={() => setOpen(href.slice(5))} title="Show source">{children}</button>;
                  }
                  if (href.startsWith("flag:")) {
                    return <mark className="flag" title="This claim could not be matched to its source when re-checked, so treat it with caution.">[UNVERIFIED]</mark>;
                  }
                  if (href.startsWith("conf:")) {
                    const level = href.slice(5);
                    return <span className={`badge ${level}`} title={`Section confidence: ${level}`}>Confidence: {level}</span>;
                  }
                  return <a href={href} target="_blank" rel="noreferrer">{children}</a>;
                },
              }}
            >
              {prepare(report.markdown)}
            </ReactMarkdown>
          </article>

          <section className="panel fc" id="fact-check" style={{ marginTop: 18 }}>
            <h2>Fact-check summary</h2>
            <div className="row">
              <div><div className="big">{Math.round((fc.pass_rate ?? 0) * 100)}%</div><div className="muted">verified</div></div>
              <div className="muted">{fc.passed} of {fc.total_checked} sampled claims matched their source; {fc.failed} flagged.</div>
            </div>
            {fc.unverified_claims?.length > 0 && (
              <ul>{fc.unverified_claims.map((u, i) => <li key={i}>{u.claim} <span className="muted">(similarity {u.score})</span></li>)}</ul>
            )}
            {report.token_count != null && (
              <p className="muted">Evidence bundle: {report.token_count} of {report.token_budget} tokens
                {report.compression_log.length > 0 && `, ${report.compression_log.length} compression step(s) logged`}.</p>
            )}
          </section>
        </div>
      </div>
      {open && <CitationDrawer citeId={open} load={props.loadEvidence} onClose={() => setOpen(null)} />}
    </>
  );
}
