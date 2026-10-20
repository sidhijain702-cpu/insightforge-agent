"use client";
import { useState } from "react";
import BriefForm from "@/components/BriefForm";
import ProgressView from "@/components/ProgressView";

export default function Home() {
  const [runId, setRunId] = useState<string | null>(null);
  const [brief, setBrief] = useState("");

  return (
    <>
      <h1 className="title">Ask a market question. Get a cited report.</h1>
      <p className="lede">Describe what you need to know. Agents search, read, cross-check and write; every claim links to the paragraph it came from.</p>
      {runId ? (
        <ProgressView
          runId={runId}
          onReset={() => setRunId(null)}
          onClarify={(q) => { setBrief((b) => `${b}\n(${q}) `); setRunId(null); }}
        />
      ) : (
        <BriefForm key={brief} initialBrief={brief} onStarted={(id, b) => { setBrief(b); setRunId(id); }} />
      )}
    </>
  );
}
