export const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(API + path, init);
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const j = await res.json();
      msg = typeof j.detail === "string" ? j.detail : msg;
    } catch {
      /* body was not JSON */
    }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export type SubTaskRow = { id: string; title: string; query: string; source_type: string; status: string; detail: string };
export type RunSnapshot = {
  run_id: string; status: string; progress: number; elapsed: number; subtasks: SubTaskRow[];
  report_id: string | null; error: string | null; clarifying_question: string | null;
};
export type LogLine = { id: number; ts: number; agent: string; message: string; kind: string; level: string };
export type FactCheck = {
  total_checked: number; passed: number; failed: number; pass_rate: number;
  unverified_claims: { claim: string; cite: string; url: string; score: number }[];
};
export type ReportPayload = {
  id: string; title: string; brief: string; markdown: string; created_at: number; factcheck: FactCheck;
  compression_log: { section: string; action: string; items_affected: number; reason: string; tokens_before: number; tokens_after: number }[];
  token_count: number | null; token_budget: number | null;
};
export type Evidence = {
  id: string; claim: string; url: string; domain: string; title: string; date: string | null;
  credibility: number; confidence: string; paragraph: string; conflict: boolean; compressed: boolean;
};
export type WatchItem = {
  id: number; topic: string; cadence: "daily" | "weekly"; webhook_url: string | null;
  last_run: number | null; next_run: number | null; active: number;
};
export type Changes = { added_findings: string[]; removed_findings: string[]; changed_findings: string[] };
export type Alert = { id: number; watch_id: number; topic: string; summary: Changes; created_at: number; read: number; webhook_status: string };

export const fmtTime = (t: number | null) => (t ? new Date(t * 1000).toLocaleString() : "not yet");
