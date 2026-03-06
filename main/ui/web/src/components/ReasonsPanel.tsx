import { motion } from "framer-motion";
import { useMemo } from "react";
import type { Reason } from "../api";

const severityClass: Record<string, string> = {
  low: "bg-slate-600/70 text-slate-100",
  medium: "bg-cyan-700/70 text-cyan-100",
  high: "bg-white/80 text-black",
  context: "bg-slate-700/60 text-slate-200",
};

function normalizeReason(r: Reason) {
  const title = r.reason || "Signal";
  const severity = (r.severity || "context").toLowerCase();
  return {
    title,
    severity,
    summary: r.summary || "No summary provided.",
    why: r.why_it_matters || "No additional context.",
  };
}

export function ReasonsPanel({ reasons }: { reasons: Reason[] }) {
  const parsed = useMemo(() => reasons.map(normalizeReason), [reasons]);

  return (
    <section className="glass-panel p-4">
      <h2 className="mb-3 text-sm uppercase tracking-[0.16em] text-slate-300">Reason Signals</h2>
      {parsed.length === 0 && <div className="rounded-lg border border-white/10 bg-black/30 p-3 text-sm text-slate-300">No reasons available.</div>}
      <div className="space-y-3">
        {parsed.map((r, idx) => (
          <motion.article
            key={`${r.title}-${idx}`}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-xl border border-white/10 bg-black/35 p-4"
          >
            <div className="mb-2 flex items-center justify-between gap-2">
              <h3 className="text-base font-semibold text-white">{r.title}</h3>
              <span className={`rounded-full px-2 py-1 text-[11px] uppercase tracking-[0.12em] ${severityClass[r.severity] || severityClass.context}`}>
                {r.severity.toUpperCase()}
              </span>
            </div>
            <p className="text-sm text-slate-200">{r.summary}</p>
            <details className="mt-2 rounded-lg border border-white/10 bg-black/25 p-2 text-xs text-slate-300">
              <summary className="cursor-pointer uppercase tracking-[0.12em]">Why it matters</summary>
              <p className="mt-2 leading-relaxed">{r.why}</p>
            </details>
          </motion.article>
        ))}
      </div>
    </section>
  );
}

export function SeverityLegend() {
  return (
    <aside className="glass-panel h-fit w-full max-w-full p-4 sm:p-5 xl:sticky xl:top-6 xl:max-w-[340px]">
      <h2 className="mb-3 text-sm uppercase tracking-[0.16em] text-slate-300">Severity Legend</h2>
      <div className="space-y-2 text-sm">
        <div className="rounded-lg border border-white/10 bg-black/30 p-3">
          <span className={`mr-2 rounded-full px-2 py-1 text-[11px] uppercase tracking-[0.12em] ${severityClass.low}`}>LOW</span>
          mild anomaly, likely explainable by normal play context.
        </div>
        <div className="rounded-lg border border-white/10 bg-black/30 p-3">
          <span className={`mr-2 rounded-full px-2 py-1 text-[11px] uppercase tracking-[0.12em] ${severityClass.medium}`}>MEDIUM</span>
          repeated pattern worth manual review with video context.
        </div>
        <div className="rounded-lg border border-white/10 bg-black/30 p-3">
          <span className={`mr-2 rounded-full px-2 py-1 text-[11px] uppercase tracking-[0.12em] ${severityClass.high}`}>HIGH</span>
          strong statistical outlier; prioritize this evidence.
        </div>
        <div className="rounded-lg border border-white/10 bg-black/30 p-3">
          <span className={`mr-2 rounded-full px-2 py-1 text-[11px] uppercase tracking-[0.12em] ${severityClass.context}`}>CONTEXT</span>
          supporting context; informative but not standalone proof.
        </div>
      </div>
    </aside>
  );
}
