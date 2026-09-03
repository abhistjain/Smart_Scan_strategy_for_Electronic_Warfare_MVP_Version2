"use client";

import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useStore } from "@/lib/store";
import { summarizeRun } from "@/lib/api";
import { STRATEGY_LABELS, STRATEGY_COLORS, type StrategyKey } from "@/lib/types";
import ComparisonChart from "./ComparisonChart";
import ClassificationDisclaimer from "./ClassificationDisclaimer";
import FormattedAIText from "./FormattedAIText";

const STRATS: StrategyKey[] = ["smart", "sequential", "random", "greedy"];

export default function EndOfRunModal({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const scenarioId = useStore((s) => s.scenarioId);
  const metrics = useStore((s) => s.metrics);
  const tick = useStore((s) => s.tick);
  const [summary, setSummary] = useState("");
  const [available, setAvailable] = useState(true);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!open || !scenarioId) return;
    let cancelled = false;
    setLoading(true);
    setSummary("");
    summarizeRun(scenarioId)
      .then((r) => {
        if (cancelled) return;
        setAvailable(r.available);
        setSummary(r.text);
      })
      .catch(() => {
        if (!cancelled) {
          setAvailable(false);
          setSummary("AI narration unavailable");
        }
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [open, scenarioId]);

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
        >
          <motion.div
            className="glass max-h-[90vh] w-full max-w-4xl overflow-y-auto rounded-xl border border-white/10 p-5"
            initial={{ scale: 0.96, y: 12 }}
            animate={{ scale: 1, y: 0 }}
            exit={{ scale: 0.96, y: 12 }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="mb-4 flex items-start justify-between">
              <div>
                <h2 className="font-mono text-lg text-slate-100">End-of-Run Report</h2>
                <p className="font-mono text-[10px] text-slate-500">
                  {tick} ticks simulated · behaviour-pattern classification (illustrative)
                </p>
              </div>
              <button
                onClick={onClose}
                className="rounded px-2 py-1 font-mono text-sm text-slate-400 hover:bg-white/10"
              >
                ✕
              </button>
            </div>

            {/* Metric summary table */}
            <div className="mb-4 overflow-hidden rounded-lg border border-white/10">
              <table className="w-full font-mono text-[11px]">
                <thead className="bg-white/5 text-slate-400">
                  <tr>
                    <th className="p-2 text-left">Strategy</th>
                    <th className="p-2 text-right">Intercept rate</th>
                    <th className="p-2 text-right">Avg reward</th>
                    <th className="p-2 text-right">% correct</th>
                    <th className="p-2 text-right">Intercepts</th>
                  </tr>
                </thead>
                <tbody>
                  {STRATS.map((k) => (
                    <tr key={k} className="border-t border-white/5">
                      <td className="p-2 text-left" style={{ color: STRATEGY_COLORS[k] }}>
                        {STRATEGY_LABELS[k]}
                      </td>
                      <td className="p-2 text-right text-slate-200">
                        {(metrics[k].intercept_rate * 100).toFixed(1)}%
                      </td>
                      <td className="p-2 text-right text-slate-200">
                        {metrics[k].avg_reward.toFixed(3)}
                      </td>
                      <td className="p-2 text-right text-slate-200">
                        {(metrics[k].pct_correct * 100).toFixed(1)}%
                      </td>
                      <td className="p-2 text-right text-slate-200">
                        {metrics[k].total_intercepts}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="mb-4 h-56">
              <ComparisonChart />
            </div>

            {/* AI narrative summary */}
            <div className="rounded-lg border border-white/10 bg-white/5 p-3">
              <div className="mb-2 flex items-center justify-between">
                <span className="font-mono text-[9px] uppercase tracking-widest text-slate-500">
                  AI-generated summary
                </span>
                <span className="font-mono text-[9px] text-slate-600">Powered by Claude</span>
              </div>
              {loading ? (
                <div className="space-y-2">
                  <div className="h-3 w-full animate-pulse rounded bg-white/10" />
                  <div className="h-3 w-11/12 animate-pulse rounded bg-white/10" />
                  <div className="h-3 w-4/5 animate-pulse rounded bg-white/10" />
                </div>
              ) : (
                <FormattedAIText text={summary || "—"} muted={!available} />
              )}
            </div>

            <div className="mt-3">
              <ClassificationDisclaimer variant="icon" />
              <span className="ml-2 font-mono text-[9px] text-slate-500">
                This report explains what the system observed and how the scheduler
                performed — not an operational assessment.
              </span>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
