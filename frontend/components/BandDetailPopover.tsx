"use client";

import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useStore } from "@/lib/store";
import { fetchBandDetail, narrateBand } from "@/lib/api";
import type { BandDetail } from "@/lib/types";
import ClassificationDisclaimer from "./ClassificationDisclaimer";
import FormattedAIText from "./FormattedAIText";

function localNarration(d: BandDetail): string {
  const cls = d.classification;
  const f = d.features;
  const name = d.label ?? `Band ${d.band}`;
  const analogue = cls.analogue_title ?? cls.analogue_short ?? "unclassified analogue";
  const conf = Number.isFinite(cls.confidence) ? `${Math.round(cls.confidence * 100)}%` : "n/a";
  return (
    `${name} matches a ${cls.short} pattern, shown as a ${analogue} ` +
    `for this demo (confidence ${conf}, ${f.evidence} scans of evidence). ` +
    `Duty cycle ${f.duty_cycle.toFixed(2)}, hop rate ${f.hop_rate.toFixed(2)}, ` +
    `period ${f.period.toFixed(1)} ticks (periodicity ${f.periodicity_strength.toFixed(2)}). ` +
    `This is an illustrative signal-behaviour analogue — not a real platform identification.`
  );
}

function Bar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-28 shrink-0 font-mono text-[10px] text-slate-400">{label}</span>
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-white/5">
        <div
          className="h-full rounded-full"
          style={{ width: `${Math.round(Math.min(1, Math.max(0, value)) * 100)}%`, background: color }}
        />
      </div>
      <span className="w-10 text-right font-mono text-[10px] text-slate-300">
        {value.toFixed(2)}
      </span>
    </div>
  );
}

export default function BandDetailPopover() {
  const scenarioId = useStore((s) => s.scenarioId);
  const selectedBand = useStore((s) => s.selectedBand);
  const setSelectedBand = useStore((s) => s.setSelectedBand);
  const liveClass = useStore((s) =>
    selectedBand == null ? undefined : s.classification.find((c) => c.band === selectedBand),
  );

  const [detail, setDetail] = useState<BandDetail | null>(null);
  const [narration, setNarration] = useState<string>("");
  const [narrAvailable, setNarrAvailable] = useState<boolean>(true);
  const [loading, setLoading] = useState(false);
  const [narrLoading, setNarrLoading] = useState(false);

  useEffect(() => {
    if (selectedBand == null || !scenarioId) return;
    let cancelled = false;
    setLoading(true);
    setNarration("");
    fetchBandDetail(scenarioId, selectedBand)
      .then((d) => {
        if (cancelled) return;
        setDetail(d);
        // Instant local copy so the box is never blank while Claude thinks.
        setNarration((prev) => prev || localNarration(d));
        setNarrAvailable(true);
      })
      .catch(() => {
        if (cancelled || !liveClass) return;
        const analogue = liveClass.analogue_title ?? liveClass.analogue_short;
        setNarrAvailable(true);
        setNarration(
          `Band ${liveClass.band} matches a ${liveClass.short} pattern` +
            (analogue ? `, shown as a ${analogue}` : "") +
            ` (confidence ${Math.round(liveClass.confidence * 100)}%). ` +
            `Illustrative signal-behaviour analogue — not a real platform identification.`,
        );
      })
      .finally(() => !cancelled && setLoading(false));

    setNarrLoading(true);
    narrateBand(scenarioId, selectedBand)
      .then((r) => {
        if (cancelled) return;
        if (r.text && r.text !== "AI narration unavailable") {
          setNarrAvailable(true);
          setNarration(r.text);
        }
      })
      .catch(() => {
        /* keep the local fallback already written from band detail */
      })
      .finally(() => !cancelled && setNarrLoading(false));

    return () => {
      cancelled = true;
    };
  }, [selectedBand, scenarioId]);

  const cls = detail?.classification;
  const b = detail?.priority_breakdown;
  const f = detail?.features;

  return (
    <AnimatePresence>
      {selectedBand != null && (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={() => setSelectedBand(null)}
        >
          <motion.div
            className="glass w-full max-w-lg rounded-xl border border-white/10 p-4"
            initial={{ scale: 0.95, y: 10 }}
            animate={{ scale: 1, y: 0 }}
            exit={{ scale: 0.95, y: 10 }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="mb-3 flex items-start justify-between">
              <div>
                <h3 className="font-mono text-sm text-slate-200">
                  {detail?.label ?? `Band ${selectedBand}`}
                  <span className="ml-2 text-[10px] text-slate-500">#{selectedBand}</span>
                </h3>
                {cls && (
                  <div className="mt-1 flex flex-wrap gap-1.5">
                    <span
                      className="inline-block rounded px-1.5 py-0.5 font-mono text-[10px]"
                      style={{ background: `${cls.color}22`, color: cls.color, border: `1px solid ${cls.color}55` }}
                    >
                      {cls.short} · {(cls.confidence * 100).toFixed(0)}%
                    </span>
                    {cls.analogue_short && (
                      <span
                        className="inline-block rounded px-1.5 py-0.5 font-mono text-[10px]"
                        style={{
                          background: `${cls.analogue_color ?? cls.color}22`,
                          color: cls.analogue_color ?? cls.color,
                          border: `1px solid ${cls.analogue_color ?? cls.color}55`,
                        }}
                      >
                        {cls.analogue_glyph} {cls.analogue_short}
                      </span>
                    )}
                  </div>
                )}
              </div>
              <div className="flex items-center gap-2">
                <ClassificationDisclaimer variant="icon" />
                <button
                  onClick={() => setSelectedBand(null)}
                  className="rounded px-2 py-0.5 font-mono text-xs text-slate-400 hover:bg-white/10"
                >
                  ✕
                </button>
              </div>
            </div>

            {loading && <p className="font-mono text-xs text-slate-500">Loading…</p>}

            {/* AI narration */}
            <div className="mb-3 rounded-lg border border-white/10 bg-white/5 p-2.5">
              <div className="mb-1 flex items-center justify-between">
                <span className="font-mono text-[9px] uppercase tracking-widest text-slate-500">
                  AI Narration
                </span>
                <span className="font-mono text-[9px] text-slate-600">Powered by Claude</span>
              </div>
              {narrLoading ? (
                <div className="h-3 w-3/4 animate-pulse rounded bg-white/10" />
              ) : (
                <FormattedAIText text={narration || "—"} muted={!narrAvailable} />
              )}
            </div>

            {/* Priority breakdown */}
            {b && (
              <div className="mb-3 space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[9px] uppercase tracking-widest text-slate-500">
                    Threat Priority Score (attention sort only)
                  </span>
                  <span className="font-mono text-xs text-cyan">{detail?.priority.toFixed(3)}</span>
                </div>
                <Bar label="belief term" value={b.belief_term} color="#22D3EE" />
                <Bar label="confidence term" value={b.confidence_term} color="#34D399" />
                <Bar label="urgency term" value={b.urgency_term} color="#F5A623" />
              </div>
            )}

            {/* Feature vector + matched rule */}
            {f && (
              <div className="rounded-lg border border-white/10 bg-white/5 p-2.5">
                <div className="mb-1.5 font-mono text-[9px] uppercase tracking-widest text-slate-500">
                  Signal Features · matched rule
                </div>
                <p className="mb-2 font-mono text-[10px] text-amber">
                  {cls?.matched_rule}
                </p>
                <div className="grid grid-cols-2 gap-x-4 gap-y-1 font-mono text-[10px] text-slate-400">
                  <span>duty cycle: <span className="text-slate-200">{f.duty_cycle.toFixed(2)}</span></span>
                  <span>periodicity: <span className="text-slate-200">{f.periodicity_strength.toFixed(2)}</span></span>
                  <span>period: <span className="text-slate-200">{f.period.toFixed(1)}</span></span>
                  <span>period trend: <span className="text-slate-200">{f.period_trend.toFixed(3)}</span></span>
                  <span>hop rate: <span className="text-slate-200">{f.hop_rate.toFixed(2)}</span></span>
                  <span>bandwidth: <span className="text-slate-200">{f.bandwidth.toFixed(2)}</span></span>
                  <span>evidence: <span className="text-slate-200">{f.evidence}</span></span>
                  <span>amp stability: <span className="text-slate-200">{f.amplitude_stability.toFixed(2)}</span></span>
                </div>
              </div>
            )}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
