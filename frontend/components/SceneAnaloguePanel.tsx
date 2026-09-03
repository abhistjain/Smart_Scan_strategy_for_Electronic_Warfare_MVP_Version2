"use client";

import { useMemo } from "react";
import { useStore } from "@/lib/store";
import type { Classification, PeriodicityEstimate } from "@/lib/types";
import ClassificationDisclaimer from "./ClassificationDisclaimer";

/**
 * Competition demo panel: maps behaviour-pattern labels to illustrative
 * analogues (fighter-like / UAV-like / missile-like / datalink-like).
 * Not IFF. Not a real inbound-platform ID. Sort/attention only.
 */
function counts(list: Classification[]) {
  const acc: Record<string, { n: number; color: string; glyph: string; title: string }> = {};
  for (const c of list) {
    const key = c.analogue_key ?? "unknown";
    if (key === "unknown") continue;
    if (!acc[key]) {
      acc[key] = {
        n: 0,
        color: c.analogue_color ?? "#64748B",
        glyph: c.analogue_glyph ?? "·",
        title: c.analogue_short ?? "Unknown",
      };
    }
    acc[key].n += 1;
  }
  return Object.entries(acc).sort((a, b) => b[1].n - a[1].n);
}

function scanReason(
  c: Classification,
  belief: number,
  per: PeriodicityEstimate | undefined,
): string {
  const bits: string[] = [`belief ${belief.toFixed(2)}`];
  if (c.analogue_key === "uav" || c.short === "Agile/Hopping") {
    bits.push("high hop-rate agility");
  }
  if (c.analogue_key === "missile" || c.short === "Tightening-Pattern") {
    bits.push("tightening period trend");
  }
  if (c.analogue_key === "fighter") {
    bits.push("regular scan cadence");
  }
  if (c.urgency > 0.45 && per?.next_active_tick) {
    bits.push(`next window ~t=${Math.round(per.next_active_tick)}`);
  } else if (c.urgency > 0.45) {
    bits.push("high intercept-ahead urgency");
  }
  return bits.join(" · ");
}

export default function SceneAnaloguePanel() {
  const classification = useStore((s) => s.classification);
  const bandInfo = useStore((s) => s.bandInfo);
  const setSelectedBand = useStore((s) => s.setSelectedBand);
  const tick = useStore((s) => s.tick);
  const beliefWindow = useStore((s) => s.beliefWindow);
  const scannedWindow = useStore((s) => s.scannedWindow);
  const periodicity = useStore((s) => s.periodicity);

  const beliefs = beliefWindow[beliefWindow.length - 1] ?? [];
  const currentScanned = scannedWindow[scannedWindow.length - 1] ?? [];
  const scannedSet = useMemo(() => new Set(currentScanned), [currentScanned]);

  const perByBand = useMemo(() => {
    const m = new Map<number, PeriodicityEstimate>();
    for (const p of periodicity) m.set(p.band, p);
    return m;
  }, [periodicity]);

  const ranked = useMemo(
    () =>
      [...classification]
        .filter((c) => (c.analogue_key ?? "unknown") !== "unknown")
        .sort((a, b) => b.priority - a.priority)
        .slice(0, 5),
    [classification],
  );

  const tally = useMemo(() => counts(classification), [classification]);

  const recommended = useMemo(() => {
    if (!classification.length) return null;
    const byPri = [...classification].sort((a, b) => b.priority - a.priority);
    // Prefer the highest-priority band the receiver is not already dwelling on.
    return byPri.find((c) => !scannedSet.has(c.band)) ?? byPri[0];
  }, [classification, scannedSet]);

  const recBelief = recommended ? (beliefs[recommended.band] ?? 0) : 0;
  const recPer = recommended ? perByBand.get(recommended.band) : undefined;
  const recInTopM = recommended ? scannedSet.has(recommended.band) : false;
  const recName = recommended
    ? (bandInfo[recommended.band]?.label ?? `B${String(recommended.band).padStart(2, "0")}`)
    : "";

  return (
    <div className="flex h-full min-h-[180px] flex-col">
      <div className="mb-1 flex items-center justify-between px-1">
        <div className="flex items-center gap-1.5">
          <span className="font-mono text-[9px] uppercase tracking-widest text-slate-500">
            AI scene analogues
          </span>
          <ClassificationDisclaimer variant="icon" />
        </div>
        <span className="font-mono text-[9px] text-slate-600">t={tick} · demo only</span>
      </div>

      {recommended ? (
        <button
          type="button"
          onClick={() => setSelectedBand(recommended.band)}
          className="mb-2 w-full rounded-lg border border-cyan/40 bg-cyan/10 px-2.5 py-2 text-left hover:bg-cyan/15"
        >
          <div className="flex items-center justify-between gap-2">
            <span className="font-mono text-[9px] uppercase tracking-widest text-cyan">
              Recommended next scan
            </span>
            <span className="font-mono text-[9px] text-slate-400">
              {recInTopM ? "in Top-M this tick" : "not in current dwell"}
            </span>
          </div>
          <div className="mt-1 flex items-baseline justify-between gap-2">
            <span className="font-mono text-sm text-slate-100">
              {recName}
              {recommended.analogue_short && (
                <span
                  className="ml-2 text-[11px]"
                  style={{ color: recommended.analogue_color ?? "#22D3EE" }}
                >
                  {recommended.analogue_glyph} {recommended.analogue_short}
                </span>
              )}
            </span>
            <span className="shrink-0 font-mono text-[10px] text-slate-400">
              pri {recommended.priority.toFixed(2)}
            </span>
          </div>
          <p className="mt-1 font-mono text-[10px] leading-snug text-slate-300">
            {scanReason(recommended, recBelief, recPer)}
          </p>
        </button>
      ) : (
        <p className="mb-2 px-1 font-mono text-[10px] italic text-slate-500">
          Waiting for enough scan evidence to propose a next look…
        </p>
      )}

      {tally.length > 0 && (
        <div className="mb-2 flex flex-wrap gap-1.5 px-1">
          {tally.map(([key, v]) => (
            <span
              key={key}
              className="rounded px-1.5 py-0.5 font-mono text-[9px]"
              style={{ background: `${v.color}22`, color: v.color, border: `1px solid ${v.color}55` }}
            >
              {v.glyph} {v.title} ×{v.n}
            </span>
          ))}
        </div>
      )}

      <div className="min-h-0 flex-1 space-y-1 overflow-y-auto pr-1">
        {ranked.map((c) => (
          <button
            key={c.band}
            type="button"
            onClick={() => setSelectedBand(c.band)}
            className="flex w-full items-center gap-2 rounded px-1.5 py-1 text-left hover:bg-white/5"
          >
            <span className="w-10 shrink-0 font-mono text-[10px] text-slate-400">
              {bandInfo[c.band]?.label ?? `B${c.band}`}
            </span>
            <span className="w-4 text-center" style={{ color: c.analogue_color }}>
              {c.analogue_glyph}
            </span>
            <span className="flex-1 font-mono text-[10px]" style={{ color: c.analogue_color }}>
              {c.analogue_short}
            </span>
            <span className="font-mono text-[9px] text-slate-500">{c.short}</span>
          </button>
        ))}
      </div>
      <p className="mt-1 px-1 font-mono text-[8px] leading-tight text-slate-600">
        Next-scan is attention-only (belief / hop rate / tightening). Not a fire recommendation.
      </p>
    </div>
  );
}
