"use client";

import { motion } from "framer-motion";
import { useStore } from "@/lib/store";

function EmitterIcon({ type }: { type: string }) {
  // Compact glyphs for static / periodic radar / frequency hopper / quiet.
  const map: Record<string, { glyph: string; title: string; color: string }> = {
    markov: { glyph: "◈", title: "static / random emitter", color: "#94a3b8" },
    periodic: { glyph: "⟳", title: "periodic scan (rotating) emitter", color: "#22D3EE" },
    hopper: { glyph: "⇄", title: "frequency-hopping (agile) emitter", color: "#F5A623" },
    tsrd: { glyph: "◉", title: "TSRD-derived emitter", color: "#A78BFA" },
    quiet: { glyph: "·", title: "quiet band", color: "#475569" },
  };
  const info = map[type] ?? map.quiet;
  return (
    <span title={info.title} style={{ color: info.color }} className="text-sm">
      {info.glyph}
    </span>
  );
}

export default function BeliefBarPanel() {
  const beliefWindow = useStore((s) => s.beliefWindow);
  const scannedWindow = useStore((s) => s.scannedWindow);
  const bandInfo = useStore((s) => s.bandInfo);
  const nBands = useStore((s) => s.nBands);
  const dataSource = useStore((s) => s.dataSource);

  const beliefs = beliefWindow[beliefWindow.length - 1] ?? new Array(nBands).fill(0);
  const currentScanned = new Set(scannedWindow[scannedWindow.length - 1] ?? []);

  return (
    <div className="flex h-full flex-col">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="font-mono text-xs uppercase tracking-widest text-slate-400">
          Live Belief · P(active)
        </h3>
        <span className="font-mono text-[10px] text-slate-500">
          {dataSource === "real_tsrd" ? "TSRD-derived" : "synthetic"}
        </span>
      </div>
      <div className="flex-1 space-y-1 overflow-y-auto pr-1">
        {Array.from({ length: nBands }).map((_, b) => {
          const val = beliefs[b] ?? 0;
          const info = bandInfo[b];
          const isTopM = currentScanned.has(b);
          const color = val > 0.66 ? "#22D3EE" : val > 0.4 ? "#F5A623" : "#3B4A6B";
          return (
            <div
              key={b}
              className={`flex items-center gap-2 rounded px-1.5 py-0.5 ${
                isTopM ? "bg-cyan/10 ring-1 ring-cyan/40" : ""
              }`}
            >
              <span className="w-14 shrink-0 font-mono text-[10px] text-slate-400">
                {info?.label ?? `B${b}`}
              </span>
              <EmitterIcon type={info?.emitter_type ?? "quiet"} />
              <div className="relative h-2.5 flex-1 overflow-hidden rounded-full bg-white/5">
                <motion.div
                  className="absolute left-0 top-0 h-full rounded-full"
                  style={{ background: color }}
                  animate={{ width: `${Math.round(val * 100)}%` }}
                  transition={{ duration: 0.25 }}
                />
                {isTopM && (
                  <div className="absolute inset-0 animate-pulse-glow rounded-full" />
                )}
              </div>
              <span className="w-8 shrink-0 text-right font-mono text-[10px] text-slate-300">
                {val.toFixed(2)}
              </span>
            </div>
          );
        })}
      </div>
      <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 border-t border-white/10 pt-2 font-mono text-[9px] text-slate-500">
        <span><span className="text-cyan">⟳</span> periodic</span>
        <span><span className="text-amber">⇄</span> hopper</span>
        <span><span className="text-slate-400">◈</span> static</span>
        <span className="text-cyan">▣ Top-M scanned</span>
      </div>
    </div>
  );
}
