"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useStore } from "@/lib/store";

// Compact badge strip showing the bands the Smart Scheduler is scanning THIS
// tick (its Top-M selection) plus which of them produced a hit.
export default function TopMHighlightOverlay() {
  const scannedWindow = useStore((s) => s.scannedWindow);
  const latestHits = useStore((s) => s.latestHits);
  const bandInfo = useStore((s) => s.bandInfo);

  const scanned = scannedWindow[scannedWindow.length - 1] ?? [];
  const hitSet = new Set(latestHits);

  return (
    <div className="flex items-center gap-2">
      <span className="font-mono text-[10px] uppercase tracking-widest text-slate-500">
        Top-M scan
      </span>
      <div className="flex flex-wrap gap-1">
        <AnimatePresence mode="popLayout">
          {scanned.map((b) => {
            const hit = hitSet.has(b);
            return (
              <motion.span
                key={`${b}`}
                layout
                initial={{ scale: 0.6, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.6, opacity: 0 }}
                className={`rounded px-2 py-0.5 font-mono text-[10px] ${
                  hit
                    ? "animate-pulse-glow bg-cyan/25 text-cyan ring-1 ring-cyan/60"
                    : "bg-white/5 text-slate-300 ring-1 ring-white/10"
                }`}
              >
                {bandInfo[b]?.label ?? `B${b}`}
                {hit ? " ✓" : ""}
              </motion.span>
            );
          })}
        </AnimatePresence>
      </div>
    </div>
  );
}
