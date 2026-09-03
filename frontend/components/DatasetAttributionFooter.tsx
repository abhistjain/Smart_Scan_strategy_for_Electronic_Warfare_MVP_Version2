"use client";

import { useStore } from "@/lib/store";

// Visible only in Real Data mode (spec 9.2): TSRD citation, license, and a note
// clarifying that Stare-mode pulse trains provide ground truth while OUR
// scheduler performs the scan.
export default function DatasetAttributionFooter() {
  const dataSource = useStore((s) => s.dataSource);
  if (dataSource !== "real_tsrd") return null;

  return (
    <div className="glass rounded-lg px-4 py-2 text-[10px] leading-relaxed text-slate-400">
      <span className="font-mono uppercase tracking-widest text-purple-300">
        Dataset ·{" "}
      </span>
      Turing Synthetic Radar Dataset (TSRD), Alan Turing Institute — Gunn,
      Hosford, Jones, Zeitler, Groves, Nockles,{" "}
      <em>“The Turing Synthetic Radar Dataset: A dataset for pulse deinterleaving.”</em>{" "}
      License: Apache-2.0.{" "}
      <a
        className="text-cyan hover:underline"
        href="https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset"
        target="_blank"
        rel="noreferrer"
      >
        dataset ↗
      </a>{" "}
      <span className="text-slate-500">
        · Stare-mode pulse trains provide ground-truth occupancy; our ES receiver
        performs the scan and adds its own P_miss / P_fa.
      </span>
    </div>
  );
}
