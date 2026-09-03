"use client";

import dynamic from "next/dynamic";
import { useMemo } from "react";
import { useStore } from "@/lib/store";

const Plot = dynamic(() => import("./Plot"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full items-center justify-center text-slate-500">
      initializing spectrum…
    </div>
  ),
});

// Belief colorscale: idle slate-blue -> amber (uncertain) -> electric cyan (active).
const BELIEF_SCALE: [number, string][] = [
  [0.0, "#0A0E17"],
  [0.25, "#1E293B"],
  [0.5, "#3B4A6B"],
  [0.7, "#F5A623"],
  [1.0, "#22D3EE"],
];

export default function SpectrumHeatmap() {
  const beliefWindow = useStore((s) => s.beliefWindow);
  const truthWindow = useStore((s) => s.truthWindow);
  const scannedWindow = useStore((s) => s.scannedWindow);
  const nBands = useStore((s) => s.nBands);
  const bandInfo = useStore((s) => s.bandInfo);
  const tick = useStore((s) => s.tick);
  const instructorMode = useStore((s) => s.instructorMode);

  const { data, layout } = useMemo(() => {
    const T = beliefWindow.length;
    // Transpose: Plotly heatmap z is [row=band][col=time].
    const z: number[][] = [];
    for (let b = 0; b < nBands; b++) {
      const row: number[] = [];
      for (let t = 0; t < T; t++) row.push(beliefWindow[t]?.[b] ?? 0);
      z.push(row);
    }
    const xTicks = beliefWindow.map((_, i) => tick - (T - 1 - i));
    const yLabels = bandInfo.map((b) => b.label);

    const traces: any[] = [
      {
        z,
        x: xTicks,
        y: yLabels.length === nBands ? yLabels : undefined,
        type: "heatmap",
        colorscale: BELIEF_SCALE,
        zmin: 0,
        zmax: 1,
        colorbar: {
          title: { text: "P(active)", font: { color: "#94a3b8", size: 10 } },
          tickfont: { color: "#94a3b8", size: 9 },
          thickness: 10,
          len: 0.9,
        },
        hovertemplate: "band %{y}<br>t=%{x}<br>belief=%{z:.2f}<extra></extra>",
      },
    ];

    // Current Top-M scanned cells (last column) as glowing markers.
    if (scannedWindow.length > 0) {
      const lastScanned = scannedWindow[scannedWindow.length - 1] ?? [];
      traces.push({
        x: lastScanned.map(() => tick),
        y: lastScanned.map((b) => yLabels[b] ?? b),
        mode: "markers",
        type: "scatter",
        marker: { size: 14, color: "rgba(34,211,238,0.0)", line: { color: "#22D3EE", width: 2 }, symbol: "square" },
        hoverinfo: "skip",
        showlegend: false,
      });
    }

    // Ground-truth tick marks (Instructor Mode only).
    if (instructorMode && truthWindow.length > 0) {
      const xs: number[] = [];
      const ys: (string | number)[] = [];
      truthWindow.forEach((truth, ti) => {
        const tx = tick - (truthWindow.length - 1 - ti);
        truth.forEach((v, b) => {
          if (v === 1) {
            xs.push(tx);
            ys.push(yLabels[b] ?? b);
          }
        });
      });
      traces.push({
        x: xs,
        y: ys,
        mode: "markers",
        type: "scatter",
        marker: { size: 3, color: "rgba(239,68,68,0.55)", symbol: "diamond" },
        name: "ground truth",
        hoverinfo: "skip",
        showlegend: false,
      });
    }

    const layout: any = {
      margin: { l: 64, r: 10, t: 10, b: 36 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#94a3b8", family: "JetBrains Mono, monospace", size: 10 },
      xaxis: { title: "", showgrid: false, zeroline: false, color: "#64748b" },
      yaxis: { showgrid: false, zeroline: false, autorange: "reversed", color: "#64748b" },
      datarevision: tick, // efficient incremental update instead of full re-render
      showlegend: false,
      uirevision: "spectrum",
    };
    return { data: traces, layout };
  }, [beliefWindow, truthWindow, scannedWindow, nBands, bandInfo, tick, instructorMode]);

  return (
    <Plot
      data={data}
      layout={layout}
      revision={tick}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: "100%", height: "100%" }}
      useResizeHandler
    />
  );
}
