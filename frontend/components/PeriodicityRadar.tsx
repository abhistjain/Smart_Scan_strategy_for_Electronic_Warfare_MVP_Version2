"use client";

import dynamic from "next/dynamic";
import { useMemo } from "react";
import { useStore } from "@/lib/store";

const Plot = dynamic(() => import("./Plot"), { ssr: false });

// Polar view of the von Mises fits for bands with a detected period (spec 6.6,
// 9.2). Each detected periodic emitter is drawn as a wedge at its expected
// active phase, with radius = confidence.
export default function PeriodicityRadar() {
  const periodicity = useStore((s) => s.periodicity);
  const bandInfo = useStore((s) => s.bandInfo);
  const tick = useStore((s) => s.tick);

  const { data, layout } = useMemo(() => {
    const detected = periodicity.filter((p) => p.confidence >= 0.3);
    const traces: any[] = detected.map((p) => {
      const theta = (p.phase_mean / Math.max(p.period, 1e-6)) * 360;
      // Spread proportional to phase concentration (higher kappa -> narrower).
      const spread = Math.max(6, 40 / (1 + p.kappa));
      return {
        type: "scatterpolar",
        mode: "lines",
        r: [0, p.confidence, p.confidence, 0],
        theta: [0, theta - spread, theta + spread, 0],
        fill: "toself",
        fillcolor: "rgba(34,211,238,0.25)",
        line: { color: "#22D3EE", width: 1 },
        name: bandInfo[p.band]?.label ?? `B${p.band}`,
        hovertemplate: `${bandInfo[p.band]?.label ?? "B" + p.band}<br>period=${p.period.toFixed(1)}<br>conf=${p.confidence.toFixed(2)}<extra></extra>`,
      };
    });

    const layout: any = {
      margin: { l: 24, r: 24, t: 16, b: 16 },
      paper_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#94a3b8", family: "JetBrains Mono, monospace", size: 8 },
      polar: {
        bgcolor: "rgba(10,14,23,0.4)",
        radialaxis: { visible: true, range: [0, 1], color: "#475569", gridcolor: "rgba(255,255,255,0.06)" },
        angularaxis: { color: "#475569", gridcolor: "rgba(255,255,255,0.06)", rotation: 90, direction: "clockwise" },
      },
      showlegend: false,
      datarevision: tick,
      uirevision: "radar",
    };
    return { data: traces, layout };
  }, [periodicity, bandInfo, tick]);

  return (
    <div className="flex h-full flex-col">
      <h3 className="mb-1 font-mono text-xs uppercase tracking-widest text-slate-400">
        Periodicity Radar
      </h3>
      {data.length === 0 ? (
        <div className="flex flex-1 items-center justify-center text-center font-mono text-[10px] text-slate-600">
          no periodic emitter
          <br />
          detected yet…
        </div>
      ) : (
        <Plot
          data={data}
          layout={layout}
          revision={tick}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: "100%", height: "100%", minHeight: "160px" }}
          useResizeHandler
        />
      )}
    </div>
  );
}
