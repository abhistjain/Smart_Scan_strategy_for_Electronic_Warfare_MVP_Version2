"use client";

import dynamic from "next/dynamic";
import { useMemo } from "react";
import { useStore } from "@/lib/store";
import { STRATEGY_COLORS, STRATEGY_LABELS, StrategyKey } from "@/lib/types";

const Plot = dynamic(() => import("./Plot"), { ssr: false });

const STRATS: StrategyKey[] = ["smart", "sequential", "random", "greedy"];

export default function ComparisonChart() {
  const rateHistory = useStore((s) => s.rateHistory);
  const tickHistory = useStore((s) => s.tickHistory);
  const metrics = useStore((s) => s.metrics);
  const tick = useStore((s) => s.tick);

  const lineData = useMemo(() => {
    return STRATS.map((k) => ({
      x: tickHistory,
      y: rateHistory[k],
      type: "scatter" as const,
      mode: "lines" as const,
      name: STRATEGY_LABELS[k],
      line: { color: STRATEGY_COLORS[k], width: k === "smart" ? 2.5 : 1.3 },
    }));
  }, [rateHistory, tickHistory]);

  const barData = useMemo(() => {
    const names = STRATS.map((k) => STRATEGY_LABELS[k]);
    return [
      {
        x: names,
        y: STRATS.map((k) => metrics[k].pd),
        name: "Pd",
        type: "bar" as const,
        marker: { color: "#22D3EE" },
      },
      {
        x: names,
        y: STRATS.map((k) => metrics[k].intercept_rate),
        name: "Intercept Rate",
        type: "bar" as const,
        marker: { color: "#F5A623" },
      },
    ];
  }, [metrics]);

  const baseLayout: any = {
    margin: { l: 44, r: 10, t: 24, b: 30 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#94a3b8", family: "JetBrains Mono, monospace", size: 9 },
    legend: { orientation: "h", font: { size: 9 }, y: 1.18 },
    xaxis: { showgrid: false, color: "#64748b" },
    yaxis: { gridcolor: "rgba(255,255,255,0.05)", color: "#64748b" },
    uirevision: "cmp",
  };

  const smartRate = metrics.smart.intercept_rate;
  const seqRate = metrics.sequential.intercept_rate || 1e-9;
  const ratio = smartRate / seqRate;

  return (
    <div className="flex h-full flex-col gap-2">
      <div className="flex items-center justify-between">
        <h3 className="font-mono text-xs uppercase tracking-widest text-slate-400">
          Strategy Comparison
        </h3>
        <span className="font-mono text-[11px] text-cyan">
          smart / sweep = {ratio.toFixed(2)}×
        </span>
      </div>
      <div className="grid flex-1 grid-cols-1 gap-2 lg:grid-cols-2">
        <div className="glass rounded-lg p-1">
          <div className="px-2 pt-1 font-mono text-[9px] uppercase text-slate-500">
            Cumulative Intercept Rate
          </div>
          <Plot
            data={lineData}
            layout={{ ...baseLayout, datarevision: tick }}
            revision={tick}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%", height: "180px" }}
            useResizeHandler
          />
        </div>
        <div className="glass rounded-lg p-1">
          <div className="px-2 pt-1 font-mono text-[9px] uppercase text-slate-500">
            Final Pd / Intercept Rate (money shot)
          </div>
          <Plot
            data={barData}
            layout={{ ...baseLayout, barmode: "group", datarevision: tick }}
            revision={tick}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%", height: "180px" }}
            useResizeHandler
          />
        </div>
      </div>
    </div>
  );
}
