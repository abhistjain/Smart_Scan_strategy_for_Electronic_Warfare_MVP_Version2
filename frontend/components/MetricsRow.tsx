"use client";

import { useStore } from "@/lib/store";
import MetricsCard from "./MetricsCard";

// The seven figures of merit required by the problem statement (spec 6.9),
// shown for the Smart Scheduler.
export default function MetricsRow() {
  const m = useStore((s) => s.metrics.smart);

  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-7">
      <MetricsCard label="Pd" value={m.pd} accent="#22D3EE"
        hint="Probability of Detection: intercepted active cells / all active cells" />
      <MetricsCard label="Pfa" value={m.pfa} accent="#F5A623"
        hint="Probability of False Alarm: false alarms / scans of idle bands" />
      <MetricsCard label="Sensitivity" value={m.sensitivity} accent="#A78BFA"
        hint="Detections / scans of truly-active bands (1 - miss rate)" />
      <MetricsCard label="Intercept Rate" value={m.intercept_rate} accent="#22D3EE"
        format={(v) => v.toFixed(2)}
        hint="Successful intercepts per simulated tick" />
      <MetricsCard label="Avg Reward" value={m.avg_reward} accent="#34D399"
        format={(v) => v.toFixed(2)}
        hint="Running mean reward (R_hit - C_dwell - C_miss_penalty)" />
      <MetricsCard label="% Correct" value={m.pct_correct} accent="#F472B6"
        format={(v) => `${(v * 100).toFixed(1)}`} suffix="%"
        hint="Fraction of Top-M scans that matched ground-truth active state" />
      <MetricsCard label="Time Error" value={m.time_error} accent="#FBBF24"
        format={(v) => v.toFixed(1)} suffix="tk"
        hint="Mean abs error between predicted and actual next-active tick (periodic emitters)" />
    </div>
  );
}
