"use client";

import { animate } from "framer-motion";
import { useEffect, useRef, useState } from "react";

interface Props {
  label: string;
  value: number | null;
  format?: (v: number) => string;
  suffix?: string;
  accent?: string;
  hint?: string;
}

// Animated numeric counter that eases toward the latest value (spec 9.1).
export default function MetricsCard({
  label,
  value,
  format = (v) => v.toFixed(3),
  suffix = "",
  accent = "#22D3EE",
  hint,
}: Props) {
  const [display, setDisplay] = useState(0);
  const prev = useRef(0);

  useEffect(() => {
    if (value === null) return;
    const controls = animate(prev.current, value, {
      duration: 0.5,
      onUpdate: (v) => setDisplay(v),
    });
    prev.current = value;
    return () => controls.stop();
  }, [value]);

  return (
    <div className="glass group relative flex flex-col rounded-lg px-3 py-2" title={hint}>
      <span className="font-mono text-[9px] uppercase tracking-widest text-slate-500">
        {label}
      </span>
      <span
        className="mt-0.5 font-mono text-xl font-semibold tabular-nums"
        style={{ color: accent }}
      >
        {value === null ? "—" : format(display)}
        <span className="ml-0.5 text-xs text-slate-500">{suffix}</span>
      </span>
      <div
        className="absolute bottom-0 left-0 h-0.5 rounded-full opacity-70"
        style={{ background: accent, width: value === null ? 0 : "100%" }}
      />
    </div>
  );
}
