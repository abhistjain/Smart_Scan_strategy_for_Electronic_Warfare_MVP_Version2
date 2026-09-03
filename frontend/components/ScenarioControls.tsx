"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { useStore } from "@/lib/store";
import {
  pauseScenario,
  resetScenario,
  setSpeed as apiSetSpeed,
  startScenario,
  exportUrl,
} from "@/lib/api";

export default function ScenarioControls({ onAbout }: { onAbout: () => void }) {
  const router = useRouter();
  const scenarioId = useStore((s) => s.scenarioId);
  const dataSource = useStore((s) => s.dataSource);
  const connected = useStore((s) => s.connected);
  const status = useStore((s) => s.status);
  const tick = useStore((s) => s.tick);
  const speed = useStore((s) => s.speed);
  const instructorMode = useStore((s) => s.instructorMode);
  const highContrast = useStore((s) => s.highContrast);
  const soundOn = useStore((s) => s.soundOn);
  const toggle = useStore((s) => s.toggle);
  const setSpeed = useStore((s) => s.setSpeed);
  const resetLive = useStore((s) => s.resetLive);
  const [busy, setBusy] = useState(false);

  const isReal = dataSource === "real_tsrd";

  async function handleStart() {
    if (!scenarioId) return;
    setBusy(true);
    await startScenario(scenarioId);
    useStore.setState({ status: "running" });
    setBusy(false);
  }
  async function handlePause() {
    if (!scenarioId) return;
    await pauseScenario(scenarioId);
    useStore.setState({ status: "paused" });
  }
  async function handleReset() {
    if (!scenarioId) return;
    await resetScenario(scenarioId);
    resetLive();
    useStore.setState({ status: "paused" });
  }
  async function handleSpeed(v: number) {
    setSpeed(v);
    if (scenarioId) await apiSetSpeed(scenarioId, v);
  }

  return (
    <div className="glass-strong flex flex-wrap items-center gap-x-4 gap-y-2 rounded-lg px-4 py-2">
      <button
        onClick={() => router.push("/")}
        className="font-mono text-xs text-slate-400 hover:text-cyan"
      >
        ← setup
      </button>

      <div className="flex items-center gap-2">
        <div className="text-sm font-semibold tracking-tight text-slate-100">
          Smart Scan · EW
        </div>
        <span
          className={`rounded px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider ${
            isReal
              ? "bg-purple-500/20 text-purple-300 ring-1 ring-purple-400/40"
              : "bg-cyan/15 text-cyan ring-1 ring-cyan/40"
          }`}
        >
          {isReal ? "Real · TSRD" : "Synthetic"}
        </span>
      </div>

      {/* live / paused indicator */}
      <div className="flex items-center gap-1.5">
        <motion.span
          className={`h-2 w-2 rounded-full ${
            status === "running" ? "bg-green-400" : "bg-slate-500"
          }`}
          animate={status === "running" ? { opacity: [1, 0.3, 1] } : { opacity: 1 }}
          transition={{ repeat: Infinity, duration: 1.2 }}
        />
        <span className="font-mono text-[10px] uppercase text-slate-400">
          {connected ? status : "connecting…"}
        </span>
      </div>

      <div className="font-mono text-xs text-slate-300">
        t = <span className="text-cyan">{tick}</span>
      </div>

      {/* transport */}
      <div className="flex items-center gap-1">
        <button
          onClick={handleStart}
          disabled={busy || status === "running"}
          className="rounded bg-cyan/20 px-3 py-1 font-mono text-xs text-cyan ring-1 ring-cyan/40 transition hover:bg-cyan/30 disabled:opacity-40"
        >
          ▶ start
        </button>
        <button
          onClick={handlePause}
          className="rounded bg-white/5 px-3 py-1 font-mono text-xs text-slate-300 ring-1 ring-white/10 hover:bg-white/10"
        >
          ❚❚ pause
        </button>
        <button
          onClick={handleReset}
          className="rounded bg-white/5 px-3 py-1 font-mono text-xs text-slate-300 ring-1 ring-white/10 hover:bg-white/10"
        >
          ↺ reset
        </button>
      </div>

      {/* speed */}
      <div className="flex items-center gap-2">
        <span className="font-mono text-[10px] text-slate-500">speed</span>
        <input
          type="range"
          min={1}
          max={30}
          value={speed}
          onChange={(e) => handleSpeed(Number(e.target.value))}
          className="w-24 accent-cyan"
        />
        <span className="w-10 font-mono text-[10px] text-slate-400">{speed}/s</span>
      </div>

      <div className="ml-auto flex items-center gap-2">
        <ToggleChip active={instructorMode} onClick={() => toggle("instructorMode")}>
          instructor
        </ToggleChip>
        <ToggleChip active={highContrast} onClick={() => toggle("highContrast")}>
          contrast
        </ToggleChip>
        <ToggleChip active={soundOn} onClick={() => toggle("soundOn")}>
          {soundOn ? "🔊" : "🔇"} blip
        </ToggleChip>
        {scenarioId && (
          <a
            href={exportUrl(scenarioId, "csv")}
            className="rounded bg-white/5 px-2 py-1 font-mono text-[10px] text-slate-300 ring-1 ring-white/10 hover:bg-white/10"
          >
            ⭳ csv
          </a>
        )}
        <button
          onClick={onAbout}
          className="rounded bg-white/5 px-2 py-1 font-mono text-[10px] text-slate-300 ring-1 ring-white/10 hover:bg-white/10"
        >
          ? about
        </button>
      </div>
    </div>
  );
}

function ToggleChip({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`rounded px-2 py-1 font-mono text-[10px] uppercase tracking-wide transition ${
        active
          ? "bg-cyan/20 text-cyan ring-1 ring-cyan/40"
          : "bg-white/5 text-slate-400 ring-1 ring-white/10 hover:bg-white/10"
      }`}
    >
      {children}
    </button>
  );
}
