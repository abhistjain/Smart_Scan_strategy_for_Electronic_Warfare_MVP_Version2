"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useStore } from "@/lib/store";
import { WSClient } from "@/lib/ws-client";
import { startScenario } from "@/lib/api";
import type { DataSource } from "@/lib/types";

import ScenarioControls from "@/components/ScenarioControls";
import SpectrumHeatmap from "@/components/SpectrumHeatmap";
import BeliefBarPanel from "@/components/BeliefBarPanel";
import MetricsRow from "@/components/MetricsRow";
import ComparisonChart from "@/components/ComparisonChart";
import PeriodicityRadar from "@/components/PeriodicityRadar";
import TopMHighlightOverlay from "@/components/TopMHighlightOverlay";
import DatasetAttributionFooter from "@/components/DatasetAttributionFooter";
import AboutDrawer from "@/components/AboutDrawer";
import AiAnalystPanel from "@/components/AiAnalystPanel";
import BandDetailPopover from "@/components/BandDetailPopover";
import EndOfRunModal from "@/components/EndOfRunModal";
import ClassificationDisclaimer from "@/components/ClassificationDisclaimer";

export default function DashboardPage() {
  const router = useRouter();
  const setInit = useStore((s) => s.setInit);
  const pushTick = useStore((s) => s.pushTick);
  const setConnected = useStore((s) => s.setConnected);
  const setScenario = useStore((s) => s.setScenario);
  const highContrast = useStore((s) => s.highContrast);
  const soundOn = useStore((s) => s.soundOn);
  const latestHits = useStore((s) => s.latestHits);
  const tick = useStore((s) => s.tick);

  const [aboutOpen, setAboutOpen] = useState(false);
  const [aiOpen, setAiOpen] = useState(false);
  const [reportOpen, setReportOpen] = useState(false);
  const wsRef = useRef<WSClient | null>(null);
  const audioCtx = useRef<AudioContext | null>(null);

  // Connect WebSocket once, using the scenario id from the URL.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const id = params.get("id");
    const src = (params.get("src") as DataSource) || "synthetic";
    if (!id) {
      router.replace("/");
      return;
    }
    setScenario(id, src);

    const ws = new WSClient(id, {
      onInit: (p) => setInit(p),
      onTick: (p) => pushTick(p),
      onStatus: (c) => setConnected(c),
    });
    ws.connect();
    wsRef.current = ws;

    // Auto-start for a friction-free demo hand-off from the setup screen.
    startScenario(id)
      .then(() => useStore.setState({ status: "running" }))
      .catch(() => {});

    return () => ws.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Soft "blip" on a hit (off by default).
  useEffect(() => {
    if (!soundOn || latestHits.length === 0) return;
    try {
      audioCtx.current ??= new (window.AudioContext ||
        (window as any).webkitAudioContext)();
      const ctx = audioCtx.current;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.frequency.value = 880;
      gain.gain.value = 0.04;
      osc.connect(gain).connect(ctx.destination);
      osc.start();
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.12);
      osc.stop(ctx.currentTime + 0.13);
    } catch {
      /* audio not available */
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick]);

  return (
    <main className={`grid-bg min-h-screen p-3 pb-8 ${highContrast ? "high-contrast" : ""}`}>
      <div className="mx-auto flex max-w-[1800px] flex-col gap-3">
        <ScenarioControls onAbout={() => setAboutOpen(true)} />

        {/* Add-on quick actions: AI analyst + end-of-run report */}
        <div className="flex items-center justify-end gap-2">
          <button
            onClick={() => setReportOpen(true)}
            className="rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 font-mono text-xs text-slate-300 hover:border-cyan/40 hover:text-cyan"
          >
            📄 Run Report
          </button>
          <button
            onClick={() => setAiOpen(true)}
            className="rounded-lg border border-cyan/30 bg-cyan/10 px-3 py-1.5 font-mono text-xs text-cyan hover:bg-cyan/20"
          >
            ✦ AI Analyst
          </button>
        </div>

        <div className="grid grid-cols-1 gap-3 xl:grid-cols-[1fr_320px]">
          {/* Hero: spectrum waterfall */}
          <section className="glass flex flex-col rounded-xl p-3">
            <div className="mb-2 flex items-center justify-between">
              <h2 className="font-mono text-xs uppercase tracking-widest text-slate-400">
                Spectrum Waterfall · belief heatmap
              </h2>
              <TopMHighlightOverlay />
            </div>
            <div className="h-[360px] w-full">
              <SpectrumHeatmap />
            </div>
          </section>

          {/* Right: live belief panel */}
          <aside className="glass rounded-xl p-3 xl:h-[404px]">
            <BeliefBarPanel />
          </aside>
        </div>

        {/* Metrics row */}
        <MetricsRow />

        {/* Comparison + periodicity */}
        <div className="grid grid-cols-1 gap-3 xl:grid-cols-[1fr_320px]">
          <section className="glass rounded-xl p-3">
            <ComparisonChart />
          </section>
          <aside className="glass rounded-xl p-3">
            <PeriodicityRadar />
          </aside>
        </div>

        <DatasetAttributionFooter />
      </div>

      <AboutDrawer open={aboutOpen} onClose={() => setAboutOpen(false)} />
      <AiAnalystPanel open={aiOpen} onClose={() => setAiOpen(false)} />
      <EndOfRunModal open={reportOpen} onClose={() => setReportOpen(false)} />
      <BandDetailPopover />
      <ClassificationDisclaimer variant="footer" />
    </main>
  );
}
