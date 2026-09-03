"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { createScenario, fetchTSRDSamples } from "@/lib/api";
import type { DataSource, ScenarioCreateBody, TSRDSample } from "@/lib/types";
import DrdoLogo from "@/components/DrdoLogo";

type Mix = { markov: number; periodic: number; hopper: number; quiet: number };

interface Preset {
  name: string;
  desc: string;
  cfg: Partial<ScenarioCreateBody> & { emitter_mix: Mix };
}

const SYN_PRESETS: Preset[] = [
  {
    name: "Sparse & Quiet",
    desc: "few emitters, low noise",
    cfg: { n_bands: 24, m: 3, p_miss: 0.08, p_fa: 0.03, emitter_mix: { markov: 0.3, periodic: 0.2, hopper: 0.1, quiet: 0.4 } },
  },
  {
    name: "Dense & Noisy",
    desc: "many emitters, high noise",
    cfg: { n_bands: 32, m: 4, p_miss: 0.15, p_fa: 0.08, emitter_mix: { markov: 0.5, periodic: 0.25, hopper: 0.2, quiet: 0.05 } },
  },
  {
    name: "Heavy Frequency-Hopping",
    desc: "agile hoppers dominate",
    cfg: { n_bands: 28, m: 3, p_miss: 0.12, p_fa: 0.05, emitter_mix: { markov: 0.2, periodic: 0.15, hopper: 0.55, quiet: 0.1 } },
  },
  {
    name: "Periodic Radar Field",
    desc: "rotating-antenna emitters",
    cfg: { n_bands: 24, m: 3, p_miss: 0.1, p_fa: 0.04, emitter_mix: { markov: 0.2, periodic: 0.6, hopper: 0.1, quiet: 0.1 } },
  },
];

export default function SetupPage() {
  const router = useRouter();
  const [dataSource, setDataSource] = useState<DataSource>("synthetic");
  const [samples, setSamples] = useState<TSRDSample[]>([]);
  const [selectedSample, setSelectedSample] = useState<number>(0);
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [nBands, setNBands] = useState(24);
  const [m, setM] = useState(3);
  const [seed, setSeed] = useState(7);
  const [pMiss, setPMiss] = useState(0.1);
  const [pFa, setPFa] = useState(0.05);
  const [mix, setMix] = useState<Mix>({ markov: 0.45, periodic: 0.25, hopper: 0.2, quiet: 0.1 });
  const [presetName, setPresetName] = useState<string>("Dense & Noisy");

  useEffect(() => {
    fetchTSRDSamples().then(setSamples);
  }, []);

  const realAvailable = samples.length > 0;

  function applyPreset(p: Preset) {
    setPresetName(p.name);
    if (p.cfg.n_bands) setNBands(p.cfg.n_bands);
    if (p.cfg.m) setM(p.cfg.m);
    if (p.cfg.p_miss !== undefined) setPMiss(p.cfg.p_miss);
    if (p.cfg.p_fa !== undefined) setPFa(p.cfg.p_fa);
    setMix(p.cfg.emitter_mix);
  }

  async function launch() {
    setLaunching(true);
    setError(null);
    try {
      const body: ScenarioCreateBody = {
        data_source: dataSource,
        n_bands: nBands,
        m,
        seed,
        p_miss: pMiss,
        p_fa: pFa,
        emitter_mix: mix,
        sample_id: selectedSample,
        r_hit: 1.0,
        c_dwell: 0.05,
        c_miss_penalty: 0.5,
        beta: 0.99,
        ucb_c: 0.05,
        name: dataSource === "real_tsrd" ? `TSRD #${selectedSample}` : presetName,
      };
      const res = await createScenario(body);
      router.push(`/dashboard?id=${res.scenario_id}&src=${dataSource}`);
    } catch (e: any) {
      setError(e?.message ?? "failed to create scenario");
      setLaunching(false);
    }
  }

  return (
    <main className="grid-bg min-h-screen">
      {/* radar sweep ambience */}
      <div className="pointer-events-none fixed inset-0 flex items-center justify-center opacity-20">
        <div className="radar-sweep h-[820px] w-[820px] animate-sweep rounded-full" />
      </div>

      <div className="relative mx-auto max-w-5xl px-6 py-10">
        <motion.header
          initial={{ opacity: 0, y: -12 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8 flex items-start gap-4"
        >
          <DrdoLogo size={72} className="mt-1 shadow-[0_0_24px_rgba(34,211,238,0.15)]" />
          <div>
          <div className="font-mono text-[11px] uppercase tracking-[0.3em] text-cyan">
            SIH26055 · Electronic Warfare · DRDO
          </div>
          <h1 className="mt-1 text-3xl font-bold tracking-tight text-slate-100">
            Smart Scan Strategy — ES Receiver Scheduler
          </h1>
          <p className="mt-2 max-w-2xl text-sm text-slate-400">
            An ML-based Electronic Support receiver that learns where and when to
            look, with no prior emitter intelligence — Bayesian belief filtering,
            Thompson sampling, a Whittle-index scheduler, and periodicity-aware
            intercept-ahead tracking.
          </p>
          </div>
        </motion.header>

        {/* data source toggle */}
        <div className="mb-6 inline-flex rounded-lg bg-white/5 p-1 ring-1 ring-white/10">
          <SourceTab active={dataSource === "synthetic"} onClick={() => setDataSource("synthetic")}>
            Synthetic (Controllable)
          </SourceTab>
          <SourceTab
            active={dataSource === "real_tsrd"}
            disabled={!realAvailable}
            onClick={() => realAvailable && setDataSource("real_tsrd")}
          >
            Real Data · TSRD {realAvailable ? "" : "(cache empty)"}
          </SourceTab>
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* left: config */}
          <motion.div
            key={dataSource}
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            className="glass rounded-xl p-5"
          >
            {dataSource === "synthetic" ? (
              <div className="space-y-4">
                <h2 className="font-mono text-xs uppercase tracking-widest text-slate-400">
                  Environment
                </h2>
                <Slider label="Total bands (N)" value={nBands} min={8} max={64} step={1} onChange={setNBands} />
                <Slider label="Receiver capacity (M)" value={m} min={1} max={Math.max(1, Math.floor(nBands / 2))} step={1} onChange={setM} />
                <Slider label="Seed" value={seed} min={0} max={999} step={1} onChange={setSeed} />
                <Slider label="P(miss)" value={pMiss} min={0} max={0.5} step={0.01} onChange={setPMiss} fmt={(v) => v.toFixed(2)} />
                <Slider label="P(false alarm)" value={pFa} min={0} max={0.3} step={0.01} onChange={setPFa} fmt={(v) => v.toFixed(2)} />

                <h3 className="pt-2 font-mono text-xs uppercase tracking-widest text-slate-400">
                  Emitter mix
                </h3>
                {(["markov", "periodic", "hopper", "quiet"] as const).map((k) => (
                  <Slider
                    key={k}
                    label={k}
                    value={mix[k]}
                    min={0}
                    max={1}
                    step={0.05}
                    fmt={(v) => v.toFixed(2)}
                    onChange={(v) => setMix({ ...mix, [k]: v })}
                  />
                ))}
              </div>
            ) : (
              <div className="space-y-3">
                <h2 className="font-mono text-xs uppercase tracking-widest text-slate-400">
                  TSRD Sample
                </h2>
                {samples.length === 0 && (
                  <p className="text-sm text-slate-500">
                    No cached samples. Run{" "}
                    <code className="text-cyan">prepare_tsrd_cache.py</code> to
                    enable Real Data mode.
                  </p>
                )}
                {samples.map((s) => (
                  <button
                    key={s.name}
                    onClick={() => setSelectedSample(s.sample_id ?? 0)}
                    className={`block w-full rounded-lg px-3 py-2 text-left transition ${
                      selectedSample === (s.sample_id ?? 0)
                        ? "bg-purple-500/20 ring-1 ring-purple-400/50"
                        : "bg-white/5 ring-1 ring-white/10 hover:bg-white/10"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-sm text-slate-200">{s.name}</span>
                      {s.stub && (
                        <span className="rounded bg-amber/20 px-1.5 py-0.5 font-mono text-[9px] text-amber">
                          STUB
                        </span>
                      )}
                    </div>
                    <div className="mt-1 font-mono text-[10px] text-slate-500">
                      {s.n_bands} bands · {s.n_slots} slots · {s.duration_s}s ·{" "}
                      {s.emitter_count} emitters
                      {s.subband_mhz ? ` · ${s.subband_mhz[0]}–${s.subband_mhz[1]} MHz` : ""}
                    </div>
                  </button>
                ))}
                <div className="pt-2">
                  <Slider label="P(miss)" value={pMiss} min={0} max={0.5} step={0.01} onChange={setPMiss} fmt={(v) => v.toFixed(2)} />
                  <Slider label="P(false alarm)" value={pFa} min={0} max={0.3} step={0.01} onChange={setPFa} fmt={(v) => v.toFixed(2)} />
                  <Slider label="Receiver capacity (M)" value={m} min={1} max={12} step={1} onChange={setM} />
                </div>
              </div>
            )}
          </motion.div>

          {/* right: presets + launch */}
          <div className="flex flex-col gap-6">
            {dataSource === "synthetic" && (
              <div className="glass rounded-xl p-5">
                <h2 className="mb-3 font-mono text-xs uppercase tracking-widest text-slate-400">
                  Quick Presets
                </h2>
                <div className="grid grid-cols-2 gap-2">
                  {SYN_PRESETS.map((p) => (
                    <button
                      key={p.name}
                      onClick={() => applyPreset(p)}
                      className={`rounded-lg px-3 py-2 text-left transition ${
                        presetName === p.name
                          ? "bg-cyan/20 ring-1 ring-cyan/50"
                          : "bg-white/5 ring-1 ring-white/10 hover:bg-white/10"
                      }`}
                    >
                      <div className="text-sm font-medium text-slate-200">{p.name}</div>
                      <div className="font-mono text-[10px] text-slate-500">{p.desc}</div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            <div className="glass rounded-xl p-5">
              <h2 className="mb-3 font-mono text-xs uppercase tracking-widest text-slate-400">
                Strategies compared
              </h2>
              <ul className="space-y-1.5 text-sm text-slate-300">
                <li><span className="text-cyan">■</span> Smart Scheduler (belief + Whittle + Thompson)</li>
                <li><span className="text-amber">■</span> Sequential Sweep (open-loop incumbent)</li>
                <li><span style={{ color: "#A78BFA" }}>■</span> Random / Round-Robin</li>
                <li><span style={{ color: "#F472B6" }}>■</span> Greedy Recent-Hit</li>
              </ul>
              <p className="mt-3 text-[11px] text-slate-500">
                All four run on the identical trajectory for a fair comparison.
              </p>
            </div>

            {error && (
              <div className="rounded-lg bg-crimson/10 px-4 py-2 text-sm text-red-300 ring-1 ring-red-500/40">
                {error}
              </div>
            )}

            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={launch}
              disabled={launching}
              className="rounded-xl bg-gradient-to-r from-cyan to-teal-400 px-6 py-3 text-center text-base font-semibold text-black shadow-lg shadow-cyan/20 transition disabled:opacity-60"
            >
              {launching ? "launching…" : "Launch Simulation →"}
            </motion.button>
          </div>
        </div>
      </div>
    </main>
  );
}

function SourceTab({
  active,
  disabled,
  onClick,
  children,
}: {
  active: boolean;
  disabled?: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`rounded-md px-4 py-2 font-mono text-xs transition ${
        active
          ? "bg-cyan/20 text-cyan ring-1 ring-cyan/40"
          : "text-slate-400 hover:text-slate-200"
      } ${disabled ? "cursor-not-allowed opacity-40" : ""}`}
    >
      {children}
    </button>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  fmt = (v) => `${v}`,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  fmt?: (v: number) => string;
}) {
  return (
    <div>
      <div className="mb-1 flex items-center justify-between">
        <span className="font-mono text-[11px] capitalize text-slate-400">{label}</span>
        <span className="font-mono text-[11px] text-cyan">{fmt(value)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-cyan"
      />
    </div>
  );
}
