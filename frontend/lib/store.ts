import { create } from "zustand";
import type {
  BandInfo,
  DataSource,
  InitPayload,
  PeriodicityEstimate,
  StrategyKey,
  StrategyMetrics,
  TickPayload,
} from "./types";

const HEATMAP_WINDOW = 200; // ticks shown in the waterfall
const CHART_WINDOW = 600; // ticks kept for comparison charts

const EMPTY_METRICS: StrategyMetrics = {
  pd: 0,
  pfa: 0,
  sensitivity: 0,
  intercept_rate: 0,
  avg_reward: 0,
  pct_correct: 0,
  time_error: null,
  total_intercepts: 0,
  ticks: 0,
};

const STRATS: StrategyKey[] = ["smart", "sequential", "random", "greedy"];

interface DashboardState {
  scenarioId: string | null;
  dataSource: DataSource;
  nBands: number;
  m: number;
  bandInfo: BandInfo[];
  config: Record<string, unknown>;

  connected: boolean;
  status: string;
  tick: number;

  // Heatmap window (each entry is a belief vector of length nBands).
  beliefWindow: number[][];
  truthWindow: number[][];
  scannedWindow: number[][]; // smart Top-M bands per tick

  // Metrics + history.
  metrics: Record<StrategyKey, StrategyMetrics>;
  rateHistory: Record<StrategyKey, number[]>;
  tickHistory: number[];

  periodicity: PeriodicityEstimate[];
  latestHits: number[]; // smart hits this tick (for flash markers)

  // UI toggles.
  instructorMode: boolean;
  highContrast: boolean;
  soundOn: boolean;
  speed: number;

  setInit: (p: InitPayload) => void;
  pushTick: (p: TickPayload) => void;
  setConnected: (c: boolean) => void;
  setScenario: (id: string, source: DataSource) => void;
  resetLive: () => void;
  toggle: (key: "instructorMode" | "highContrast" | "soundOn") => void;
  setSpeed: (s: number) => void;
}

function emptyRates(): Record<StrategyKey, number[]> {
  return { smart: [], sequential: [], random: [], greedy: [] };
}
function emptyMetrics(): Record<StrategyKey, StrategyMetrics> {
  return {
    smart: { ...EMPTY_METRICS },
    sequential: { ...EMPTY_METRICS },
    random: { ...EMPTY_METRICS },
    greedy: { ...EMPTY_METRICS },
  };
}

export const useStore = create<DashboardState>((set) => ({
  scenarioId: null,
  dataSource: "synthetic",
  nBands: 24,
  m: 3,
  bandInfo: [],
  config: {},

  connected: false,
  status: "paused",
  tick: 0,

  beliefWindow: [],
  truthWindow: [],
  scannedWindow: [],

  metrics: emptyMetrics(),
  rateHistory: emptyRates(),
  tickHistory: [],

  periodicity: [],
  latestHits: [],

  instructorMode: false,
  highContrast: false,
  soundOn: false,
  speed: 8,

  setInit: (p) =>
    set({
      nBands: p.n_bands,
      m: p.m,
      bandInfo: p.band_info,
      config: p.config,
      status: p.status,
      tick: p.tick,
    }),

  pushTick: (p) =>
    set((state) => {
      const smart = p.strategies.smart;
      const beliefs = smart.beliefs ?? new Array(state.nBands).fill(0);

      const beliefWindow = [...state.beliefWindow, beliefs].slice(-HEATMAP_WINDOW);
      const truthWindow = [...state.truthWindow, p.ground_truth].slice(-HEATMAP_WINDOW);
      const scannedWindow = [...state.scannedWindow, smart.scanned_bands].slice(
        -HEATMAP_WINDOW,
      );

      const metrics = { ...state.metrics };
      const rateHistory = { ...state.rateHistory };
      for (const k of STRATS) {
        const st = p.strategies[k];
        if (st) {
          metrics[k] = st.metrics;
          rateHistory[k] = [...rateHistory[k], st.metrics.intercept_rate].slice(
            -CHART_WINDOW,
          );
        }
      }
      const tickHistory = [...state.tickHistory, p.t].slice(-CHART_WINDOW);

      return {
        tick: p.t,
        beliefWindow,
        truthWindow,
        scannedWindow,
        metrics,
        rateHistory,
        tickHistory,
        periodicity: p.periodicity,
        latestHits: smart.hits,
        // NB: status is driven by explicit control actions (start/pause/reset),
        // not by tick messages, so the indicator can't be flipped back to
        // "running" by an in-flight tick that arrives just after a pause.
      };
    }),

  setConnected: (c) => set({ connected: c }),
  setScenario: (id, source) => set({ scenarioId: id, dataSource: source }),

  resetLive: () =>
    set({
      beliefWindow: [],
      truthWindow: [],
      scannedWindow: [],
      metrics: emptyMetrics(),
      rateHistory: emptyRates(),
      tickHistory: [],
      periodicity: [],
      latestHits: [],
      tick: 0,
    }),

  toggle: (key) => set((s) => ({ [key]: !s[key] }) as Partial<DashboardState>),
  setSpeed: (s) => set({ speed: s }),
}));
