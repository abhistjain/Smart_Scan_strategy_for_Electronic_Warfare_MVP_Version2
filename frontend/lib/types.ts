export type DataSource = "synthetic" | "real_tsrd";

export type StrategyKey = "smart" | "sequential" | "random" | "greedy";

export interface StrategyMetrics {
  pd: number;
  pfa: number;
  sensitivity: number;
  intercept_rate: number;
  avg_reward: number;
  pct_correct: number;
  time_error: number | null;
  total_intercepts: number;
  ticks: number;
}

export interface StrategyTick {
  scanned_bands: number[];
  hits: number[];
  metrics: StrategyMetrics;
  beliefs?: number[];
  index?: number[];
}

export interface PeriodicityEstimate {
  band: number;
  period: number;
  phase_mean: number;
  kappa: number;
  confidence: number;
  next_active_tick: number;
  power: number;
}

export interface TickPayload {
  t: number;
  data_source: DataSource;
  strategies: Record<StrategyKey, StrategyTick>;
  ground_truth: number[];
  periodicity: PeriodicityEstimate[];
}

export interface BandInfo {
  index: number;
  label: string;
  emitter_type: string;
  stats: Record<string, unknown>;
}

export interface InitPayload {
  type: "init";
  scenario_id: string;
  n_bands: number;
  m: number;
  band_info: BandInfo[];
  config: Record<string, unknown>;
  tick: number;
  status: string;
}

export interface TSRDSample {
  sample_id: number | null;
  name: string;
  n_bands: number | null;
  n_slots: number | null;
  slot_us: number | null;
  duration_s: number | null;
  emitter_count: number;
  subband_mhz: number[] | null;
  stub: boolean;
  aoa_available: boolean;
}

export interface ScenarioCreateBody {
  data_source: DataSource;
  n_bands: number;
  m: number;
  seed: number;
  p_miss: number;
  p_fa: number;
  emitter_mix?: {
    markov: number;
    periodic: number;
    hopper: number;
    quiet: number;
  };
  sample_id?: number;
  r_hit: number;
  c_dwell: number;
  c_miss_penalty: number;
  beta: number;
  ucb_c: number;
  name?: string;
}

export const STRATEGY_LABELS: Record<StrategyKey, string> = {
  smart: "Smart Scheduler",
  sequential: "Sequential Sweep",
  random: "Random",
  greedy: "Greedy Recent-Hit",
};

export const STRATEGY_COLORS: Record<StrategyKey, string> = {
  smart: "#22D3EE",
  sequential: "#F5A623",
  random: "#A78BFA",
  greedy: "#F472B6",
};
