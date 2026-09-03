import type {
  AIStatus,
  BandDetail,
  ChatResponse,
  NarrateResponse,
  ScenarioCreateBody,
  TSRDSample,
} from "./types";

// All REST calls go through Next.js rewrites (/api/* -> backend), so the
// browser only ever talks to same-origin.
const BASE = "";

export async function createScenario(body: ScenarioCreateBody) {
  const res = await fetch(`${BASE}/api/scenario`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`createScenario failed: ${res.status} ${await res.text()}`);
  return res.json() as Promise<{
    scenario_id: string;
    name: string;
    n_bands: number;
    m: number;
    band_info: import("./types").BandInfo[];
    config: Record<string, unknown>;
  }>;
}

export async function startScenario(id: string) {
  await fetch(`${BASE}/api/scenario/${id}/start`, { method: "POST" });
}
export async function pauseScenario(id: string) {
  await fetch(`${BASE}/api/scenario/${id}/pause`, { method: "POST" });
}
export async function resetScenario(id: string) {
  await fetch(`${BASE}/api/scenario/${id}/reset`, { method: "POST" });
}
export async function setSpeed(id: string, ticks_per_sec: number) {
  await fetch(`${BASE}/api/scenario/${id}/speed`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ticks_per_sec }),
  });
}

export async function fetchTSRDSamples(): Promise<TSRDSample[]> {
  try {
    const res = await fetch(`${BASE}/api/data/tsrd/samples`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.samples ?? [];
  } catch {
    return [];
  }
}

export function exportUrl(id: string, fmt: "json" | "csv") {
  return `${BASE}/api/scenario/${id}/export?fmt=${fmt}`;
}

// --- v3 add-on: classification + AI analyst ------------------------------- //

export async function setPriorityWeights(
  id: string,
  w: { w_belief: number; w_conf: number; w_urgency: number },
) {
  await fetch(`${BASE}/api/scenario/${id}/priority-weights`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(w),
  });
}

export async function fetchBandDetail(id: string, band: number): Promise<BandDetail> {
  const res = await fetch(`${BASE}/api/scenario/${id}/band/${band}`);
  if (!res.ok) throw new Error(`band detail failed: ${res.status}`);
  return res.json() as Promise<BandDetail>;
}

export async function aiStatus(): Promise<AIStatus> {
  try {
    const res = await fetch(`${BASE}/api/ai/status`);
    if (!res.ok) return { available: false, model: null, reason: "unreachable" };
    return res.json() as Promise<AIStatus>;
  } catch {
    return { available: false, model: null, reason: "unreachable" };
  }
}

export async function narrateBand(
  id: string,
  band: number,
  force = false,
): Promise<NarrateResponse> {
  const res = await fetch(`${BASE}/api/ai/narrate-band`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scenario_id: id, band, force }),
  });
  if (!res.ok) throw new Error(`narrate failed: ${res.status}`);
  return res.json() as Promise<NarrateResponse>;
}

export async function aiChat(
  id: string,
  question: string,
  topN = 6,
): Promise<ChatResponse> {
  const res = await fetch(`${BASE}/api/ai/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scenario_id: id, question, top_n: topN }),
  });
  if (!res.ok) throw new Error(`chat failed: ${res.status}`);
  return res.json() as Promise<ChatResponse>;
}

export async function summarizeRun(id: string): Promise<ChatResponse> {
  const res = await fetch(`${BASE}/api/ai/summarize-run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scenario_id: id }),
  });
  if (!res.ok) throw new Error(`summarize failed: ${res.status}`);
  return res.json() as Promise<ChatResponse>;
}
