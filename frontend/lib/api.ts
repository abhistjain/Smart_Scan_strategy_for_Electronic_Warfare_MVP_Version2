import type { ScenarioCreateBody, TSRDSample } from "./types";

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
