"""FastAPI server: REST + WebSocket (spec section 7).

Endpoints
    POST   /api/scenario                     create scenario -> {scenario_id}
    GET    /api/scenario/{id}                 fetch config + band info
    POST   /api/scenario/{id}/start          begin/resume ticking
    POST   /api/scenario/{id}/pause          pause ticking
    POST   /api/scenario/{id}/reset          reset to t=0
    POST   /api/scenario/{id}/speed          set tick rate
    GET    /api/scenario/{id}/metrics/summary cumulative snapshot
    GET    /api/scenario/{id}/export         CSV/JSON run history
    GET    /api/data/tsrd/samples            list cached TSRD samples
    WS     /ws/scenario/{id}                  one JSON tick per timestep
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import uuid
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

# Load the repo-root .env (and backend/.env if present) BEFORE importing modules
# that read environment variables. uvicorn does not load .env files itself, so
# without this ANTHROPIC_API_KEY / ANTHROPIC_MODEL / HF_TOKEN stay unset.
_HERE = Path(__file__).resolve()
load_dotenv(_HERE.parents[2] / ".env")  # F:\DRDO-SIH\.env
load_dotenv(_HERE.parents[1] / ".env")  # F:\DRDO-SIH\backend\.env (optional override)

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from api.schemas import (
    ChatBody,
    NarrateBandBody,
    PriorityWeightsBody,
    ScenarioCreate,
    SpeedUpdate,
    SummarizeRunBody,
)
from api.simulation import ScenarioConfig, Simulation
from ai_analyst.claude_client import get_client
from ai_analyst.prompts import (
    SCOPE_SYSTEM,
    build_chat_prompt,
    build_narrate_prompt,
    build_summary_prompt,
    fallback_narration,
)
from sim.tsrd_environment import list_cached_samples

app = FastAPI(title="Smart Scan Strategy for EW - ES Receiver Scheduler", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScenarioRunner:
    """Holds one Simulation plus its live-streaming state."""

    def __init__(self, sim: Simulation, name: str) -> None:
        self.sim = sim
        self.name = name
        self.status = "paused"  # paused | running | finished
        self.ticks_per_sec = 8.0
        self.clients: set[WebSocket] = set()
        self._task: asyncio.Task | None = None
        self._latest: dict | None = None
        # Band narration cache keyed by band -> (label, text). Regenerated only
        # when a band's classification label changes (debounce, Section 3.2a).
        self.narration_cache: dict[int, tuple[str, str]] = {}

    async def _loop(self) -> None:
        try:
            while self.status == "running":
                payload = self.sim.step()
                self._latest = payload
                await self._broadcast(payload)
                await asyncio.sleep(1.0 / max(self.ticks_per_sec, 0.1))
        except asyncio.CancelledError:  # pragma: no cover
            pass

    async def _broadcast(self, payload: dict) -> None:
        dead = []
        msg = json.dumps(payload)
        for ws in list(self.clients):
            try:
                await ws.send_text(msg)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    def start(self) -> None:
        if self.status == "running":
            return
        self.status = "running"
        self._task = asyncio.create_task(self._loop())

    def pause(self) -> None:
        self.status = "paused"
        if self._task:
            self._task.cancel()
            self._task = None

    def reset(self) -> None:
        self.pause()
        self.sim.reset()
        self._latest = None


SCENARIOS: Dict[str, ScenarioRunner] = {}


def _get(scenario_id: str) -> ScenarioRunner:
    runner = SCENARIOS.get(scenario_id)
    if runner is None:
        raise HTTPException(status_code=404, detail="scenario not found")
    return runner


# --------------------------------------------------------------------------- #
# REST
# --------------------------------------------------------------------------- #


@app.post("/api/scenario")
def create_scenario(body: ScenarioCreate) -> dict:
    cfg = ScenarioConfig(
        data_source=body.data_source,
        n_bands=body.n_bands,
        m=body.m,
        seed=body.seed,
        p_miss=body.p_miss,
        p_fa=body.p_fa,
        emitter_mix=body.emitter_mix.model_dump() if body.emitter_mix else None,
        sample_id=body.sample_id,
        r_hit=body.r_hit,
        c_dwell=body.c_dwell,
        c_miss_penalty=body.c_miss_penalty,
        beta=body.beta,
        ucb_c=body.ucb_c,
    )
    try:
        sim = Simulation(cfg)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    scenario_id = uuid.uuid4().hex[:12]
    name = body.name or f"{body.data_source}-{scenario_id[:6]}"
    SCENARIOS[scenario_id] = ScenarioRunner(sim, name)
    return {
        "scenario_id": scenario_id,
        "name": name,
        "n_bands": sim.n_bands,
        "m": sim.m,
        "band_info": sim.band_info(),
        "config": cfg.to_dict(),
    }


@app.get("/api/scenario/{scenario_id}")
def get_scenario(scenario_id: str) -> dict:
    runner = _get(scenario_id)
    return {
        "scenario_id": scenario_id,
        "name": runner.name,
        "status": runner.status,
        "tick": runner.sim.t,
        "n_bands": runner.sim.n_bands,
        "m": runner.sim.m,
        "config": runner.sim.cfg.to_dict(),
        "band_info": runner.sim.band_info(),
    }


@app.post("/api/scenario/{scenario_id}/start")
async def start_scenario(scenario_id: str) -> dict:
    # async so asyncio.create_task() runs on the live event loop.
    runner = _get(scenario_id)
    runner.start()
    return {"status": runner.status}


@app.post("/api/scenario/{scenario_id}/pause")
async def pause_scenario(scenario_id: str) -> dict:
    runner = _get(scenario_id)
    runner.pause()
    return {"status": runner.status}


@app.post("/api/scenario/{scenario_id}/reset")
async def reset_scenario(scenario_id: str) -> dict:
    runner = _get(scenario_id)
    runner.reset()
    return {"status": runner.status, "tick": runner.sim.t}


@app.post("/api/scenario/{scenario_id}/speed")
def set_speed(scenario_id: str, body: SpeedUpdate) -> dict:
    runner = _get(scenario_id)
    runner.ticks_per_sec = body.ticks_per_sec
    return {"ticks_per_sec": runner.ticks_per_sec}


@app.get("/api/scenario/{scenario_id}/metrics/summary")
def metrics_summary(scenario_id: str) -> dict:
    runner = _get(scenario_id)
    return {
        "scenario_id": scenario_id,
        "tick": runner.sim.t,
        "metrics": runner.sim.metrics_summary(),
    }


@app.get("/api/scenario/{scenario_id}/export", response_model=None)
def export_scenario(scenario_id: str, fmt: str = "json"):
    runner = _get(scenario_id)
    history = runner.sim.history
    if fmt == "csv":
        buf = io.StringIO()
        if history:
            keys = ["t"]
            for strat in runner.sim.STRATEGY_KEYS:
                for mkey in history[0][strat].keys():
                    keys.append(f"{strat}.{mkey}")
            writer = csv.writer(buf)
            writer.writerow(keys)
            for row in history:
                out = [row["t"]]
                for strat in runner.sim.STRATEGY_KEYS:
                    out.extend(row[strat].values())
                writer.writerow(out)
        buf.seek(0)
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=scenario_{scenario_id}.csv"},
        )
    return JSONResponse({"scenario_id": scenario_id, "history": history})


@app.get("/api/data/tsrd/samples")
def tsrd_samples() -> dict:
    return {"samples": list_cached_samples()}


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "scenarios": len(SCENARIOS)}


# --------------------------------------------------------------------------- #
# v3 add-on: classification + priority
# --------------------------------------------------------------------------- #


@app.get("/api/scenario/{scenario_id}/band/{band}")
def band_detail(scenario_id: str, band: int) -> dict:
    runner = _get(scenario_id)
    if band < 0 or band >= runner.sim.n_bands:
        raise HTTPException(status_code=404, detail="band out of range")
    return runner.sim.band_detail(band)


@app.post("/api/scenario/{scenario_id}/priority-weights")
def set_priority_weights(scenario_id: str, body: PriorityWeightsBody) -> dict:
    runner = _get(scenario_id)
    runner.sim.set_priority_weights(body.w_belief, body.w_conf, body.w_urgency)
    return {"ok": True, "weights": body.model_dump()}


# --------------------------------------------------------------------------- #
# v3 add-on: AI analyst (all endpoints degrade gracefully without an API key)
# --------------------------------------------------------------------------- #


@app.get("/api/ai/status")
def ai_status() -> dict:
    return get_client().status()


@app.post("/api/ai/narrate-band")
def ai_narrate_band(body: NarrateBandBody) -> dict:
    runner = _get(body.scenario_id)
    if body.band < 0 or body.band >= runner.sim.n_bands:
        raise HTTPException(status_code=404, detail="band out of range")
    detail = runner.sim.band_detail(body.band)
    label = detail.get("classification", {}).get("label", "")

    # Debounce: reuse cached narration unless the label changed or force=True.
    cached = runner.narration_cache.get(body.band)
    if cached and cached[0] == label and not body.force:
        return {"available": True, "text": cached[1], "cached": True, "detail": detail}

    result = get_client().generate(
        build_narrate_prompt(detail), max_tokens=180, system=SCOPE_SYSTEM
    )
    text = result.text if result.available else fallback_narration(detail)
    if result.available:
        runner.narration_cache[body.band] = (label, text)
    return {
        "available": True,
        "text": text,
        "cached": False,
        "error": result.error,
        "source": "claude" if result.available else "local",
        "detail": detail,
    }


@app.post("/api/ai/chat")
def ai_chat(body: ChatBody) -> dict:
    runner = _get(body.scenario_id)
    snapshot = {
        "tick": runner.sim.t,
        "data_source": runner.sim.cfg.data_source,
        "aggregate_metrics": runner.sim.metrics_summary(),
        "top_bands_by_priority": [
            {
                "band": d["band"],
                "label": d.get("label"),
                "behaviour": d["classification"]["label"],
                "confidence": d["classification"]["confidence"],
                "priority": d["priority"],
                "belief": d["belief"],
            }
            for d in runner.sim.top_bands(body.top_n)
        ],
    }
    result = get_client().generate(
        build_chat_prompt(body.question, snapshot),
        max_tokens=500,
        system=SCOPE_SYSTEM,
    )
    return {"available": result.available, "text": result.text, "error": result.error}


@app.post("/api/ai/summarize-run")
def ai_summarize_run(body: SummarizeRunBody) -> dict:
    runner = _get(body.scenario_id)
    metrics = runner.sim.metrics_summary()
    classifications = runner.sim.top_bands(8)
    periodicity = runner.sim.strategies["smart"].periodicity_summary()
    result = get_client().generate(
        build_summary_prompt(metrics, classifications, periodicity),
        max_tokens=700,
        system=SCOPE_SYSTEM,
    )
    return {"available": result.available, "text": result.text, "error": result.error}


# --------------------------------------------------------------------------- #
# WebSocket
# --------------------------------------------------------------------------- #


@app.websocket("/ws/scenario/{scenario_id}")
async def ws_scenario(websocket: WebSocket, scenario_id: str) -> None:
    await websocket.accept()
    runner = SCENARIOS.get(scenario_id)
    if runner is None:
        await websocket.send_text(json.dumps({"error": "scenario not found"}))
        await websocket.close()
        return

    runner.clients.add(websocket)
    # Send an initial snapshot so a late-joining client has context immediately.
    await websocket.send_text(
        json.dumps(
            {
                "type": "init",
                "scenario_id": scenario_id,
                "n_bands": runner.sim.n_bands,
                "m": runner.sim.m,
                "band_info": runner.sim.band_info(),
                "config": runner.sim.cfg.to_dict(),
                "tick": runner.sim.t,
                "status": runner.status,
            }
        )
    )
    if runner._latest is not None:
        await websocket.send_text(json.dumps(runner._latest))

    try:
        while True:
            # We don't require inbound messages, but keep the socket alive and
            # allow simple control commands over WS too.
            msg = await websocket.receive_text()
            try:
                cmd = json.loads(msg)
            except json.JSONDecodeError:
                continue
            action = cmd.get("action")
            if action == "start":
                runner.start()
            elif action == "pause":
                runner.pause()
            elif action == "reset":
                runner.reset()
            elif action == "speed":
                runner.ticks_per_sec = float(cmd.get("ticks_per_sec", runner.ticks_per_sec))
    except WebSocketDisconnect:
        runner.clients.discard(websocket)
    except Exception:  # noqa: BLE001
        runner.clients.discard(websocket)
