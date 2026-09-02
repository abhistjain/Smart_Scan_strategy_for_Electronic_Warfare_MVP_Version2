"""Pydantic schemas for the REST/WebSocket API (spec section 7)."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class EmitterMix(BaseModel):
    markov: float = 0.45
    periodic: float = 0.25
    hopper: float = 0.20
    quiet: float = 0.10


class ScenarioCreate(BaseModel):
    data_source: Literal["synthetic", "real_tsrd"] = "synthetic"
    n_bands: int = Field(24, ge=4, le=128)
    m: int = Field(3, ge=1, le=32)
    seed: int = 0
    p_miss: float = Field(0.1, ge=0.0, le=0.9)
    p_fa: float = Field(0.05, ge=0.0, le=0.9)
    emitter_mix: Optional[EmitterMix] = None
    sample_id: int = 0
    r_hit: float = 1.0
    c_dwell: float = 0.05
    c_miss_penalty: float = 0.5
    beta: float = Field(0.99, gt=0.0, lt=1.0)
    ucb_c: float = Field(0.05, ge=0.0)
    name: Optional[str] = None


class ScenarioInfo(BaseModel):
    scenario_id: str
    config: dict
    status: str
    tick: int
    n_bands: int
    m: int
    band_info: list[dict]


class TSRDSample(BaseModel):
    sample_id: Optional[int]
    name: str
    n_bands: Optional[int]
    n_slots: Optional[int]
    slot_us: Optional[float]
    duration_s: Optional[float]
    emitter_count: int
    subband_mhz: Optional[list]
    stub: bool
    aoa_available: bool


class SpeedUpdate(BaseModel):
    ticks_per_sec: float = Field(8.0, gt=0.0, le=200.0)


class MetricsSummary(BaseModel):
    scenario_id: str
    tick: int
    metrics: dict
