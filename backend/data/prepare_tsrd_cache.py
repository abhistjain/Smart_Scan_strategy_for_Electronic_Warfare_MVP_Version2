"""Offline one-time TSRD preprocessing (spec section 2.2).

This script turns a small sample of the *Turing Synthetic Radar Dataset*
(TSRD, Alan Turing Institute) Stare-mode pulse trains into compact, demo-safe
cache files that the live simulation replays. The raw dataset is ~70 GB / 4B
pulses and gated behind a Hugging Face account, so we only ever download a
handful of pulse trains and never touch the raw stream at runtime.

Two modes:

    python prepare_tsrd_cache.py --real  --num 5        # gated HF download
    python prepare_tsrd_cache.py --stub  --num 2        # no network, fake PDWs

Both modes run through the *identical* binning pipeline, so the cache format is
the same and the rest of the app cannot tell them apart (except that stub caches
are flagged ``"stub": true`` so the UI can label them honestly).

Outputs, per sample, under backend/data/cache/:
    tsrd_sample_<k>.npz    occupancy grid  (T_slots x N_bands, int8)
    tsrd_sample_<k>.json   band edges, slot duration, per-emitter stats, meta

Also writes backend/data/tsrd_defaults.json: aggregate stat ranges used to
parameterise the synthetic generator's defaults (section 6.1).

HF setup (documented in README): create a free HF account, accept the dataset
access terms at
https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset
then set HF_TOKEN in your environment. The token is read via os.environ and is
never written to disk.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
DEFAULTS_PATH = os.path.join(os.path.dirname(__file__), "tsrd_defaults.json")
HF_DATASET = "alan-turing-institute/turing-synthetic-radar-dataset"

# --------------------------------------------------------------------------- #
# PDW representation
# --------------------------------------------------------------------------- #


@dataclass
class PDWColumns:
    """Column-name mapping for the PDW records. The gated repo layout is not
    publicly documented column-for-column, so this is CLI-overridable."""

    toa_us: str = "toa"
    freq_mhz: str = "frequency"
    pulse_width_us: str = "pulse_width"
    aoa_deg: str = "aoa"
    amplitude_db: str = "amplitude"


# --------------------------------------------------------------------------- #
# Binning pipeline (shared by real + stub)
# --------------------------------------------------------------------------- #


def build_occupancy_grid(
    pdw: dict[str, np.ndarray],
    n_bands: int,
    slot_us: float,
    subband_mhz: Optional[tuple[float, float]] = None,
    duration_us: Optional[float] = None,
) -> tuple[np.ndarray, dict]:
    """Discretise a stream of PDWs into a (T_slots x N_bands) occupancy grid.

    A cell (slot s, band b) is marked 1 (transmitting) if any pulse whose centre
    frequency falls in band b has a [ToA, ToA + PulseWidth] interval overlapping
    slot s. This is exactly the ground-truth occupancy grid the problem
    statement asks for.
    """
    toa = np.asarray(pdw["toa_us"], dtype=np.float64)
    freq = np.asarray(pdw["freq_mhz"], dtype=np.float64)
    pw = np.asarray(pdw["pulse_width_us"], dtype=np.float64)

    if toa.size == 0:
        raise ValueError("empty PDW stream")

    # Pick the busiest sub-band if not given: 500 MHz slice with most pulses.
    if subband_mhz is None:
        lo, hi = float(freq.min()), float(freq.max())
        width = 500.0
        if hi - lo <= width:
            subband_mhz = (lo, max(hi, lo + width))
        else:
            edges = np.arange(lo, hi, width)
            counts = [(np.sum((freq >= e) & (freq < e + width)), e) for e in edges]
            best = max(counts, key=lambda x: x[0])[1]
            subband_mhz = (best, best + width)

    f_lo, f_hi = subband_mhz
    band_edges = np.linspace(f_lo, f_hi, n_bands + 1)

    t0 = float(toa.min())
    if duration_us is None:
        duration_us = float((toa + pw).max()) - t0
    n_slots = max(1, int(np.ceil(duration_us / slot_us)))

    grid = np.zeros((n_slots, n_bands), dtype=np.int8)

    in_band = (freq >= f_lo) & (freq < f_hi)
    idx = np.where(in_band)[0]
    for k in idx:
        b = int(np.searchsorted(band_edges, freq[k], side="right") - 1)
        if b < 0 or b >= n_bands:
            continue
        start = (toa[k] - t0) / slot_us
        end = (toa[k] + max(pw[k], slot_us * 0.01) - t0) / slot_us
        s0 = max(0, int(np.floor(start)))
        s1 = min(n_slots - 1, int(np.floor(end)))
        grid[s0 : s1 + 1, b] = 1

    meta = {
        "subband_mhz": [round(f_lo, 3), round(f_hi, 3)],
        "band_edges_mhz": [round(float(e), 3) for e in band_edges],
        "slot_us": slot_us,
        "n_slots": n_slots,
        "n_bands": n_bands,
        "t0_us": t0,
    }
    return grid, meta


def _cluster_emitter_stats(
    pdw: dict[str, np.ndarray], meta: dict
) -> list[dict]:
    """Approximate per-emitter stats via simple frequency binning + PRI
    estimation. This is NOT a deinterleaving solution (that is a different
    challenge); it only extracts summary stats good enough to (a) parameterise
    the synthetic generator and (b) validate periodicity detection.
    """
    toa = np.asarray(pdw["toa_us"], dtype=np.float64)
    freq = np.asarray(pdw["freq_mhz"], dtype=np.float64)
    pw = np.asarray(pdw["pulse_width_us"], dtype=np.float64)
    band_edges = np.asarray(meta["band_edges_mhz"], dtype=np.float64)
    slot_us = meta["slot_us"]

    stats: list[dict] = []
    for b in range(meta["n_bands"]):
        lo, hi = band_edges[b], band_edges[b + 1]
        mask = (freq >= lo) & (freq < hi)
        n = int(mask.sum())
        if n < 4:
            continue
        t_sorted = np.sort(toa[mask])
        diffs = np.diff(t_sorted)
        diffs = diffs[diffs > 0]
        if diffs.size == 0:
            continue
        # Dominant PRI ~ median inter-pulse gap (robust to missed pulses).
        pri_us = float(np.median(diffs))
        duty = float(min(1.0, (pw[mask].sum()) / (t_sorted[-1] - t_sorted[0] + 1e-9)))
        stats.append(
            {
                "band": b,
                "freq_center_mhz": round(float((lo + hi) / 2), 3),
                "n_pulses": n,
                "dominant_pri_us": round(pri_us, 3),
                "dominant_period_slots": max(1, int(round(pri_us / slot_us))),
                "duty_cycle": round(duty, 4),
            }
        )
    return stats


def _hop_stats(pdw: dict[str, np.ndarray], meta: dict) -> dict:
    """Estimate a coarse frequency-hop rate/range across the sub-band."""
    toa = np.asarray(pdw["toa_us"], dtype=np.float64)
    freq = np.asarray(pdw["freq_mhz"], dtype=np.float64)
    order = np.argsort(toa)
    f = freq[order]
    if f.size < 2:
        return {"hop_rate": 0.0, "hop_range_mhz": 0.0}
    hops = np.sum(np.abs(np.diff(f)) > 5.0)  # >5 MHz jump = a hop
    return {
        "hop_rate": round(float(hops) / f.size, 4),
        "hop_range_mhz": round(float(f.max() - f.min()), 3),
    }


# --------------------------------------------------------------------------- #
# Real download
# --------------------------------------------------------------------------- #


def load_real_pdws(num: int, columns: PDWColumns) -> list[dict[str, np.ndarray]]:
    """Download a small sample of Stare-mode test pulse trains from HF.

    Kept intentionally defensive: the gated repo's exact file layout is not
    publicly known, so we discover files, try common tabular formats, and map
    columns via ``columns``. On any failure we raise so the caller can fall back
    to stub mode rather than crashing the whole build.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN not set. Accept the dataset terms and export HF_TOKEN, "
            "or run with --stub."
        )
    from huggingface_hub import HfApi, hf_hub_download  # lazy import

    api = HfApi()
    files = api.list_repo_files(HF_DATASET, repo_type="dataset", token=token)
    # Prefer stare-mode test split tabular files.
    cand = [
        f
        for f in files
        if any(f.lower().endswith(ext) for ext in (".parquet", ".csv", ".h5", ".hdf5"))
        and "stare" in f.lower()
        and ("test" in f.lower() or "tst" in f.lower())
    ]
    if not cand:
        cand = [f for f in files if f.lower().endswith((".parquet", ".csv"))]
    if not cand:
        raise RuntimeError("No tabular PDW files discovered in the TSRD repo.")

    out: list[dict[str, np.ndarray]] = []
    for f in cand[:num]:
        local = hf_hub_download(HF_DATASET, f, repo_type="dataset", token=token)
        out.append(_read_pdw_file(local, columns))
    if not out:
        raise RuntimeError("Downloaded 0 usable pulse trains.")
    return out


def _read_pdw_file(path: str, columns: PDWColumns) -> dict[str, np.ndarray]:
    import pandas as pd  # lazy import

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_hdf(path)

    cols = {c.lower(): c for c in df.columns}

    def pick(name: str, *aliases: str) -> str:
        for cand in (name, *aliases):
            if cand.lower() in cols:
                return cols[cand.lower()]
        raise KeyError(f"Column '{name}' not found. Available: {list(df.columns)}")

    return {
        "toa_us": df[pick(columns.toa_us, "toa", "time_of_arrival", "t")].to_numpy(float),
        "freq_mhz": df[pick(columns.freq_mhz, "frequency", "cf", "centre_frequency")].to_numpy(float),
        "pulse_width_us": df[pick(columns.pulse_width_us, "pw", "pulse_width")].to_numpy(float),
        "aoa_deg": df[pick(columns.aoa_deg, "aoa", "angle")].to_numpy(float) if any(
            a in cols for a in ("aoa", "angle")
        ) else np.zeros(len(df)),
        "amplitude_db": df[pick(columns.amplitude_db, "amp", "amplitude")].to_numpy(float) if any(
            a in cols for a in ("amp", "amplitude")
        ) else np.zeros(len(df)),
    }


# --------------------------------------------------------------------------- #
# Stub generation (no network) - realistic-shaped fake PDWs
# --------------------------------------------------------------------------- #


def make_stub_pdws(seed: int) -> dict[str, np.ndarray]:
    """Generate fake PDWs with the same 5-D structure and realistic mix of
    periodic-PRI, frequency-hopping and steady emitters, over a 500 MHz slice
    across ~1e6 microseconds (short, for a small demo grid)."""
    rng = np.random.default_rng(seed)
    f_lo, f_hi = 9000.0, 9500.0  # MHz slice
    duration_us = 1_000_000.0  # 1 second window (keeps the grid small)

    toa_list, freq_list, pw_list, aoa_list, amp_list = [], [], [], [], []

    # A few periodic (rotating) emitters with clean PRIs.
    for _ in range(4):
        pri = float(rng.uniform(2000, 8000))  # us
        cf = float(rng.uniform(f_lo, f_hi))
        pw = float(rng.uniform(1.0, 5.0))
        aoa0 = float(rng.uniform(0, 360))
        n = int(duration_us / pri)
        base = np.arange(n) * pri + rng.uniform(0, pri)
        jitter = rng.normal(0, pri * 0.01, n)
        toa_list.append(base + jitter)
        freq_list.append(np.full(n, cf) + rng.normal(0, 2.0, n))
        pw_list.append(np.full(n, pw))
        # rotating antenna -> AoA drifts slowly
        aoa_list.append((aoa0 + np.arange(n) * 0.5) % 360)
        amp_list.append(rng.normal(-40, 3, n))

    # A couple of frequency hoppers.
    for _ in range(2):
        pri = float(rng.uniform(500, 1500))
        n = int(duration_us / pri)
        base = np.arange(n) * pri + rng.uniform(0, pri)
        hopset = rng.uniform(f_lo, f_hi, size=int(rng.integers(4, 7)))
        hops = hopset[rng.integers(0, len(hopset), size=n)]
        toa_list.append(base)
        freq_list.append(hops + rng.normal(0, 1.0, n))
        pw_list.append(np.full(n, float(rng.uniform(0.5, 2.0))))
        aoa_list.append(rng.uniform(0, 360, n))
        amp_list.append(rng.normal(-45, 4, n))

    # Steady/noise-like emitters.
    for _ in range(3):
        n = int(rng.integers(400, 1200))
        toa_list.append(np.sort(rng.uniform(0, duration_us, n)))
        freq_list.append(rng.uniform(f_lo, f_hi, n))
        pw_list.append(rng.uniform(0.5, 6.0, n))
        aoa_list.append(rng.uniform(0, 360, n))
        amp_list.append(rng.normal(-50, 5, n))

    pdw = {
        "toa_us": np.concatenate(toa_list),
        "freq_mhz": np.concatenate(freq_list),
        "pulse_width_us": np.concatenate(pw_list),
        "aoa_deg": np.concatenate(aoa_list),
        "amplitude_db": np.concatenate(amp_list),
    }
    order = np.argsort(pdw["toa_us"])
    return {k: v[order] for k, v in pdw.items()}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def _write_sample(k: int, grid: np.ndarray, meta: dict, is_stub: bool) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    npz_path = os.path.join(CACHE_DIR, f"tsrd_sample_{k}.npz")
    json_path = os.path.join(CACHE_DIR, f"tsrd_sample_{k}.json")
    np.savez_compressed(npz_path, grid=grid)
    meta = dict(meta)
    meta["stub"] = is_stub
    meta["sample_id"] = k
    meta["emitter_count"] = len(meta.get("emitter_stats", []))
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  wrote {npz_path}  grid={grid.shape}  active={int(grid.sum())}")


def _write_defaults(all_stats: list[dict], all_hop: list[dict], is_stub: bool) -> None:
    """Aggregate extracted stats into default ranges for the synthetic gen."""
    periods = [s["dominant_period_slots"] for s in all_stats if s["dominant_period_slots"] > 1]
    duties = [s["duty_cycle"] for s in all_stats]
    hop_rates = [h["hop_rate"] for h in all_hop]

    def _range(vals, lo_default, hi_default):
        if not vals:
            return [lo_default, hi_default]
        arr = np.asarray(vals, dtype=float)
        return [float(np.percentile(arr, 20)), float(np.percentile(arr, 80))]

    defaults = {
        "p01_range": [0.01, 0.08],
        "p10_range": [0.15, 0.45],
        "periodic_period_range": [
            int(max(5, np.percentile(periods, 20))) if periods else 18,
            int(max(10, np.percentile(periods, 80))) if periods else 60,
        ],
        "periodic_dwell_range": [1, 3],
        "periodic_jitter": 1,
        "hopper_hopset_range": [3, 6],
        "hopper_dwell_range": [1, 2],
        "duty_cycle_range": _range(duties, 0.04, 0.20),
        "hop_rate_hint": round(float(np.mean(hop_rates)), 4) if hop_rates else 0.1,
        "source": "tsrd_stub" if is_stub else "tsrd_real",
    }
    with open(DEFAULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(defaults, fh, indent=2)
    print(f"  wrote {DEFAULTS_PATH} (source={defaults['source']})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare TSRD occupancy-grid cache.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--real", action="store_true", help="download gated HF sample")
    mode.add_argument("--stub", action="store_true", help="offline fake PDWs")
    ap.add_argument("--num", type=int, default=2, help="number of samples")
    ap.add_argument("--n-bands", type=int, default=24)
    ap.add_argument("--slot-us", type=float, default=2000.0, help="time-slot width (us)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    columns = PDWColumns()
    all_stats: list[dict] = []
    all_hop: list[dict] = []

    if args.real:
        try:
            trains = load_real_pdws(args.num, columns)
        except Exception as exc:  # noqa: BLE001 - explicit fall-through message
            print(f"[real] failed: {exc}\n[real] falling back to --stub.")
            trains = [make_stub_pdws(args.seed + k) for k in range(args.num)]
            args.real = False
    else:
        trains = [make_stub_pdws(args.seed + k) for k in range(args.num)]

    for k, pdw in enumerate(trains):
        grid, meta = build_occupancy_grid(pdw, n_bands=args.n_bands, slot_us=args.slot_us)
        stats = _cluster_emitter_stats(pdw, meta)
        hop = _hop_stats(pdw, meta)
        meta["emitter_stats"] = stats
        meta["hop_stats"] = hop
        meta["aoa_available"] = bool(np.any(pdw.get("aoa_deg", np.zeros(1))))
        all_stats.extend(stats)
        all_hop.append(hop)
        _write_sample(k, grid, meta, is_stub=not args.real)

    _write_defaults(all_stats, all_hop, is_stub=not args.real)
    print("Done.")


if __name__ == "__main__":
    main()
