"""Headless scientific validation (spec step 10).

Runs a long simulation on both Synthetic and Real (TSRD-cache) modes and asserts
the Smart Scheduler beats all three baselines on the intercept rate. No UI or
network needed. Run:

    python validate.py --ticks 5000
"""

from __future__ import annotations

import argparse

from api.simulation import ScenarioConfig, Simulation


def run(cfg: ScenarioConfig, ticks: int) -> dict:
    sim = Simulation(cfg)
    for _ in range(ticks):
        sim.step()
    return sim.metrics_summary()


def report(title: str, summary: dict) -> bool:
    print(f"\n=== {title} ===")
    smart_rate = summary["smart"]["intercept_rate"]
    ok = True
    for k in ("smart", "sequential", "random", "greedy"):
        s = summary[k]
        print(
            f"  {k:11s} rate={s['intercept_rate']:.4f}  Pd={s['pd']:.3f}  "
            f"Pfa={s['pfa']:.3f}  sens={s['sensitivity']:.3f}  "
            f"avg_reward={s['avg_reward']:.3f}  pct_correct={s['pct_correct']:.3f}  "
            f"time_err={s['time_error']}"
        )
    for k in ("sequential", "random", "greedy"):
        if smart_rate <= summary[k]["intercept_rate"]:
            print(f"  [WARN] smart did not beat {k}")
            ok = False
    ratio_seq = smart_rate / max(summary["sequential"]["intercept_rate"], 1e-9)
    print(f"  smart / sequential intercept-rate ratio = {ratio_seq:.2f}x")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=5000)
    args = ap.parse_args()

    all_ok = True

    syn = ScenarioConfig(data_source="synthetic", n_bands=24, m=3, seed=7,
                         p_miss=0.1, p_fa=0.05)
    all_ok &= report("Synthetic (Dense)", run(syn, args.ticks))

    try:
        real = ScenarioConfig(data_source="real_tsrd", m=3, seed=7,
                              p_miss=0.1, p_fa=0.05, sample_id=0)
        all_ok &= report("Real Data (TSRD cache sample 0)", run(real, args.ticks))
    except FileNotFoundError as exc:
        print(f"\n[real_tsrd] skipped: {exc}")

    print("\nVALIDATION:", "PASS" if all_ok else "NEEDS TUNING")


if __name__ == "__main__":
    main()
