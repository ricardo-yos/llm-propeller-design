"""
LLM Propeller Design — Benchmark: LLM agent vs. Differential Evolution.

This module runs experiments with both methods and compares their results
under a shared evaluation budget.

Both methods use the same BEMT evaluator, design bounds, operating conditions,
feasibility criteria, and evaluation-counting rules, defined in optimize_de.py.
The agent.py and tools.py implementations remain unchanged.

Usage:
    python benchmark.py llm --runs N [--temperature T] [--top-p P] [--seed-base S]
        Run the LLM agent N times, saving each run in a separate directory
        under results/llm/runNNN/. Each run stores tool calls, metrics,
        standard output, and metadata. Optional generation parameters are
        passed through environment variables only when explicitly provided.
        If a seed base S is provided, run n receives seed S+n.

    python benchmark.py de --budget K --seeds 0 1 2 ...
        Run the Differential Evolution baseline with the specified evaluation
        budget for each seed.

    python benchmark.py report --budget K --checkpoints C1 C2 ...
        Replay the logged LLM evaluation calls through the shared evaluator,
        audit their comparability, and compare both methods using the same
        number of BEMT evaluations.

Counting rule:
    Each call to evaluate_blade_design counts as one evaluation, regardless
    of whether the candidate is valid. Calls to blade_element_force are
    counted separately and do not consume the BEMT evaluation budget.
    LLM candidates outside the design bounds or with different fixed operating
    conditions still count as evaluations but are excluded from the
    comparable solutions.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from urllib.parse import urlparse

import optimize_de as od

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LLM_DIR, DE_DIR = RESULTS / "llm", RESULTS / "de"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# --------------------------------------------------------------------------
# LLM: independent runs of the unchanged agent
# --------------------------------------------------------------------------
def cmd_llm(a):
    from dotenv import dotenv_values
    LLM_DIR.mkdir(parents=True, exist_ok=True)
    # agent.py loads .env from its working directory. Since each run executes
    # in a separate directory, pass the project's environment variables explicitly.
    env = {**os.environ, **{k: v for k, v in dotenv_values(HERE / ".env").items() if v is not None}}

    for _ in range(a.runs):
        n = len(list(LLM_DIR.glob("run*")))
        run_dir = LLM_DIR / f"run{n:03d}"
        run_dir.mkdir()
        run_env = dict(env)
        if a.temperature is not None:
            run_env["LLM_TEMPERATURE"] = str(a.temperature)
        if a.top_p is not None:
            run_env["LLM_TOP_P"] = str(a.top_p)
        if a.seed_base is not None:
            run_env["LLM_SEED"] = str(a.seed_base + n)
        started = datetime.now(timezone.utc).isoformat()
        t0 = time.time()
        with open(run_dir / "stdout.log", "w", encoding="utf-8") as log:
            proc = subprocess.run([sys.executable, str(HERE / "agent.py")], cwd=run_dir,
                                  env=run_env, stdout=log, stderr=subprocess.STDOUT)
        wall = round(time.time() - t0, 2)

        metrics = {}
        if (run_dir / "metrics.json").exists():
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        cfg = metrics.get("config", {})
        agent_usage = (metrics.get("agents") or [{}])[0]
        meta = {
            "method": "llm", "run": run_dir.name, "started_utc": started,
            "wall_time_s": wall, "returncode": proc.returncode,
            "model_requested": cfg.get("MODEL"),
            "model_served": cfg.get("served_models"),
            "system_fingerprints": cfg.get("system_fingerprints"),
            # Record the sampling parameters actually sent by agent.py.
            # An empty value means the provider's default settings were used.
            "generation_params_sent": cfg.get("generation_params"),
            "base_url_host": urlparse(run_env.get("LLM_BASE_URL", "")).netloc,
            "determinism_note": "not guaranteed: whether temperature/seed are honored depends on the provider",
            "task": metrics.get("inputs", {}).get("task"),
            "system_prompt_sha256": sha256(HERE / "system_prompt.md"),
            "agent_py_sha256": sha256(HERE / "agent.py"),
            "tools_py_sha256": sha256(HERE / "tools.py"),
            "problem_py_sha256": sha256(HERE / "problem.py"),
            "iterations": agent_usage.get("iterations"),
            "input_tokens": agent_usage.get("input_tokens"),
            "output_tokens": agent_usage.get("output_tokens"),
            "openai_sdk": importlib.metadata.version("openai") if _has("openai") else None,
            "python": platform.python_version(),
        }
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"{run_dir.name}: rc={proc.returncode} | {wall} s | sent={meta['generation_params_sent']} | "
              f"tokens in/out = {meta['input_tokens']}/{meta['output_tokens']}")


def _has(pkg: str) -> bool:
    """Return whether the specified Python package is installed."""
    try:
        importlib.metadata.version(pkg)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


# --------------------------------------------------------------------------
# Differential Evolution: independent runs for each seed
# --------------------------------------------------------------------------
def cmd_de(a):
    for seed in a.seeds:
        h, meta = od.run_de(seed, a.budget, a.popsize)
        od.save_run(DE_DIR, f"seed{seed}", h, meta)
        print(f"seed {seed}: {h.n_evals} evals | best = "
              f"{'none' if h.best_P is None else f'{h.best_P:.2f} W'}")


# --------------------------------------------------------------------------
# Report: replay, compare, summarize, and plot results
# --------------------------------------------------------------------------
def replay_llm_run(run_dir: Path, budget: int):
    """Replay logged LLM evaluations through the shared BEMT evaluator.

    Process evaluation calls in their original order, up to the specified
    budget, and rebuild the evaluation history for comparison with DE.
    Also return the total number of evaluation calls and calls to other tools.
    """
    calls = [json.loads(l) for l in (run_dir / "tool_calls.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    eval_calls = [c["args"] for c in calls if c["tool"] == "evaluate_blade_design"]
    hist = od.RunHistory("llm", run_dir.name)
    for args in eval_calls[:budget]:
        hist.add(args, od.safe_evaluate(args))
    return hist, len(eval_calls), len(calls) - len(eval_calls)


def curve(records, k):
    """Track the best feasible and comparable power over k evaluations.

    If a run ends before k evaluations, carry its final best value forward.
    Use infinity when no feasible and comparable solution has been found.
    """
    vals = [r["best_feasible_P_W"] if r["best_feasible_P_W"] is not None else math.inf for r in records]
    if not vals:
        return [math.inf] * k
    return (vals + [vals[-1]] * k)[:k]


def fmt(x):
    return "  --  " if x is None or x == math.inf else f"{x:6.2f}"


def stat(values):
    fin = sorted(v for v in values if v != math.inf)
    if not fin:
        return f"{'none':>26}"
    return f"{median(fin):6.2f} [{fin[0]:5.2f}-{fin[-1]:5.2f}] {len(fin)}/{len(values)}"


def cmd_report(a):
    K = a.budget
    llm = []  # LLM runs with evaluation records, call counts, and metadata.
    for d in sorted(LLM_DIR.glob("run*")):
        if not (d / "tool_calls.jsonl").exists():
            print(f"[skip] {d.name}: no tool_calls.jsonl (agent crashed before writing it?)")
            continue
        h, n_eval_calls, n_other = replay_llm_run(d, K)
        od.save_run(d, "records", h, {"replayed_from": "tool_calls.jsonl", "budget_cut": K,
                                      "evaluate_calls_total": n_eval_calls, "other_tool_calls": n_other})
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}
        llm.append(dict(name=d.name, recs=h.records, n_total=n_eval_calls, n_other=n_other, meta=meta))
    de, short = [], []
    for p in sorted(DE_DIR.glob("seed*.jsonl")):
        full = od.load_records(p)
        if len(full) < K:
            short.append(f"{p.stem} ({len(full)})")
        meta = json.loads(p.with_suffix(".meta.json").read_text(encoding="utf-8")) if p.with_suffix(".meta.json").exists() else {}
        de.append(dict(name=p.stem, recs=full[:K], meta=meta))
    if short:
        # Reject DE runs with fewer than K evaluations. Padding these runs would
        # incorrectly imply that the algorithm had been evaluated up to budget K.
        print(f"ERROR: DE run(s) with fewer than K={K} evaluations: {', '.join(short)}.\n"
              f"Re-run `de` with --budget >= {K}, or lower the report's --budget.")
        return
    if not llm or not de:
        print(f"Need both methods: {len(llm)} LLM run(s), {len(de)} DE run(s) found.")
        return

    print(f"\nBudget K = {K} evaluations | LLM runs: {len(llm)} | DE seeds: {len(de)}")
    print(f"Feasible = valid AND T >= {od.T_MIN_N} N | metric = lowest feasible power P [W]\n")

    # Report metrics for each individual run.
    print("Per run".ljust(14), "evals", "valid", "feas", "1st feas", "not-comp", "best P[W]", "T[N]", "Q[Nm]", "time[s]")
    summary = {"budget": K, "llm": [], "de": [], "de_evals_per_seed": [len(d["recs"]) for d in de]}
    for group, label in ((llm, "llm"), (de, "de")):
        for r in group:
            recs = r["recs"]
            comp_feas = [x for x in recs if x["feasible"] and x["comparable"]]
            best = min(comp_feas, key=lambda x: x["P_W"]) if comp_feas else None
            first = next((x["eval"] for x in recs if x["feasible"] and x["comparable"]), None)
            t = r["meta"].get("wall_time_s")
            row = dict(run=r["name"], evals=len(recs), valid=sum(x["valid"] for x in recs),
                       feasible=sum(x["feasible"] for x in recs), first_feasible=first,
                       not_comparable=sum(not x["comparable"] for x in recs),
                       best_P_W=None if best is None else best["P_W"],
                       best_T_N=None if best is None else best["T_N"],
                       best_Q_Nm=None if best is None else best["Q_Nm"],
                       best_design=None if best is None else best["design"], wall_time_s=t)
            summary[label].append(row)
            print(f"{label+'/'+r['name']:<14}{row['evals']:>5}{row['valid']:>7}{row['feasible']:>6}"
                  f"{str(first):>9}{row['not_comparable']:>10}  {fmt(row['best_P_W'])}   "
                  f"{fmt(row['best_T_N'])[:5]:>5} {('%.4f' % best['Q_Nm']) if best else '  --':>6}  {t if t is not None else '--'}")

    # Compare the best feasible power at the requested evaluation checkpoints.
    cps = sorted(set(a.checkpoints))
    print("\nBest feasible power at checkpoints: median [min-max] n_runs_with_a_feasible_solution/n_runs")
    for c in cps:
        kc = min(c, K)
        print(f"  @{c:<4} LLM {stat([curve(r['recs'], kc)[-1] for r in llm])}   |   DE {stat([curve(r['recs'], kc)[-1] for r in de])}")

    # Compare each LLM run against DE using the same number of evaluations.
    print("\nSame number of evaluations (each LLM run vs all DE seeds at that count)")
    for r in llm:
        k = max(1, min(len(r["recs"]), K))
        p_llm = curve(r["recs"], k)[-1]
        de_at_k = [curve(d["recs"], k)[-1] for d in de]
        better_de = sum(1 for v in de_at_k if v <= p_llm)
        print(f"  {r['name']}: {k:>3} evals | LLM {fmt(p_llm)} W | DE median {stat(de_at_k).split('[')[0].strip():>6} W"
              f" | DE seeds at least as good: {better_de}/{len(de)}"
              f" | calls beyond K ignored: {max(0, r['n_total'] - K)} | blade_element_force calls: {r['n_other']}")
    tot_nc = sum(sum(not x['comparable'] for x in r['recs']) for r in llm)
    if tot_nc:
        print(f"\n  note: {tot_nc} LLM evaluation(s) were not comparable (outside BOUNDS or different fixed "
              f"conditions); see 'audit_problems' in results/llm/*/records.jsonl")
    invalid_rate = lambda g: sum(not x["valid"] for r in g for x in r["recs"]) / max(1, sum(len(r["recs"]) for r in g))
    print(f"  rejected (invalid) evaluations: LLM {invalid_rate(llm):.0%} | DE {invalid_rate(de):.0%}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed: plot skipped)")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xs = list(range(1, K + 1))
    mat = [curve(d["recs"], K) for d in de]
    mid = [median([m[i] for m in mat if m[i] != math.inf] or [math.nan]) for i in range(K)]
    lo = [min([m[i] for m in mat if m[i] != math.inf] or [math.nan]) for i in range(K)]
    hi = [max([m[i] for m in mat if m[i] != math.inf] or [math.nan]) for i in range(K)]
    ax.plot(xs, mid, color="#1b9e77", lw=2, label=f"DE median ({len(de)} seeds)")
    ax.fill_between(xs, lo, hi, color="#1b9e77", alpha=0.18)
    for i, r in enumerate(llm):
        n = max(1, min(len(r["recs"]), K))
        c = [v if v != math.inf else math.nan for v in curve(r["recs"], n)]
        ax.step(range(1, n + 1), c, where="post", color="#d95f02", alpha=0.8, lw=1.5, label="LLM run" if i == 0 else None)
        ax.plot([n], [c[-1]], "o", color="#d95f02")
    ax.set_xlabel("BEMT evaluations")
    ax.set_ylabel("best feasible power [W]")
    ax.set_title(f"LLM vs DE: min power with T >= {od.T_MIN_N} N")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS / "benchmark.png", dpi=150)
    print(f"\nSaved results/summary.json and results/benchmark.png")


def main():
    p = argparse.ArgumentParser(
        description="LLM Propeller Design benchmark: LLM agent vs. Differential Evolution"
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("llm")
    s.add_argument("--runs", type=int, required=True)
    s.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="LLM sampling temperature; passed only when explicitly specified",
    )
    s.add_argument(
        "--top-p",
        dest="top_p",
        type=float,
        default=None,
        help="LLM nucleus sampling parameter; passed only when explicitly specified",
    )
    s.add_argument(
        "--seed-base",
        dest="seed_base",
        type=int,
        default=None,
        help="Base for LLM seeds; run n receives seed S+n. The provider may ignore it",
    )
    s.set_defaults(fn=cmd_llm)
    s = sub.add_parser("de"); s.add_argument("--budget", type=int, required=True)
    s.add_argument("--seeds", type=int, nargs="+", required=True)
    s.add_argument("--popsize", type=int, default=5, help="scipy popsize multiplier; 5 = value used in earlier tests")
    s.set_defaults(fn=cmd_de)
    s = sub.add_parser("report"); s.add_argument("--budget", type=int, required=True,
                                          help="K: evaluations compared per run; every DE run must have at least K")
    s.add_argument("--checkpoints", type=int, nargs="+", required=True); s.set_defaults(fn=cmd_report)
    a = p.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
