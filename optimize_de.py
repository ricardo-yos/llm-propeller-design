"""
LLM Propeller Design — Differential Evolution baseline.

Optimizes the same four design variables as the LLM agent using SciPy's
Differential Evolution algorithm and the same BEMT evaluator.

Problem definition and feasibility criteria are shared through problem.py.
The fitness function guides DE's search and is not a benchmark metric.

Each evaluation corresponds to one call to the BEMT evaluator, whether
the candidate is valid or invalid.

Usage:
    python optimize_de.py --budget K --seeds 0 1 2
"""
import argparse
import json
import math
import platform
import time
from pathlib import Path

import scipy
from scipy.optimize import differential_evolution

import tools

HERE = Path(__file__).resolve().parent

# --- The problem (identical for LLM and DE): defined once in problem.py ----
from problem import BOUNDS, CONDITIONS, T_MIN_N, VAR_NAMES, evaluator_defaults  # noqa: E402

# Penalties used by DE to guide the search toward feasible candidates.
# They affect optimization only, not the benchmark metrics.
PENALTY_W_PER_REL_SHORTFALL = 1000.0  # W added per 100% thrust shortfall
INVALID_FITNESS_W = 1.0e4             # fitness of a rejected (invalid) candidate

# Bypass the telemetry decorator to keep DE evaluations out of the
# agent's tool-call log.
_EVALUATOR = tools.evaluate_blade_design.__wrapped__


def evaluate_bemt(args: dict) -> dict:
    """Evaluate one candidate and parse the evaluator's JSON response."""
    return json.loads(_EVALUATOR(**args))


def safe_evaluate(args: dict) -> dict:
    """Return a failed evaluation if the evaluator raises an exception."""
    try:
        return evaluate_bemt(args)
    except Exception as e:  # noqa: BLE001
        return {"valid": False, "exception": f"{type(e).__name__}: {e}",
                "invalid_stations": [{"reason": f"exception: {type(e).__name__}"}]}


# --- Judging a result (shared) ----------------------------------------------
def is_feasible(result: dict) -> bool:
    """Check whether a valid design meets the minimum thrust requirement."""
    return bool(result.get("valid")) and result["T_N"] >= T_MIN_N


def failure_reasons(result: dict) -> list[str]:
    """Return the reasons a candidate fails validity or thrust requirements."""
    if not result.get("valid"):
        return sorted({s["reason"] for s in result.get("invalid_stations", [])}) or ["invalid"]
    if result["T_N"] < T_MIN_N:
        return ["thrust below minimum"]
    return []


def fitness(result: dict) -> float:
    """
    Calculate the scalar objective minimized by DE.

    Valid candidates are scored by required power plus a penalty proportional
    to relative thrust shortfall. Invalid candidates receive a fixed penalty.
    """
    if not result.get("valid"):
        return INVALID_FITNESS_W
    shortfall = max(0.0, (T_MIN_N - result["T_N"]) / T_MIN_N)
    return result["P_W"] + PENALTY_W_PER_REL_SHORTFALL * shortfall


def audit(args: dict) -> list[str]:
    """
    Check whether an evaluation respects the benchmark's design bounds
    and fixed operating conditions.

    An empty list means the evaluation is comparable with the DE baseline.
    """
    problems = []
    for k, (lo, hi) in BOUNDS.items():
        v = args.get(k)
        if v is None or not (lo <= v <= hi):
            problems.append(f"{k}={v} outside bounds [{lo}, {hi}]")
    expected = {**evaluator_defaults(), **CONDITIONS}
    for k, want in expected.items():
        got = args.get(k, want)  # Omitted optional arguments use their defaults.
        if isinstance(want, str):
            same = str(got).strip().lower().replace(" ", "") == want.lower()
        else:
            same = math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-12)
        if not same:
            problems.append(f"{k}={got} (benchmark uses {want})")
    return problems


# --- One record schema + evaluation counter for both methods ----------------
class RunHistory:
    """Store evaluation records and track the best comparable feasible design."""
    def __init__(self, method: str, run_id: str):
        self.method, self.run_id = method, run_id
        self.records: list[dict] = []
        self.best_P: float | None = None   # best feasible AND comparable power

    @property
    def n_evals(self) -> int:
        """Return the number of evaluations recorded so far."""
        return len(self.records)

    def add(self, args: dict, result: dict, t_s: float | None = None) -> dict:
        """Record an evaluation and update the best feasible power."""
        problems = audit(args)
        feasible = is_feasible(result)
        comparable = not problems
        if feasible and comparable and (self.best_P is None or result["P_W"] < self.best_P):
            self.best_P = result["P_W"]
        rec = {
            "method": self.method, "run": self.run_id, "eval": self.n_evals + 1,
            "design": {k: args.get(k) for k in VAR_NAMES},
            "other_args": {k: v for k, v in args.items() if k not in VAR_NAMES},
            "valid": bool(result.get("valid")),
            "T_N": result.get("T_N"), "Q_Nm": result.get("Q_Nm"), "P_W": result.get("P_W"),
            "feasible": feasible,
            "failure_reasons": failure_reasons(result),
            "comparable": comparable, "audit_problems": problems,
            "best_feasible_P_W": self.best_P,
            "t_eval_s": t_s,
        }
        self.records.append(rec)
        return rec


def save_run(directory: Path, name: str, history: RunHistory, meta: dict) -> None:
    """Save evaluation records and run metadata to JSON files."""
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in history.records:
            f.write(json.dumps(r) + "\n")
    (directory / f"{name}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_records(path: Path) -> list[dict]:
    """Load evaluation records from a JSONL file."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- Differential evolution -------------------------------------------------
class BudgetReached(Exception):
    """Raised when a run reaches its evaluation budget."""
    pass


def run_de(seed: int, budget: int, popsize: int) -> tuple[RunHistory, dict]:
    """Run Differential Evolution with a fixed evaluation budget and seed."""
    history = RunHistory("de", f"seed{seed}")
    t_start = time.time()

    def objective(x):
        """Evaluate one candidate and return its DE fitness."""
        if history.n_evals >= budget:
            raise BudgetReached
        args = {**dict(zip(VAR_NAMES, map(float, x))), **CONDITIONS}
        t0 = time.time()
        result = evaluate_bemt(args)
        history.add(args, result, t_s=round(time.time() - t0, 5))
        return fitness(result)

    de_config = dict(strategy="best1bin", maxiter=10**6, popsize=popsize, tol=0.0,
                     mutation=(0.5, 1), recombination=0.7, init="latinhypercube",
                     atol=0, updating="immediate", workers=1, polish=False)
    try:
        differential_evolution(objective, [BOUNDS[k] for k in VAR_NAMES], seed=seed, **de_config)
    except BudgetReached:
        pass  # The evaluation budget is the stopping criterion.

    meta = {
        "method": "de", "seed": seed, "budget": budget, "n_evals": history.n_evals,
        "wall_time_s": round(time.time() - t_start, 3),
        "de_config": {k: list(v) if isinstance(v, tuple) else v for k, v in de_config.items()},
        "scipy": scipy.__version__, "python": platform.python_version(),
        "bounds": BOUNDS, "conditions": CONDITIONS, "evaluator_defaults": evaluator_defaults(),
        "t_min_N": T_MIN_N, "penalty_w_per_rel_shortfall": PENALTY_W_PER_REL_SHORTFALL,
        "invalid_fitness_w": INVALID_FITNESS_W,
    }
    return history, meta


def main():
    """Parse command-line arguments and run the requested DE experiments."""
    p = argparse.ArgumentParser(
    description="Differential Evolution baseline for LLM Propeller Design"
    )
    p.add_argument("--budget", type=int, required=True, help="BEMT evaluations per run")
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--popsize", type=int, default=5,
                   help="scipy popsize multiplier (population = popsize x 4). 5 = value used in earlier tests")
    a = p.parse_args()
    for seed in a.seeds:
        h, meta = run_de(seed, a.budget, a.popsize)
        save_run(HERE / "results" / "de", f"seed{seed}", h, meta)
        print(f"seed {seed}: {h.n_evals} evals | best feasible P = "
              f"{'none' if h.best_P is None else f'{h.best_P:.2f} W'} | {meta['wall_time_s']} s")


if __name__ == "__main__":
    main()
