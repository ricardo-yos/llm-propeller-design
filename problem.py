"""
LLM Propeller Design — Shared problem definition.

This module is the single source of truth for the optimization problem,
ensuring that the LLM agent and the Differential Evolution baseline use
the same definitions:

- CONDITIONS, T_MIN_N, BOUNDS, VAR_NAMES: shared problem constants.
- evaluator_defaults(): retrieves default values from the BEMT evaluator's
  function signature instead of duplicating them.
- task_text(): generates the task description for the LLM from these constants.

Imports are kept lightweight so that agent.py can import this module
without requiring SciPy.
"""
import inspect

import tools

# Fixed operating conditions from the original task in agent.py.
# B, N, rho, nu, and airfoil use the evaluator's default values.
CONDITIONS = dict(r_root=0.025, R=0.10, RPM=8000, V=0.0)
T_MIN_N = 5.0  # Minimum required thrust [N].

# Design variables and their allowed ranges for the benchmark.
BOUNDS = {
    "beta_root_deg": (5.0, 40.0),
    "beta_tip_deg": (0.0, 30.0),
    "c_root": (0.010, 0.060),
    "c_tip": (0.005, 0.040),
}
VAR_NAMES = list(BOUNDS)


def evaluator_defaults() -> dict:
    """Return the default arguments of the BEMT evaluator.

    Defaults are read from the function signature rather than duplicated here.
    Both optimization methods use these values when optional arguments are omitted.
    """
    sig = inspect.signature(tools.evaluate_blade_design.__wrapped__)
    return {
        k: p.default
        for k, p in sig.parameters.items()
        if p.default is not inspect.Parameter.empty
    }


def task_text() -> str:
    """Build the LLM task description from the shared problem definition."""
    d = evaluator_defaults()
    fixed = {
        "RPM": CONDITIONS["RPM"],
        "R": CONDITIONS["R"],
        "r_root": CONDITIONS["r_root"],
        "V": CONDITIONS["V"],
        **{k: d[k] for k in ("B", "N", "rho", "nu", "airfoil")},
    }
    bounds = "\n".join(
        f"  - {k}: {lo} to {hi}" for k, (lo, hi) in BOUNDS.items()
    )
    fixed_txt = ", ".join(f"{k} = {v}" for k, v in fixed.items())

    return (
        f"Design the blade geometry ({d['airfoil']}) of a {d['B']}-blade propeller in "
        f"hover, RPM {CONDITIONS['RPM']}, r_root = {CONDITIONS['r_root']} m, "
        f"R = {CONDITIONS['R']} m. Target thrust: {T_MIN_N:g} N. Goal: MINIMIZE the "
        "required power while reaching that thrust, by varying the root/tip pitch "
        "and chord.\n\n"
        "Only these four variables may change, each within its allowed range:\n"
        f"{bounds}\n\n"
        f"Everything else is FIXED and must not be changed: {fixed_txt}."
    )
