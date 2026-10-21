"""
LLM Propeller Design — Engineering tools.

Provides deterministic engineering tools for propeller blade analysis:
- blade_element_force: evaluates loads on a single blade element using
  prescribed lift and drag coefficients.
- evaluate_blade_design: evaluates a complete blade geometry using an
  iterative Blade Element Momentum Theory (BEMT) model, XFOIL polar data,
  and numerical integration along the blade radius.

The LLM interprets the design request, selects tools, and supplies
parameters. The engineering calculations are performed by Python.

Based on the tool format from:
https://github.com/readytensor/building-agents/tree/main/episodes/02-tools
"""
import functools
import inspect
import json
import math
from typing import get_type_hints

from polar import get_polar_db, reynolds_number

# --- Tool-call telemetry: records every tool call, in order, so we can see
# the path the agent took in a given round.
TOOL_CALLS = []  # list of {"round": n, "tool": name, "args": {...}}
CURRENT_ROUND = 0  # agent.py sets this on every loop iteration


def write_tool_telemetry():
    """Write this run's tool calls to tool_calls.jsonl."""
    with open("tool_calls.jsonl", "w", encoding="utf-8") as f:
        for call in TOOL_CALLS:
            f.write(json.dumps(call) + "\n")


# --- The @tool decorator: function signature -> JSON-schema tool definition.
def tool(description: str):
    """Attach a JSON-schema tool definition to a Python function."""
    json_types = {str: "string", int: "integer", float: "number", bool: "boolean"}

    def decorator(func):
        sig = inspect.signature(func)
        hints = get_type_hints(func)
        properties, required = {}, []
        for name, param in sig.parameters.items():
            t = hints.get(name, str)
            properties[name] = {"type": json_types.get(t, "string")}
            if param.default is inspect.Parameter.empty:
                required.append(name)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            TOOL_CALLS.append({"round": CURRENT_ROUND, "tool": func.__name__, "args": dict(bound.arguments)})
            return func(*args, **kwargs)

        wrapper.tool_definition = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }
        return wrapper
    return decorator


# --- The engineering tool: pure blade element (no BEM / no induction).
@tool(
    "Computes dT and dQ (load per unit span, not yet integrated along the "
    "radius) for a single propeller blade element, given V, RPM, r, beta "
    "(local pitch angle in degrees), chord c and constant Cl/Cd "
    "coefficients. Does not include axial/tangential induction factors "
    "(that is the next step, full BEMT)."
)
def blade_element_force(
    V: float, RPM: float, r: float, beta_deg: float, c: float,
    Cl: float, Cd: float, rho: float = 1.225,
) -> str:
    beta = math.radians(beta_deg)
    omega = RPM * 2 * math.pi / 60
    Vt = omega * r
    W = math.hypot(Vt, V)
    phi = math.atan2(V, Vt)
    alpha = beta - phi

    q = 0.5 * rho * W**2
    dL = q * c * Cl
    dD = q * c * Cd

    dT = dL * math.cos(phi) - dD * math.sin(phi)
    dQ = r * (dL * math.sin(phi) + dD * math.cos(phi))

    result = {
        "phi_deg": round(math.degrees(phi), 3),
        "alpha_deg": round(math.degrees(alpha), 3),
        "W_m_s": round(W, 4),
        "dT_per_r_N_per_m": round(dT, 4),
        "dQ_per_r_N": round(dQ, 4),
    }
    return json.dumps(result)


# --- The engineering tool: whole-blade evaluation (linear root-to-tip
# interpolation of chord and pitch, integrated along the radius). This is
# what lets the agent run a "small experiment": one call = one candidate
# geometry, scored by thrust and required power.
#
# Cl/Cd are not constants: at each station, local alpha and local Reynolds
# number are used to look up real XFOIL polar data (see polar.py and
# data/*.csv), linearly interpolated in alpha and in Re.
#
# Full BEMT: at each station, the axial and tangential induced velocities
# (vi, vt) are solved so that Blade Element loads match differential
# Momentum Theory for that annulus — not assumed zero. Induced velocities
# are solved in ABSOLUTE terms (m/s), not as a fraction of V, which is
# what makes this valid in hover (V=0) as well as forward flight: a
# ratio-based induction factor (Va = V*(1+a)) is singular at V=0, an
# absolute induced velocity is not.
def _solve_station_bemt(r, beta_deg, c, omega, V, rho, nu, db, B,
                         max_iter=3000, tol=1e-6, relax=0.15):
    beta = math.radians(beta_deg)
    vi, vt = 0.05 * omega * r, 0.0  # seed: a small positive axial induction
    converged = False

    for _ in range(max_iter):
        Va = V + vi
        Vt_local = omega * r - vt
        W = math.hypot(Va, Vt_local)
        phi = math.atan2(Va, Vt_local)
        alpha_deg = math.degrees(beta - phi)
        Re = reynolds_number(W, c, nu)
        Cl, Cd, _ = db.get_cl_cd(alpha_deg, Re)
        Cn = Cl * math.cos(phi) - Cd * math.sin(phi)  # thrust-direction coeff.
        Ct = Cl * math.sin(phi) + Cd * math.cos(phi)  # torque-direction coeff.

        Va_safe = max(Va, 1e-6)
        vi_new = B * W**2 * c * Cn / (8 * math.pi * r * Va_safe)
        vt_new = B * W**2 * c * Ct / (8 * math.pi * r * Va_safe)

        vi_next = vi + relax * (vi_new - vi)
        vt_next = vt + relax * (vt_new - vt)
        # Keep iterates physically bounded so a bad trial geometry can't
        # blow up the fixed-point loop before it has a chance to settle.
        vi_next = max(0.0, min(vi_next, 10 * omega * r + 1.0))
        vt_next = max(-0.9 * omega * r, min(vt_next, 0.9 * omega * r))

        if abs(vi_next - vi) < tol * max(1.0, vi) and abs(vt_next - vt) < tol * max(1.0, abs(vt)):
            vi, vt = vi_next, vt_next
            converged = True
            break
        vi, vt = vi_next, vt_next

    # Final pass at the converged (or last) induction state.
    Va = V + vi
    Vt_local = omega * r - vt
    W = math.hypot(Va, Vt_local)
    phi = math.atan2(Va, Vt_local)
    alpha_deg = math.degrees(beta - phi)
    Re = reynolds_number(W, c, nu)
    Cl, Cd, extrapolated = db.get_cl_cd(alpha_deg, Re)
    Cn = Cl * math.cos(phi) - Cd * math.sin(phi)
    Ct = Cl * math.sin(phi) + Cd * math.cos(phi)

    q = 0.5 * rho * W**2
    dT_dr = q * c * Cn          # per blade, per unit radius
    dQ_dr = r * q * c * Ct      # per blade, per unit radius
    return dT_dr, dQ_dr, alpha_deg, Re, Cl, Cd, extrapolated, converged, vi, vt


@tool(
    "Evaluates ONE candidate blade geometry end to end using an iterative "
    "Blade Element Momentum Theory (BEMT) model: beta(r) and c(r) are "
    "linearly interpolated between root and tip values, discretized into "
    "N stations, and at EACH station the axial and tangential induced "
    "velocities are solved iteratively so blade-element loads match "
    "differential momentum theory for that annulus (not assumed zero). "
    "Cl and Cd come from real XFOIL polar data for the chosen airfoil, "
    "looked up at each station's induction-corrected angle of attack and "
    "local Reynolds number. Results are integrated along the radius "
    "(trapezoidal rule) into total thrust T, torque Q and required power "
    "P. Use this to run small experiments: "
    "call it once per candidate geometry (varying beta_root_deg, "
    "beta_tip_deg, c_root, c_tip) and compare T and P across calls. "
    "VALIDITY: if at ANY station the local angle of attack or Reynolds "
    "number falls outside the range covered by the XFOIL polar data, the "
    "candidate is REJECTED as invalid (valid=false): no T or P are "
    "returned, because they would rely on coefficients the polar does not "
    "contain. The candidate is also REJECTED (valid=false) if the "
    "induction solver does not converge at any station, which signals a "
    "numerically unreasonable geometry (e.g. extreme pitch or chord). The "
    "response lists the offending stations, each with its reason, and the "
    "polar limits, so you can adjust the geometry and try again."
)
def evaluate_blade_design(
    beta_root_deg: float, beta_tip_deg: float,
    c_root: float, c_tip: float,
    r_root: float, R: float,
    RPM: float, V: float,
    B: int = 2, N: int = 12, rho: float = 1.225,
    nu: float = 1.46e-5, airfoil: str = "NACA4412",
) -> str:
    """Evaluate a propeller blade geometry using the implemented BEMT model."""
    db = get_polar_db(airfoil)
    omega = RPM * 2 * math.pi / 60
    stations = [r_root + (R - r_root) * i / (N - 1) for i in range(N)]

    dT_list, dQ_list = [], []
    alpha_min, alpha_max = float("inf"), float("-inf")
    invalid_stations = []
    re_min, re_max = db._re_list[0], db._re_list[-1]
    for r in stations:
        frac = (r - r_root) / (R - r_root)
        beta_deg = beta_root_deg + (beta_tip_deg - beta_root_deg) * frac
        c = c_root + (c_tip - c_root) * frac
        dT_dr, dQ_dr, alpha_deg, Re, Cl, Cd, extrapolated, converged, vi, vt = (
            _solve_station_bemt(r, beta_deg, c, omega, V, rho, nu, db, B)
        )
        dT_list.append(dT_dr)
        dQ_list.append(dQ_dr)
        alpha_min = min(alpha_min, alpha_deg)
        alpha_max = max(alpha_max, alpha_deg)
        reasons = []
        if extrapolated:
            # The polar was clamped at its edge for this station: that
            # value is NOT used. The candidate is rejected below.
            if Re < re_min:
                reasons.append(f"Reynolds below polar data (min {re_min})")
            elif Re > re_max:
                reasons.append(f"Reynolds above polar data (max {re_max})")
            else:
                reasons.append("angle of attack outside the polar data at this Reynolds")
        if not converged:
            reasons.append("induction solver did not converge")
        if reasons:
            invalid_stations.append({
                "r_m": round(r, 4), "alpha_deg": round(alpha_deg, 2),
                "Re": round(Re), "reason": "; ".join(reasons),
            })

    if invalid_stations:
        return json.dumps({
            "valid": False,
            "airfoil": airfoil,
            "reason": "candidate rejected: at the stations listed, local alpha "
                      "and/or Reynolds number is outside the XFOIL polar data "
                      "and/or the induction solver did not converge",
            "n_stations": N,
            "n_stations_invalid": len(invalid_stations),
            "invalid_stations": invalid_stations,
            "polar_limits": {"Re_min": re_min, "Re_max": re_max},
        })

    # Trapezoidal integration along the radius, then scale by blade count.
    T_per_blade = sum(
        0.5 * (dT_list[i] + dT_list[i + 1]) * (stations[i + 1] - stations[i])
        for i in range(N - 1)
    )
    Q_per_blade = sum(
        0.5 * (dQ_list[i] + dQ_list[i + 1]) * (stations[i + 1] - stations[i])
        for i in range(N - 1)
    )
    T = B * T_per_blade
    Q = B * Q_per_blade
    P = Q * omega

    result = {
        "valid": True,
        "airfoil": airfoil,
        "T_N": round(T, 4),
        "Q_Nm": round(Q, 5),
        "P_W": round(P, 3),
        "alpha_min_deg": round(alpha_min, 2),
        "alpha_max_deg": round(alpha_max, 2),
        "n_stations": N,
    }
    return json.dumps(result)


# --- Tool registry: name -> callable, plus the list of schemas for the LLM.
TOOLS = [blade_element_force, evaluate_blade_design]
TOOLS_BY_NAME = {t.__name__: t for t in TOOLS}
TOOL_DEFS = [t.tool_definition for t in TOOLS]
