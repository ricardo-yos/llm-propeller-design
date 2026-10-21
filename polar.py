"""
LLM Propeller Design — Airfoil polar database

Loads real XFOIL polar data (alpha, Cl, Cd, one CSV per Reynolds number)
and interpolates Cl(alpha, Re) / Cd(alpha, Re): linear in alpha within
each Re's polar, then linear between the two bracketing Re polars.

Data source: pre-computed XFOIL polars for NACA 4412 from
http://airfoiltools.com (Ncrit=5, Mach=0), stored under data/.
Swap in your own airfoil by dropping CSVs named
"<airfoil>_re<Reynolds>.csv" (columns: alpha,Cl,Cd) into data/ and
pointing PolarDatabase at that prefix.
"""
import bisect
import csv
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


class PolarDatabase:
    def __init__(self, airfoil: str = "NACA4412", data_dir: Path = DATA_DIR):
        self.airfoil = airfoil
        # CSV file names use the standard uppercase airfoil name
        # (e.g. NACA4412_re50000.csv). Normalize the input so callers can pass
        # "NACA4412", "naca4412", "NACA 4412", etc.
        search_name = airfoil.strip().replace(" ", "").upper()
        self.polars = {}  # Re -> sorted list of (alpha_deg, Cl, Cd)
        for path in sorted(data_dir.glob(f"{search_name}_re*.csv")):
            re = int(path.stem.split("_re")[-1])
            rows = []
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    rows.append((float(row["alpha"]), float(row["Cl"]), float(row["Cd"])))
            rows.sort(key=lambda r: r[0])
            self.polars[re] = rows
        if not self.polars:
            raise FileNotFoundError(f"No polar CSVs found for airfoil '{airfoil}' in {data_dir}")
        self._re_list = sorted(self.polars)

    def _interp_alpha(self, rows, alpha_deg):
        """Linear interpolation of Cl/Cd in alpha within one Re's polar.
        Clamps (flagging extrapolation) outside the data range."""
        alphas = [r[0] for r in rows]
        extrapolated = alpha_deg < alphas[0] or alpha_deg > alphas[-1]
        a = min(max(alpha_deg, alphas[0]), alphas[-1])
        i = bisect.bisect_left(alphas, a)
        if i == 0:
            return rows[0][1], rows[0][2], extrapolated
        if i >= len(alphas):
            return rows[-1][1], rows[-1][2], extrapolated
        a0, cl0, cd0 = rows[i - 1]
        a1, cl1, cd1 = rows[i]
        if a1 == a0:
            return cl0, cd0, extrapolated
        t = (a - a0) / (a1 - a0)
        return cl0 + t * (cl1 - cl0), cd0 + t * (cd1 - cd0), extrapolated

    def get_cl_cd(self, alpha_deg: float, Re: float):
        """Cl, Cd at (alpha_deg, Re), linearly interpolated in alpha
        within the bracketing Re polars and then between them.
        Returns (Cl, Cd, extrapolated) — extrapolated is True if alpha_deg
        or Re fell outside the available data range (result is clamped,
        not a reliable physical prediction)."""
        re_clamped = min(max(Re, self._re_list[0]), self._re_list[-1])
        re_extrap = re_clamped != Re

        if re_clamped in self.polars:
            cl, cd, a_extrap = self._interp_alpha(self.polars[re_clamped], alpha_deg)
            return cl, cd, bool(a_extrap or re_extrap)

        i = bisect.bisect_left(self._re_list, re_clamped)
        re0, re1 = self._re_list[i - 1], self._re_list[i]
        cl0, cd0, extrap0 = self._interp_alpha(self.polars[re0], alpha_deg)
        cl1, cd1, extrap1 = self._interp_alpha(self.polars[re1], alpha_deg)
        t = (re_clamped - re0) / (re1 - re0)
        cl = cl0 + t * (cl1 - cl0)
        cd = cd0 + t * (cd1 - cd0)
        return cl, cd, bool(extrap0 or extrap1 or re_extrap)


def reynolds_number(W: float, c: float, nu: float = 1.46e-5) -> float:
    """Local Reynolds number from relative velocity W, chord c and the
    air's kinematic viscosity nu (default: standard air at 15 degC)."""
    return W * c / nu


# Database cache, keyed by normalized airfoil name, lazily built on first
# use so importing this module never touches disk unless a tool actually
# needs it. Case/spacing in the requested name doesn't matter — "NACA4412"
# (the standard form) and "naca4412" share one cached instance.
_db_cache: dict[str, PolarDatabase] = {}


def get_polar_db(airfoil: str = "NACA4412") -> PolarDatabase:
    key = airfoil.strip().lower().replace(" ", "")
    if key not in _db_cache:
        _db_cache[key] = PolarDatabase(airfoil)
    return _db_cache[key]
