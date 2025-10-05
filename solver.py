# solver.py — Motor estable para asignación de barcos a atraques con OR-Tools (CP-SAT)
# Compatible con Python 3.8+ (tipado clásico), validación robusta y soporte opcional de tramos lineales.

from ortools.sat.python import cp_model
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

DEFAULT_POLICY: Dict[str, float] = {
    "alpha": 1.3,        # W_calle >= alpha * LOA + beta
    "beta": 0.0,
    "ukc": 0.3,          # margen bajo quilla (m)
    "tide_safety": 0.1,  # margen de marea (m)
    "fender_mono": 0.25, # defensa por lado (monocasco)
    "fender_cata": 0.40, # defensa por lado (catamarán)
    "end_margin": 0.5,   # margen proa+popa dentro del finger
    "min_gap_linear": 0.4 # separación mínima entre barcos en costado
}

# --------------------------------------
# Validación de datos (robusta y simple)
# --------------------------------------
class BerthingValidator:
    def __init__(self) -> None:
        self.vessel_cols = ["vessel_id","loa","beam","draft","type","power_kw"]
        self.berth_cols  = ["berth_id","length","slip_width","depth","fairway_width","power_kw","type"]

    def _check_required(self, df: pd.DataFrame, cols: List[str], lbl: str, errors: List[str]) -> None:
        for c in cols:
            if c not in df.columns:
                errors.append(f"[{lbl}] Falta columna requerida: `{c}`")

    def validate_vessels(self, vessels: pd.DataFrame) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        self._check_required(vessels, self.vessel_cols, "Barcos", errors)
        if errors: return False, errors

        # Tipos numéricos seguros
        num_cols = ["loa","beam","draft","power_kw"]
        numeric = vessels[num_cols].apply(pd.to_numeric, errors="coerce")
        nulls = numeric.isnull().any()
        for col, isnull in nulls.items():
            if isnull: errors.append(f"[Barcos] Valores nulos o no numéricos en `{col}`")
        if (numeric < 0).any().any():
            errors.append("[Barcos] Hay valores negativos en loa/beam/draft/power_kw")

        # IDs
        if vessels["vessel_id"].isnull().any():
            errors.append("[Barcos] Hay `vessel_id` vacíos")
        if vessels["vessel_id"].duplicated().any():
            dups = vessels.loc[vessels["vessel_id"].duplicated(),"vessel_id"].unique().tolist()
            errors.append(f"[Barcos] IDs duplicados: {dups}")
        return len(errors)==0, errors

    def validate_berths(self, berths: pd.DataFrame) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        self._check_required(berths, self.berth_cols, "Atraques", errors)
        if errors: return False, errors

        # length y depth son obligatorios numéricos; el resto puede ser NaN según el tipo
        req = berths[["length","depth"]].apply(pd.to_numeric, errors="coerce")
        if req.isnull().any().any():
            errors.append("[Atraques] `length`/`depth` con valores nulos o no numéricos")
        if (req < 0).any().any():
            errors.append("[Atraques] `length`/`depth` con valores negativos")

        # IDs
        if berths["berth_id"].isnull().any():
            errors.append("[Atraques] Hay `berth_id` vacíos")
        if berths["berth_id"].duplicated().any():
            dups = berths.loc[berths["berth_id"].duplicated(),"berth_id"].unique().tolist()
            errors.append(f"[Atraques] IDs duplicados: {dups}")

        # Si hay lineales, exigir group_id
        has_linear = berths["type"].astype(str).str.lower().isin(["linear","costado"]).any()
        if has_linear and "group_id" not in berths.columns:
            errors.append("[Atraques] Existen filas `type=linear/costado` y falta columna `group_id`")
        return len(errors)==0, errors

# --------------------------------------
# Compatibilidad barco-atraque (detallada)
# --------------------------------------
def _fender(v_type: str, policy: Dict[str, float]) -> float:
    if isinstance(v_type, str) and v_type.lower().startswith("cata"):
        return policy["fender_cata"]
    return policy["fender_mono"]

def _pair_feasibility(v: Dict, b: Dict, policy: Dict[str, float]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    ok = True
    # longitud útil
    eff_len = b["length"] - policy["end_margin"]
    if v["loa"] > eff_len:
        ok = False; reasons.append("longitud: LOA excede longitud útil")

    # calado operativo
    depth_op = b["depth"] - (policy["ukc"] + policy["tide_safety"])
    if v["draft"] > depth_op:
        ok = False; reasons.append("calado: insuficiente (con UKC+marea)")

    # ancho de slip (si disponible)
    if "slip_width" in b and pd.notnull(b["slip_width"]):
        required = v["beam"] + 2*_fender(v.get("type",""), policy)
        if required > b["slip_width"]:
            ok = False; reasons.append("manga: no cabe con defensas")

    # potencia (si aplica)
    if "power_kw" in b and pd.notnull(b["power_kw"]):
        if v["power_kw"] > b["power_kw"]:
            ok = False; reasons.append("potencia: insuficiente")

    # calle (si aplica)
    if "fairway_width" in b and pd.notnull(b["fairway_width"]):
        loa_max = (b["fairway_width"] - policy["beta"]) / max(policy["alpha"], 1e-6)
        if v["loa"] > loa_max:
            ok = False; reasons.append("calle: LOA > LOA_max por maniobra")
    return ok, reasons

def calculate_compatibility_matrix(vessels_df: pd.DataFrame, berths_df: pd.DataFrame, policy: Dict[str, float]) -> pd.DataFrame:
    rows: List[Dict] = []
    for _, v in vessels_df.iterrows():
        for _, b in berths_df.iterrows():
            ok, reasons = _pair_feasibility(v.to_dict(), b.to_dict(), policy)
            rows.append({
                "vessel_id": v["vessel_id"],
                "berth_id": b["berth_id"],
                "compatible": bool(ok),
                "reasons": "; ".join(reasons) if reasons else "Compatible"
            })
    return pd.DataFrame(rows)

def _precheck_summary(v: Dict, berths: pd.DataFrame, policy: Dict[str, float]) -> str:
    counts = {"longitud":0,"calado":0,"manga":0,"potencia":0,"calle":0}
    best_id: Optional[str] = None
    best_issues = 999
    for _, b in berths.iterrows():
        ok, reasons = _pair_feasibility(v, b.to_dict(), policy)
        if ok:
            return "Compatible con algún atraque; no asignado por decisión global"
        if len(reasons) < best_issues:
            best_id = str(b["berth_id"]); best_issues = len(reasons)
        for r in reasons:
            for k in counts:
                if r.startswith(k):
                    counts[k]+=1
    detail = ", ".join([f"{k}:{v}" for k,v in counts.items() if v>0]) or "sin detalle"
    return f"Mejor candidato: {best_id or 'N/A'} | Causas: {detail}"

# --------------------------------------
# Solver principal
# --------------------------------------
def solve_assignment(
    vessels_df: pd.DataFrame,
    berths_df: pd.DataFrame,
    policy: Optional[Dict[str, float]] = None,
    time_limit: int = 20,
    prioritize_by: Optional[str] = None
):
    policy = policy or DEFAULT_POLICY

    validator = BerthingValidator()
    ok_v, err_v = validator.validate_vessels(vessels_df)
    ok_b, err_b = validator.validate_berths(berths_df)
    if not ok_v or not ok_b:
        raise ValueError("Errores de validación:\n" + "\n".join(err_v+err_b))

    vessels = vessels_df.copy()
    berths  = berths_df.copy()
    vessels["type"] = vessels["type"].astype(str).str.lower()
    berths["type"]  = berths["type"].astype(str).str.lower()

    if prioritize_by and prioritize_by in vessels.columns:
        vessels = vessels.sort_values(prioritize_by, ascending=False)

    discrete = berths[berths["type"].isin(["finger","thead"])].reset_index(drop=True)
    linear   = berths[berths["type"].isin(["linear","costado"])].reset_index(drop=True)

    V = vessels.to_dict("records")
    Bd = discrete.to_dict("records")
    Bl = linear.to_dict("records")

    model = cp_model.CpModel()

    # --- Discretos
    x: Dict[Tuple[int,int], cp_model.IntVar] = {}
    for i, v in enumerate(V):
        for j, b in enumerate(Bd):
            ok, _ = _pair_feasibility(v, b, policy)
            if ok:
                x[(i,j)] = model.NewBoolVar(f"x_{i}_{j}")

    # Un barco a lo sumo un slot/lineal (luego sumamos con y)
    # Un slot a lo sumo un barco
    for j, _ in enumerate(Bd):
        model.Add(sum(x[(i,j)] for i,_ in enumerate(V) if (i,j) in x) <= 1)

    # --- Lineales (opcional)
    y: Dict[Tuple[int,str], cp_model.IntVar] = {}
    s: Dict[Tuple[int,str], cp_model.IntVar] = {}
    groups: List[str] = []
    if not linear.empty:
        if "group_id" not in linear.columns:
            raise ValueError("Atraques `linear/costado` requieren columna `group_id`.")
        Lg = linear.groupby("group_id")["length"].sum().to_dict()
        Gdepth = linear.groupby("group_id")["depth"].min().to_dict()
        Gfair  = linear.groupby("group_id")["fairway_width"].min().to_dict()
        Gpow   = linear.groupby("group_id")["power_kw"].max().to_dict()
        groups = list(Lg.keys())

        for i, v in enumerate(V):
            for g in groups:
                b_rep = {
                    "length": Lg[g], "slip_width": np.nan,
                    "depth": float(Gdepth.get(g, np.nan)),
                    "fairway_width": float(Gfair.get(g, np.nan)),
                    "power_kw": float(Gpow.get(g, np.nan)), "type": "linear"
                }
                ok, _ = _pair_feasibility(v, b_rep, policy)
                if ok:
                    y[(i,g)] = model.NewBoolVar(f"y_{i}_{g}")
                    s[(i,g)] = model.NewIntVar(0, int(round(100*Lg[g])), f"s_{i}_{g}")

        # no solapamiento por grupo (intervalos opcionales)
        for g in groups:
            intervals = []
            for i,_ in enumerate(V):
                if (i,g) in y:
                    size_cm = int(round(100*(V[i]["loa"] + policy["min_gap_linear"])))
                    interval = model.NewOptionalFixedSizeIntervalVar(
                        s[(i,g)], size_cm, y[(i,g)], f"I_{i}_{g}"
                    )
                    intervals.append(interval)
                    # límite dentro del tramo
                    model.Add(s[(i,g)] + int(round(100*V[i]["loa"])) <= int(round(100*Lg[g]))).OnlyEnforceIf(y[(i,g)])
            if intervals:
                model.AddNoOverlap(intervals)

    # --- Cada barco a lo sumo a un sitio (slot o un grupo lineal)
    for i,_ in enumerate(V):
        choices = [x[(i,j)] for j,_ in enumerate(Bd) if (i,j) in x]
        choices += [y[(i,g)] for g in groups if (i,g) in y]
        if choices:
            model.Add(sum(choices) <= 1)

    # --- Objetivo multinivel (ponderado)
    assign_terms: List[cp_model.IntVar] = []
    waste_terms: List[cp_model.IntVar]  = []
    safety_terms: List[cp_model.IntVar] = []

    # Discretos
    for i, v in enumerate(V):
        for j, b in enumerate(Bd):
            if (i,j) in x:
                assign_terms.append(x[(i,j)])
                w = model.NewIntVar(0, 100000, f"w_d_{i}_{j}")
                waste_val = max(b["length"] - v["loa"], 0.0)
                model.Add(w == int(round(100*waste_val))).OnlyEnforceIf(x[(i,j)])
                model.Add(w == 0).OnlyEnforceIf(x[(i,j)].Not())
                waste_terms.append(w)

                sm = model.NewIntVar(0, 100000, f"sm_d_{i}_{j}")
                depth_margin = b["depth"] - (policy["ukc"]+policy["tide_safety"]) - v["draft"]
                model.Add(sm == int(round(100*max(depth_margin,0)))).OnlyEnforceIf(x[(i,j)])
                model.Add(sm == 0).OnlyEnforceIf(x[(i,j)].Not())
                safety_terms.append(sm)

    # Lineales
    for i,_ in enumerate(V):
        for g in groups:
            if (i,g) in y:
                assign_terms.append(y[(i,g)])
                waste_terms.append(s[(i,g)])  # compactación simple
                depth = float(Gdepth.get(g, np.nan))
                if not np.isnan(depth):
                    sm = model.NewIntVar(0, 100000, f"sm_l_{i}_{g}")
                    margin = depth - (policy["ukc"]+policy["tide_safety"]) - V[i]["draft"]
                    model.Add(sm == int(round(100*max(margin,0)))).OnlyEnforceIf(y[(i,g)])
                    model.Add(sm == 0).OnlyEnforceIf(y[(i,g)].Not())
                    safety_terms.append(sm)

    model.Maximize(1_000_000*sum(assign_terms) - 1_000*sum(waste_terms) + 1*sum(safety_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 8
    res = solver.Solve(model)

    assigned_rows: List[Dict] = []
    if res in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Discretos
        for i, v in enumerate(V):
            for j, b in enumerate(Bd):
                if (i,j) in x and solver.Value(x[(i,j)]) == 1:
                    assigned_rows.append({
                        "vessel_id": v["vessel_id"],
                        "berth_id": b["berth_id"],
                        "mode": "slot",
                        "loa": v["loa"],
                        "berth_length": b["length"],
                        "utilization_%": round(100*v["loa"]/max(b["length"],1e-6),1),
                        "length_margin": round(b["length"] - v["loa"], 2),
                        "depth_margin": round(b["depth"] - (policy["ukc"]+policy["tide_safety"]) - v["draft"], 2)
                    })
        # Lineales
        for i,_ in enumerate(V):
            for g in groups:
                if (i,g) in y and solver.Value(y[(i,g)]) == 1:
                    start_m = solver.Value(s[(i,g)]) / 100.0
                    assigned_rows.append({
                        "vessel_id": V[i]["vessel_id"],
                        "berth_id": str(g),
                        "mode": "linear",
                        "start_m": round(start_m, 2),
                        "loa": V[i]["loa"],
                        "depth_margin": round(Gdepth[g] - (policy["ukc"]+policy["tide_safety"]) - V[i]["draft"], 2)
                    })

    # No
