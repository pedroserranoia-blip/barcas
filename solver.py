# solver.py — Motor de optimización para ubicar embarcaciones en pantalanes
# - Compatible con OR-Tools 9.10 (firma de NewOptionalFixedSizeIntervalVar con 4 args)
# - Valida datos de entrada de forma robusta
# - Soporta:
#     * Atraques discretos: type in {"finger","thead"} (un barco por slot)
#     * Tramos lineales/costado: type in {"linear","costado"} con group_id (varios barcos por tramo, sin solape)
# - Objetivo multinivel: max asignados -> min "waste"/huecos -> max márgenes de calado
# - Devuelve SIEMPRE (assignments_df, unassigned_df, stats_dict)

from ortools.sat.python import cp_model
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

DEFAULT_POLICY: Dict[str, float] = {
    "alpha": 1.3,         # W_calle >= alpha * LOA + beta
    "beta": 0.0,
    "ukc": 0.3,           # margen bajo quilla (m)
    "tide_safety": 0.1,   # margen extra por marea (m)
    "fender_mono": 0.25,  # defensa por lado (monocasco)
    "fender_cata": 0.40,  # defensa por lado (catamarán)
    "end_margin": 0.5,    # margen total proa+popa dentro del finger
    "min_gap_linear": 0.4 # separación mínima entre barcos en costado (m)
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
        if errors:
            return False, errors

        # Tipos numéricos seguros
        num_cols = ["loa","beam","draft","power_kw"]
        numeric = vessels[num_cols].apply(pd.to_numeric, errors="coerce")

        nulls = numeric.isnull().any()
        for col, isnull in nulls.items():
            if isnull:
                errors.append(f"[Barcos] Valores nulos o no numéricos en `{col}`")

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
        if errors:
            return False, errors

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
    """Comprueba compatibilidad v (vessel) con b (berth). Devuelve (ok, razones)."""
    reasons: List[str] = []
    ok = True

    # Longitud útil
    eff_len = float(b["length"]) - policy["end_margin"]
    if float(v["loa"]) > eff_len:
        ok = False; reasons.append("longitud: LOA excede longitud útil")

    # Calado operativo
    depth_op = float(b["depth"]) - (policy["ukc"] + policy["tide_safety"])
    if float(v["draft"]) > depth_op:
        ok = False; reasons.append("calado: insuficiente (con UKC+marea)")

    # Ancho de slip (si está definido)
    if "slip_width" in b and pd.notnull(b["slip_width"]):
        required = float(v["beam"]) + 2.0*_fender(v.get("type",""), policy)
        if required > float(b["slip_width"]):
            ok = False; reasons.append("manga: no cabe con defensas")

    # Potencia (si aplica)
    if "power_kw" in b and pd.notnull(b["power_kw"]):
        if float(v["power_kw"]) > float(b["power_kw"]):
            ok = False; reasons.append("potencia: insuficiente")

    # Calle (si aplica)
    if "fairway_width" in b and pd.notnull(b["fairway_width"]):
        loa_max = (float(b["fairway_width"]) - policy["beta"]) / max(policy["alpha"], 1e-6)
        if float(v["loa"]) > loa_max:
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
    """Devuelve SIEMPRE (assignments_df, unassigned_df, stats_dict)."""
    policy = policy or DEFAULT_POLICY

    # ---------- Validación (return temprano, sin raise) ----------
    validator = BerthingValidator()
    ok_v, err_v = validator.validate_vessels(vessels_df)
    ok_b, err_b = validator.validate_berths(berths_df)

    if not ok_v or not ok_b:
        errors = (err_v or []) + (err_b or [])
        # Construimos estructura válida para la app
        empty_assign = pd.DataFrame(columns=[
            "vessel_id","berth_id","mode","loa","berth_length",
            "utilization_%","length_margin","depth_margin","start_m"
        ])
        unassigned_rows: List[Dict] = []
        if isinstance(vessels_df, pd.DataFrame) and "vessel_id" in vessels_df.columns:
            for _, v in vessels_df.iterrows():
                unassigned_rows.append({
                    "vessel_id": v.get("vessel_id", "N/A"),
                    "reason": "Errores de validación: " + " | ".join(errors) if errors else "Errores de validación"
                })
        stats = {
            "n_vessels": int(len(vessels_df)) if isinstance(vessels_df, pd.DataFrame) else 0,
            "n_berths": int(len(berths_df)) if isinstance(berths_df, pd.DataFrame) else 0,
            "assigned": 0,
            "unassigned": len(unassigned_rows),
            "occupancy_pct": 0.0,
            "rejection_reasons": {"validacion": len(unassigned_rows)}
        }
        return empty_assign, pd.DataFrame(unassigned_rows), stats

    # ---------- Preparación de datos ----------
    vessels = vessels_df.copy()
    berths  = berths_df.copy()
    vessels["type"] = vessels["type"].astype(str).str.lower()
    berths["type"]  = berths["type"].astype(str).str.lower()

    if prioritize_by and prioritize_by in vessels.columns:
        vessels = vessels.sort_values(prioritize_by, ascending=False)

    discrete = berths[berths["type"].isin(["finger","thead"])].reset_index(drop=True)
    linear   = berths[berths["type"].isin(["linear","costado"])].reset_index(drop=True)

    V  = vessels.to_dict("records")
    Bd = discrete.to_dict("records")

    # Agregados para lineales (por grupo)
    groups: List[str] = []
    Lg: Dict[str, float] = {}
    Gdepth: Dict[str, float] = {}
    Gfair: Dict[str, float] = {}
    Gpow: Dict[str, float] = {}

    if not linear.empty:
        if "group_id" not in linear.columns:
            # Si hay lineales sin group_id, lo tratamos como error suave (return válido)
            empty_assign = pd.DataFrame(columns=[
                "vessel_id","berth_id","mode","loa","berth_length",
                "utilization_%","length_margin","depth_margin","start_m"
            ])
            unassigned_rows = [{"vessel_id": r["vessel_id"], "reason": "Falta columna group_id en atraques lineales"}
                               for _, r in vessels.iterrows()]
            stats = {
                "n_vessels": len(vessels),
                "n_berths": len(berths),
                "assigned": 0,
                "unassigned": len(unassigned_rows),
                "occupancy_pct": 0.0,
                "rejection_reasons": {"estructura_lineal": len(unassigned_rows)}
            }
            return empty_assign, pd.DataFrame(unassigned_rows), stats

        Lg     = linear.groupby("group_id")["length"].sum().to_dict()
        Gdepth = linear.groupby("group_id")["depth"].min().to_dict()
        Gfair  = linear.groupby("group_id")["fairway_width"].min().to_dict()
        Gpow   = linear.groupby("group_id")["power_kw"].max().to_dict()
        groups = list(Lg.keys())

    # ---------- Modelo CP-SAT ----------
    model = cp_model.CpModel()

    # Variables discretas: x[i,j]
    x: Dict[Tuple[int,int], cp_model.IntVar] = {}
    for i, v in enumerate(V):
        for j, b in enumerate(Bd):
            ok, _ = _pair_feasibility(v, b, policy)
            if ok:
                x[(i,j)] = model.NewBoolVar(f"x_{i}_{j}")

    # Cada slot a lo sumo 1 barco
    for j, _ in enumerate(Bd):
        model.Add(sum(x[(i,j)] for i,_ in enumerate(V) if (i,j) in x) <= 1)

    # Variables lineales: y[i,g] y posiciones s[i,g] (cm)
    y: Dict[Tuple[int,str], cp_model.IntVar] = {}
    s: Dict[Tuple[int,str], cp_model.IntVar] = {}
    if groups:
        for i, v in enumerate(V):
            for g in groups:
                b_rep = {
                    "length": Lg[g],
                    "slip_width": np.nan,
                    "depth": float(Gdepth.get(g, np.nan)),
                    "fairway_width": float(Gfair.get(g, np.nan)),
                    "power_kw": float(Gpow.get(g, np.nan)),
                    "type": "linear"
                }
                ok, _ = _pair_feasibility(v, b_rep, policy)
                if ok:
                    y[(i,g)] = model.NewBoolVar(f"y_{i}_{g}")
                    s[(i,g)] = model.NewIntVar(0, int(round(100*Lg[g])), f"s_{i}_{g}")

        # No solapamiento por grupo (intervalos opcionales)
        for g in groups:
            intervals = []
            for i,_ in enumerate(V):
                if (i,g) in y:
                    size_cm = int(round(100*(V[i]["loa"] + policy["min_gap_linear"])))
                    # Firma correcta en OR-Tools 9.10: start, size, is_present, name
                    interval = model.NewOptionalFixedSizeIntervalVar(
                        s[(i,g)], size_cm, y[(i,g)], f"I_{i}_{g}"
                    )
                    intervals.append(interval)
                    # Cierre de tramo
                    model.Add(s[(i,g)] + int(round(100*V[i]["loa"])) <= int(round(100*Lg[g]))).OnlyEnforceIf(y[(i,g)])
            if intervals:
                model.AddNoOverlap(intervals)

    # Un barco a lo sumo a un sitio (slot o lineal)
    for i,_ in enumerate(V):
        choices = [x[(i,j)] for j,_ in enumerate(Bd) if (i,j) in x]
        if groups:
            choices += [y[(i,g)] for g in groups if (i,g) in y]
        if choices:
            model.Add(sum(choices) <= 1)

    # ---------- Objetivo ----------
    assign_terms: List[cp_model.IntVar] = []
    waste_terms:  List[cp_model.IntVar] = []
    safety_terms: List[cp_model.IntVar] = []

    # Discretos
    for i, v in enumerate(V):
        for j, b in enumerate(Bd):
            if (i,j) in x:
                assign_terms.append(x[(i,j)])
                # desperdicio de longitud (cm)
                w = model.NewIntVar(0, 100000, f"w_d_{i}_{j}")
                waste_val = max(float(b["length"]) - float(v["loa"]), 0.0)
                model.Add(w == int(round(100*waste_val))).OnlyEnforceIf(x[(i,j)])
                model.Add(w == 0).OnlyEnforceIf(x[(i,j)].Not())
                waste_terms.append(w)
                # margen de calado (cm, sólo positivo)
                sm = model.NewIntVar(0, 100000, f"sm_d_{i}_{j}")
                depth_margin = float(b["depth"]) - (policy["ukc"]+policy["tide_safety"]) - float(v["draft"])
                model.Add(sm == int(round(100*max(depth_margin,0)))).OnlyEnforceIf(x[(i,j)])
                model.Add(sm == 0).OnlyEnforceIf(x[(i,j)].Not())
                safety_terms.append(sm)

    # Lineales
    if groups:
        for i,_ in enumerate(V):
            for g in groups:
                if (i,g) in y:
                    assign_terms.append(y[(i,g)])
                    # compactación: penaliza inicios tardíos
                    waste_terms.append(s[(i,g)])
                    # margen de calado (cm, sólo positivo)
                    depth = float(Gdepth.get(g, np.nan))
                    if not np.isnan(depth):
                        sm = model.NewIntVar(0, 100000, f"sm_l_{i}_{g}")
                        margin = depth - (policy["ukc"]+policy["tide_safety"]) - float(V[i]["draft"])
                        model.Add(sm == int(round(100*max(margin,0)))).OnlyEnforceIf(y[(i,g)])
                        model.Add(sm == 0).OnlyEnforceIf(y[(i,g)].Not())
                        safety_terms.append(sm)

    model.Maximize(1_000_000*sum(assign_terms) - 1_000*sum(waste_terms) + 1*sum(safety_terms))

    # ---------- Resolver ----------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 8
    res = solver.Solve(model)

    # ---------- Recoger solución ----------
    assigned_rows: List[Dict] = []
    if res in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # slots
        for i, v in enumerate(V):
            for j, b in enumerate(Bd):
                if (i,j) in x and solver.Value(x[(i,j)]) == 1:
