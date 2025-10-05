/# solver.py (optimizado)
# Motor de optimización para asignar barcos a atraques usando OR-Tools (CP-SAT)
# - Soporta berths discretos tipo "finger"/"thead" (uno por slot)
# - Soporta "lineales" (costado) con empaque 1D por grupos (group_id)
# - Objetivo jerárquico: max asignados -> min desperdicio -> max márgenes

from __future__ import annotations
from ortools.sat.python import cp_model
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

DEFAULT_POLICY = {
    "alpha": 1.3,
    "beta": 0.0,
    "ukc": 0.3,
    "fender_mono": 0.25,
    "fender_cata": 0.40,
    "end_margin": 0.5,
    "tide_safety": 0.1,
    "min_gap_linear": 0.4  # separación mínima entre barcos en costado
}

# -----------------------
# Validación de entradas
# -----------------------
class BerthingValidator:
    def __init__(self) -> None:
        self.vessel_cols = ["vessel_id","loa","beam","draft","type","power_kw"]
        self.berth_cols  = ["berth_id","length","slip_width","depth","fairway_width","power_kw","type"]
        # opcionales para lineales:
        self.optional_berth_cols = ["group_id","position","zone","price_day"]

    def _check_required(self, df: pd.DataFrame, cols: List[str], label: str, errors: List[str]):
        for c in cols:
            if c not in df.columns:
                errors.append(f"[{label}] Falta columna requerida: `{c}`")

    class BerthingValidator:
    def __init__(self) -> None:
        self.vessel_cols = ["vessel_id","loa","beam","draft","type","power_kw"]
        self.berth_cols  = ["berth_id","length","slip_width","depth","fairway_width","power_kw","type"]
        # opcionales para lineales:
        self.optional_berth_cols = ["group_id","position","zone","price_day"]

    def _check_required(self, df: pd.DataFrame, cols: List[str], label: str, errors: List[str]):
        for c in cols:
            if c not in df.columns:
                errors.append(f"[{label}] Falta columna requerida: `{c}`")

    def validate_vessels(self, vessels: pd.DataFrame) -> tuple[bool, List[str]]:
        errors: List[str] = []
        self._check_required(vessels, self.vessel_cols, "Barcos", errors)
        if errors:
            return False, errors

        # Asegurar tipos numéricos (coerce -> NaN si algo no es número)
        num_cols = ["loa","beam","draft","power_kw"]
        numeric = vessels[num_cols].apply(pd.to_numeric, errors="coerce")

        # Nulos en numéricos
        nulls = numeric.isnull().any()
        for col, isnull in nulls.items():
            if isnull:
                errors.append(f"[Barcos] Valores nulos o no numéricos en `{col}`")

        # Valores negativos
        if (numeric < 0).any().any():
            errors.append("[Barcos] Hay valores negativos en métricas (loa/beam/draft/power_kw)")

        # IDs
        if "vessel_id" not in vessels.columns:
            errors.append("[Barcos] Falta `vessel_id`")
        else:
            if vessels["vessel_id"].isnull().any():
                errors.append("[Barcos] Hay `vessel_id` vacíos")
            if vessels["vessel_id"].duplicated().any():
                dups = vessels.loc[vessels["vessel_id"].duplicated(), "vessel_id"].unique().tolist()
                errors.append(f"[Barcos] IDs duplicados: {dups}")

        return len(errors) == 0, errors

    def validate_berths(self, berths: pd.DataFrame) -> tuple[bool, List[str]]:
        errors: List[str] = []
        self._check_required(berths, self.berth_cols, "Atraques", errors)
        if errors:
            return False, errors

        # En los atraques, algunas columnas pueden ser opcionales según tipo:
        # - Requerimos SIEMPRE: length, depth (numéricos)
        # - Permitimos NaN: slip_width, fairway_width, power_kw (porque el solver ya los trata como opcionales)
        required_numeric = ["length","depth"]
        optional_numeric = ["slip_width","fairway_width","power_kw"]

        # Convierte a numérico
        req_df = berths[required_numeric].apply(pd.to_numeric, errors="coerce")
        opt_df = berths[optional_numeric].apply(pd.to_numeric, errors="coerce")

        # Nulos en requeridos
        nulls_req = req_df.isnull().any()
        for col, isnull in nulls_req.items():
            if isnull:
                errors.append(f"[Atraques] Valores nulos o no numéricos en requerido `{col}`")

        # Negativos en requeridos
        if (req_df < 0).any().any():
            errors.append("[Atraques] Hay valores negativos en campos requeridos (length/depth)")

        # IDs
        if "berth_id" not in berths.columns:
            errors.append("[Atraques] Falta `berth_id`")
        else:
            if berths["berth_id"].isnull().any():
                errors.append("[Atraques] Hay `berth_id` vacíos")
            if berths["berth_id"].duplicated().any():
                dups = berths.loc[berths["berth_id"].duplicated(), "berth_id"].unique().tolist()
                errors.append(f"[Atraques] IDs duplicados: {dups}")

        # Si hay tramos lineales, exige group_id
        if "type" in berths.columns:
            has_linear = berths["type"].astype(str).str.lower().isin(["linear","costado"]).any()
            if has_linear and "group_id" not in berths.columns:
                errors.append("[Atraques] Hay filas `type=linear/costado` y falta columna `group_id`")

        return len(errors) == 0, errors


# -----------------------
# Compatibilidad
# -----------------------
def _get_fender(vessel_type: str, policy: dict) -> float:
    if isinstance(vessel_type, str) and vessel_type.lower().startswith("cata"):
        return policy["fender_cata"]
    return policy["fender_mono"]

def _pair_feasibility(v: dict, b: dict, policy: dict) -> tuple[bool, List[str]]:
    reasons: List[str] = []
    ok = True

    # Longitud útil
    if v["loa"] > (b["length"] - policy["end_margin"]):
        ok = False; reasons.append("longitud: LOA excede longitud útil")

    # Calado con UKC + marea
    depth_op = b["depth"] - (policy["ukc"] + policy["tide_safety"])
    if v["draft"] > depth_op:
        ok = False; reasons.append("calado: insuficiente con margen UKC+marea")

    # Ancho de slip (si aplica; en lineales/slips abiertos se puede poner NaN o gran valor)
    if pd.notnull(b.get("slip_width", np.nan)):
        beam_required = v["beam"] + 2*_get_fender(v.get("type",""), policy)
        if beam_required > b["slip_width"]:
            ok = False; reasons.append("manga: no cabe con defensas")

    # Potencia
    if pd.notnull(b.get("power_kw", np.nan)):
        if v["power_kw"] > b["power_kw"]:
            ok = False; reasons.append("potencia: insuficiente")

    # Calle maniobra
    if pd.notnull(b.get("fairway_width", np.nan)):
        loa_max = (b["fairway_width"] - policy["beta"]) / max(policy["alpha"], 1e-6)
        if v["loa"] > loa_max:
            ok = False; reasons.append("calle: LOA > LOA_max por maniobra")

    return ok, reasons

def calculate_compatibility_matrix(vessels_df: pd.DataFrame, berths_df: pd.DataFrame, policy: dict) -> pd.DataFrame:
    rows = []
    for _, v in vessels_df.iterrows():
        for _, b in berths_df.iterrows():
            is_ok, reasons = _pair_feasibility(v.to_dict(), b.to_dict(), policy)
            rows.append({
                "vessel_id": v["vessel_id"],
                "berth_id": b["berth_id"],
                "compatible": bool(is_ok),
                "reasons": "; ".join(reasons) if reasons else "Compatible",
                "length_margin": (b["length"] - v["loa"] - policy["end_margin"]),
                "depth_margin": (b["depth"] - (policy["ukc"] + policy["tide_safety"]) - v["draft"]),
                "width_margin": (b["slip_width"] - (v["beam"] + 2*_get_fender(v["type"], policy))) if pd.notnull(b.get("slip_width", np.nan)) else np.nan,
                "power_margin": (b["power_kw"] - v["power_kw"]) if pd.notnull(b.get("power_kw", np.nan)) else np.nan
            })
    return pd.DataFrame(rows)

def _precheck_summary(v: dict, berths: pd.DataFrame, policy: dict) -> str:
    counts = {"longitud":0,"calado":0,"manga":0,"potencia":0,"calle":0}
    best = None; best_issues = 99; best_margins = {}
    for _, b in berths.iterrows():
        ok, reasons = _pair_feasibility(v, b.to_dict(), policy)
        if ok:
            return f"Compatible con {b['berth_id']}, no elegido por optimización global"
        n = len(reasons)
        if n < best_issues:
            best = b["berth_id"]; best_issues = n
            best_margins = {
                "length_margin": b["length"] - v["loa"] - policy["end_margin"],
                "depth_margin": b["depth"] - (policy["ukc"]+policy["tide_safety"]) - v["draft"],
                "width_margin": (b["slip_width"] - (v["beam"] + 2*_get_fender(v.get("type",""), policy))) if pd.notnull(b.get("slip_width", np.nan)) else np.nan,
                "power_margin": (b["power_kw"] - v["power_kw"]) if pd.notnull(b.get("power_kw", np.nan)) else np.nan
            }
        for r in reasons:
            for k in counts:
                if r.startswith(k):
                    counts[k]+=1
    detail = ", ".join([f"{k}:{v}" for k,v in counts.items()])
    margins = " | ".join([f"{k}={best_margins.get(k, np.nan):+.2f}" for k in ["length_margin","depth_margin","width_margin","power_margin"]])
    return f"Mejor candidato {best} (issues={best_issues}). Causas: {detail}. Márgenes: {margins}"

# -----------------------
# Solver principal
# -----------------------
def solve_assignment(
    vessels_df: pd.DataFrame,
    berths_df: pd.DataFrame,
    policy: dict | None = None,
    time_limit: int = 20,
    prioritize_by: Optional[str] = None
):
    """
    Asigna barcos a:
      - Atraques discretos (uno por slot): type in {'finger','thead'}
      - Tramos lineales agrupados por 'group_id' con type == 'linear' (costado)
    """
    policy = policy or DEFAULT_POLICY

    vessels = vessels_df.copy()
    berths = berths_df.copy()
    validator = BerthingValidator()
    ok_v, err_v = validator.validate_vessels(vessels)
    ok_b, err_b = validator.validate_berths(berths)
    if not ok_v or not ok_b:
        raise ValueError("Errores de validación:\n" + "\n".join(err_v+err_b))

    vessels["type"] = vessels["type"].astype(str).str.lower()
    if prioritize_by and prioritize_by in vessels.columns:
        vessels = vessels.sort_values(prioritize_by, ascending=False)

    # Separar berths por tipo
    berths["type"] = berths["type"].astype(str).str.lower()
    discrete = berths[berths["type"].isin(["finger","thead"])].reset_index(drop=True)
    linear   = berths[berths["type"].isin(["linear","costado"])].reset_index(drop=True)

    V = vessels.to_dict("records")
    Bd = discrete.to_dict("records")
    Bl = linear.to_dict("records")

    model = cp_model.CpModel()

    # -----------------------
    # Parte A: slots discretos
    # -----------------------
    x = {}  # x[i,j] para discretos
    compd = {}
    for i, v in enumerate(V):
        for j, b in enumerate(Bd):
            ok, _ = _pair_feasibility(v, b, policy)
            compd[(i,j)] = ok
            if ok:
                x[(i,j)] = model.NewBoolVar(f"x_d_{i}_{j}")

    # Un barco no puede ser asignado a más de una cosa (discreto o lineal)
    # Crearemos más variables para lineal y luego haremos la suma.

    # Uno por slot
    for j, _ in enumerate(Bd):
        model.Add(sum(x[(i,j)] for i,_ in enumerate(V) if (i,j) in x) <= 1)

    # -----------------------
    # Parte B: tramos lineales (costado)
    # Agrupar por group_id; dentro de cada grupo se empaca 1D
    # Requisito: cada fila en linear debe tener: group_id (str), length (longitud total del tramo).
    # Si hay múltiples filas por el mismo group_id se suman por seguridad.
    # -----------------------
    y = {} # y[i,g] = 1 si barco i va al grupo g (lineal)
    s = {} # start position en cm
    groups = []
    if len(Bl)>0:
        if "group_id" not in linear.columns:
            raise ValueError("Atraques tipo 'linear' requieren columna `group_id`.")
        # longitud por grupo
        Lg = linear.groupby("group_id")["length"].sum().to_dict()
        groups = list(Lg.keys())

        # Variables por barco-grupo
        for i, v in enumerate(V):
            for g in groups:
                # factibilidad mínima para ese grupo: usar un berth representativo del grupo (primero)
                # aplicamos reglas salvo slip_width (normalmente NaN en costado)
                b_rep = {
                    "length": Lg[g], "slip_width": np.nan,
                    "depth": float(linear[linear["group_id"]==g]["depth"].min()),
                    "fairway_width": float(linear[linear["group_id"]==g]["fairway_width"].min()),
                    "power_kw": float(linear[linear["group_id"]==g]["power_kw"].max()) if "power_kw" in linear.columns else np.nan,
                    "type": "linear"
                }
                ok, _ = _pair_feasibility(v, b_rep, policy)
                if ok:
                    y[(i,g)] = model.NewBoolVar(f"y_l_{i}_{g}")
                    # posición inicial en cm para estabilidad
                    s[(i,g)] = model.NewIntVar(0, int(round(100*Lg[g])), f"s_{i}_{g}")

        # No solapamiento en cada grupo (NoOverlap con IntervalVar)
        for g in groups:
            intervals = []
            # recolectar candidatos
            cand = [(i,(i,g)) for i,_ in enumerate(V) if (i,g) in y]
            for i, key in cand:
                size_cm = int(round(100*(V[i]["loa"] + policy["min_gap_linear"])))
                # Si asignado a g, crear intervalo reificado
                interval = model.NewOptionalFixedSizeIntervalVar(
                    s[key], size_cm, int(round(100*Lg[g])), y[key], f"I_{i}_{g}"
                )
                intervals.append(interval)
            # no-overlap se garantiza con interval vars opcionales
            model.AddNoOverlap(intervals)

            # límites del tramo
            for i,_ in enumerate(V):
                if (i,g) in y:
                    model.Add(s[(i,g)] + int(round(100*(V[i]["loa"]))) <= int(round(100*Lg[g]))).OnlyEnforceIf(y[(i,g)])

    # -----------------------
    # Vínculo: cada barco a lo sumo a un lugar (discreto o un grupo lineal)
    # -----------------------
    for i,_ in enumerate(V):
        lhs = []
        lhs += [x[(i,j)] for j,_ in enumerate(Bd) if (i,j) in x]
        lhs += [y[(i,g)] for g in groups if (i,g) in y]
        if lhs:
            model.Add(sum(lhs) <= 1)

    # -----------------------
    # Objetivo multi-nivel
    # 1) Max asignados
    # 2) Min desperdicio (discretos: L-LOA; lineal: huecos implícitos)
    # 3) Max márgenes de seguridad (sum de margins positivos)
    # -----------------------
    assign_terms = []
    waste_terms  = []
    safety_terms = []

    # Discretos
    for i, v in enumerate(V):
        for j, b in enumerate(Bd):
            if (i,j) in x:
                assign_terms.append(x[(i,j)])
                # desperdicio
                waste_val = max(b["length"] - v["loa"], 0.0)
                w = model.NewIntVar(0, 100000, f"w_d_{i}_{j}")
                model.Add(w == int(round(100*waste_val))).OnlyEnforceIf(x[(i,j)])
                model.Add(w == 0).OnlyEnforceIf(x[(i,j)].Not())
                waste_terms.append(w)
                # safety margins (profundidad + ancho si aplica)
                depth_margin = b["depth"] - (policy["ukc"]+policy["tide_safety"]) - v["draft"]
                sm = model.NewIntVar(0, 100000, f"sm_d_{i}_{j}")
                model.Add(sm == int(round(100*max(depth_margin,0)))).OnlyEnforceIf(x[(i,j)])
                model.Add(sm == 0).OnlyEnforceIf(x[(i,j)].Not())
                safety_terms.append(sm)

    # Lineales
    for i,_ in enumerate(V):
        for g in groups:
            if (i,g) in y:
                assign_terms.append(y[(i,g)])
                # en lineal, penalizamos inicio lejos para compactación ligera (reduce huecos)
                # (heurística simple)
                s_cm = s[(i,g)]
                waste_terms.append(s_cm)  # cuanto más tarde empieza, más "waste"
                # márgenes: usamos solo profundidad aquí
                depth = float(linear[linear["group_id"]==g]["depth"].min())
                depth_margin = depth - (policy["ukc"]+policy["tide_safety"]) - V[i]["draft"]
                sm = model.NewIntVar(0, 100000, f"sm_l_{i}_{g}")
                model.Add(sm == int(round(100*max(depth_margin,0)))).OnlyEnforceIf(y[(i,g)])
                model.Add(sm == 0).OnlyEnforceIf(y[(i,g)].Not())
                safety_terms.append(sm)

    # única función objetivo (ponderada grande para jerarquía)
    model.Maximize(
        1_000_000 * sum(assign_terms)
        - 1_000 * sum(waste_terms)
        + 1 * sum(safety_terms)
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 8
    result = solver.Solve(model)

    assigned_rows = []
    used_discrete = set()
    used_groups: Dict[str, List[Tuple[str, float, float]]] = {}

    if result in (cp_model.OPTIMAL, cp_model.FEASIBLE):
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
                        "utilization_%": round(100* v["loa"]/max(b["length"],1e-6),1),
                        "length_margin": round(b["length"] - v["loa"],2),
                        "depth_margin": round(b["depth"] - (policy["ukc"]+policy["tide_safety"]) - v["draft"],2)
                    })
                    used_discrete.add(b["berth_id"])

        # Lineales
        for i,_ in enumerate(V):
            for g in groups:
                if (i,g) in y and solver.Value(y[(i,g)]) == 1:
                    start_m = solver.Value(s[(i,g)]) / 100.0
                    assigned_rows.append({
                        "vessel_id": V[i]["vessel_id"],
                        "berth_id": str(g),
                        "mode": "linear",
                        "start_m": round(start_m,2),
                        "loa": V[i]["loa"],
                        "depth_margin": round(float(linear[linear["group_id"]==g]["depth"].min()) - (policy["ukc"]+policy["tide_safety"]) - V[i]["draft"],2)
                    })
                    used_groups.setdefault(g, []).append((V[i]["vessel_id"], start_m, start_m+V[i]["loa"]))

    # No asignados + razones
    assigned_ids = {r["vessel_id"] for r in assigned_rows}
    unassigned_rows = []
    all_berths_for_reason = pd.concat([discrete, linear], ignore_index=True) if len(linear)>0 else discrete
    for i, v in enumerate(V):
        if v["vessel_id"] not in assigned_ids:
            reason = _precheck_summary(v, all_berths_for_reason, policy)
            unassigned_rows.append({"vessel_id": v["vessel_id"], "reason": reason})

    stats = {
        "n_vessels": len(V),
        "n_berths": len(berths),
        "assigned": len(assigned_rows),
        "unassigned": len(unassigned_rows),
        "occupancy_pct": round(100 * (len(used_discrete) + sum(len(used_groups.get(g,[]))>0 for g in groups)) / max(1,len(berths)),1)
    }

    # análisis de motivos
    if unassigned_rows:
        rej = {}
        for r in unassigned_rows:
            t = r["reason"].lower()
            for key in ["longitud","calado","manga","potencia","calle"]:
                if key in t:
                    rej[key] = rej.get(key,0)+1
        stats["rejection_reasons"] = rej

    return pd.DataFrame(assigned_rows), pd.DataFrame(unassigned_rows), stats
