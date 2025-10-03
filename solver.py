 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/solver_mejorado.py b/solver.py
index e77a47bf0d42330720ea11c170736310308eb352..e1d23ed06b5dd8e59dd5998d45e245f83a8bc315 100644
--- a/solver_mejorado.py
+++ b/solver.py
@@ -33,63 +33,95 @@ class BerthingValidator:
 
     def _check_required(self, df: pd.DataFrame, cols: List[str], label: str, errors: List[str]):
         for c in cols:
             if c not in df.columns:
                 errors.append(f"[{label}] Falta columna requerida: `{c}`")
 
     def validate_vessels(self, vessels: pd.DataFrame) -> tuple[bool, List[str]]:
         errors: List[str] = []
         self._check_required(vessels, self.vessel_cols, "Barcos", errors)
         if errors: return False, errors
         # tipos
         num_cols = ["loa","beam","draft","power_kw"]
         bad = vessels[num_cols].isnull().any()
         for c in num_cols[bad.values]:
             errors.append(f"[Barcos] Valores nulos en `{c}`")
         if (vessels[num_cols] < 0).any().any():
             errors.append("[Barcos] Hay valores negativos en métricas")
         if vessels["vessel_id"].duplicated().any():
             dups = vessels[vessels["vessel_id"].duplicated()]["vessel_id"].unique().tolist()
             errors.append(f"[Barcos] IDs duplicados: {dups}")
         return len(errors)==0, errors
 
     def validate_berths(self, berths: pd.DataFrame) -> tuple[bool, List[str]]:
         errors: List[str] = []
         self._check_required(berths, self.berth_cols, "Atraques", errors)
-        # opcionales presentes? ok.
-        if errors: return False, errors
-        num_cols = ["length","slip_width","depth","fairway_width","power_kw"]
-        bad = berths[num_cols].isnull().any()
-        for c in num_cols[bad.values]:
-            errors.append(f"[Atraques] Valores nulos en `{c}`")
-        if (berths[num_cols] < 0).any().any():
-            errors.append("[Atraques] Hay valores negativos en métricas")
+        if errors:
+            return False, errors
+
+        def _label(row: pd.Series, idx: int) -> str:
+            bid = row.get("berth_id")
+            return f"[Atraques] {bid}" if pd.notnull(bid) else f"[Atraques] fila {idx}"
+
+        for idx, row in berths.iterrows():
+            label = _label(row, idx)
+            # Valores básicos obligatorios
+            for col in ["length", "depth"]:
+                if pd.isnull(row.get(col)):
+                    errors.append(f"{label}: valor nulo en `{col}`")
+                elif row[col] < 0:
+                    errors.append(f"{label}: valor negativo en `{col}`")
+
+            berth_type = str(row.get("type", "")).lower()
+
+            # slip_width solo obligatorio para slots discretos
+            slip_width = row.get("slip_width")
+            if berth_type in {"finger", "thead"}:
+                if pd.isnull(slip_width):
+                    errors.append(f"{label}: `slip_width` requerido para atraques discretos")
+                elif slip_width <= 0:
+                    errors.append(f"{label}: `slip_width` debe ser positivo")
+            elif pd.notnull(slip_width) and slip_width < 0:
+                errors.append(f"{label}: `slip_width` no puede ser negativo")
+
+            # fairway/power pueden omitirse, pero si existen deben ser positivos
+            fairway = row.get("fairway_width")
+            if pd.notnull(fairway) and fairway <= 0:
+                errors.append(f"{label}: `fairway_width` debe ser positivo")
+
+            power = row.get("power_kw")
+            if pd.notnull(power) and power < 0:
+                errors.append(f"{label}: `power_kw` no puede ser negativo")
+
+            if berth_type in {"linear", "costado"} and pd.isnull(row.get("group_id")):
+                errors.append(f"{label}: tramos lineales requieren `group_id`")
+
         if berths["berth_id"].duplicated().any():
             dups = berths[berths["berth_id"].duplicated()]["berth_id"].unique().tolist()
             errors.append(f"[Atraques] IDs duplicados: {dups}")
-        # si hay lineales, requieren group_id y length total por grupo (sum de length si modela tramos contiguos)
-        return len(errors)==0, errors
+
+        return len(errors) == 0, errors
 
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
diff --git a/solver_mejorado.py b/solver.py
index e77a47bf0d42330720ea11c170736310308eb352..e1d23ed06b5dd8e59dd5998d45e245f83a8bc315 100644
--- a/solver_mejorado.py
+++ b/solver.py
@@ -99,69 +131,73 @@ def _pair_feasibility(v: dict, b: dict, policy: dict) -> tuple[bool, List[str]]:
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
-                "width_margin": (b["slip_width"] - (v["beam"] + 2*_get_fender(v["type"], policy))) if pd.notnull(b.get("slip_width", np.nan)) else np.nan,
+                "width_margin": (
+                    b["slip_width"] - (v["beam"] + 2 * _get_fender(v["type"], policy))
+                ) if pd.notnull(b.get("slip_width", np.nan)) else np.nan,
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
-                "width_margin": (b["slip_width"] - (v["beam"] + 2*_get_fender(v.get("type",""), policy))) if pd.notnull(b.get("slip_width", np.nan)) else np.nan,
+                "width_margin": (
+                    b["slip_width"] - (v["beam"] + 2 * _get_fender(v.get("type", ""), policy))
+                ) if pd.notnull(b.get("slip_width", np.nan)) else np.nan,
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
 
EOF
)