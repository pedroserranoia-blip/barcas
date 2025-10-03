 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/app_mejorado_v3.py b/app_mejorado_v3.py
index 86bcacec3761361f8e3b2bfeda10a4147d5c3c1e..61248f04fff25b5e94c1e894f76937fb778797ec 100644
--- a/app_mejorado_v3.py
+++ b/app_mejorado_v3.py
@@ -1,234 +1,501 @@
-# app.py — UI centrada en eficiencia de ubicación de embarcaciones
-# - Soporta slots discretos (finger/thead) y tramos lineales (costado)
-# - KPIs, validación, compatibilidad, descarga de resultados
-# - Visualización de tramos lineales (orden/posición) con Plotly
+"""Aplicación Streamlit para optimizar la ubicación de embarcaciones.
+
+Incluye:
+* Carga y validación de datos
+* Ejecución del solver con métricas resumidas
+* Visualización de asignaciones en tramos lineales
+* Matriz de compatibilidad y análisis de rechazos
+"""
 
 from __future__ import annotations
+
 import io
 import json
+from typing import Optional
+
 import pandas as pd
-import numpy as np
-import streamlit as st
-import plotly.graph_objects as go
 import plotly.express as px
+import plotly.graph_objects as go
+import streamlit as st
 
-from solver import (
-    solve_assignment,
-    calculate_compatibility_matrix,
+from solver_mejorado import (
     BerthingValidator,
-    DEFAULT_POLICY
+    DEFAULT_POLICY,
+    calculate_compatibility_matrix,
+    solve_assignment,
 )
 
 st.set_page_config(page_title="Optimización de Atraques", layout="wide")
 st.title("🛥️ Optimización de ubicación de embarcaciones")
 
-# ============================
-# Sidebar: Parámetros y ayuda
-# ============================
-with st.sidebar:
-    st.header("Parámetros de maniobra/seguridad")
-    alpha = st.number_input("α (calle)", value=float(DEFAULT_POLICY["alpha"]), step=0.1,
-                            help="W_calle ≥ α · LOA + β")
-    beta = st.number_input("β (calle)", value=float(DEFAULT_POLICY["beta"]), step=0.1)
-    ukc = st.number_input("UKC (margen bajo quilla, m)", value=float(DEFAULT_POLICY["ukc"]), step=0.1)
-    tide_safety = st.number_input("Margen marea (m)", value=float(DEFAULT_POLICY.get("tide_safety",0.1)), step=0.1)
-    f_mono = st.number_input("Defensa/lado monocasco (m)", value=float(DEFAULT_POLICY["fender_mono"]), step=0.05)
-    f_cata = st.number_input("Defensa/lado catamarán (m)", value=float(DEFAULT_POLICY["fender_cata"]), step=0.05)
-    end_m = st.number_input("Margen proa+popa en finger (m)", value=float(DEFAULT_POLICY["end_margin"]), step=0.1)
-    min_gap_linear = st.number_input("Separación en costado (m)", value=float(DEFAULT_POLICY.get("min_gap_linear",0.4)), step=0.1,
-                                     help="Hueco mínimo entre barcos en tramos lineales (costado).")
-
-    st.divider()
-    st.subheader("Ejecución")
-    time_limit = st.slider("Tiempo límite del solver (s)", 5, 120, 20, 5)
-    prioritize_by = st.selectbox(
-        "Priorizar barcos por… (opcional)",
-        options=["(ninguno)", "loa", "draft", "power_kw"],
-        index=0,
-        help="Ordena barcos para orientar la búsqueda (no impone prioridad dura)."
+LINEAR_COLOR_CYCLE = px.colors.qualitative.Safe
+
+
+# ---------------------------------------------------------------------------
+# Utilidades
+# ---------------------------------------------------------------------------
+
+def read_any(file) -> Optional[pd.DataFrame]:
+    """Leer CSV/Excel sin importar la extensión."""
+    if file is None:
+        return None
+    suffix = file.name.lower().split(".")[-1]
+    if suffix in ("xls", "xlsx"):
+        return pd.read_excel(file)
+    return pd.read_csv(file)
+
+
+def _ensure_policy() -> dict:
+    """Devuelve la política seleccionada en la barra lateral."""
+    with st.sidebar:
+        st.header("Parámetros de maniobra/seguridad")
+        alpha = st.number_input(
+            "α (calle)",
+            value=float(DEFAULT_POLICY["alpha"]),
+            step=0.1,
+            help="W_calle ≥ α · LOA + β",
+        )
+        beta = st.number_input("β (calle)", value=float(DEFAULT_POLICY["beta"]), step=0.1)
+        ukc = st.number_input(
+            "UKC (margen bajo quilla, m)",
+            value=float(DEFAULT_POLICY["ukc"]),
+            step=0.1,
+        )
+        tide_safety = st.number_input(
+            "Margen marea (m)",
+            value=float(DEFAULT_POLICY.get("tide_safety", 0.1)),
+            step=0.1,
+        )
+        f_mono = st.number_input(
+            "Defensa/lado monocasco (m)",
+            value=float(DEFAULT_POLICY["fender_mono"]),
+            step=0.05,
+        )
+        f_cata = st.number_input(
+            "Defensa/lado catamarán (m)",
+            value=float(DEFAULT_POLICY["fender_cata"]),
+            step=0.05,
+        )
+        end_m = st.number_input(
+            "Margen proa+popa en finger (m)",
+            value=float(DEFAULT_POLICY["end_margin"]),
+            step=0.1,
+        )
+        min_gap_linear = st.number_input(
+            "Separación en costado (m)",
+            value=float(DEFAULT_POLICY.get("min_gap_linear", 0.4)),
+            step=0.1,
+            help="Hueco mínimo entre barcos en tramos lineales (costado).",
+        )
+
+        st.divider()
+        st.subheader("Ejecución")
+        time_limit = st.slider("Tiempo límite del solver (s)", 5, 120, 20, 5)
+        prioritize_by = st.selectbox(
+            "Priorizar barcos por… (opcional)",
+            options=["(ninguno)", "loa", "draft", "power_kw"],
+            index=0,
+            help="Ordena barcos para orientar la búsqueda (no impone prioridad dura).",
+        )
+        if prioritize_by == "(ninguno)":
+            prioritize_by = None
+
+    st.session_state["time_limit"] = time_limit
+    st.session_state["prioritize_by"] = prioritize_by
+    policy = {
+        "alpha": alpha,
+        "beta": beta,
+        "ukc": ukc,
+        "tide_safety": tide_safety,
+        "fender_mono": f_mono,
+        "fender_cata": f_cata,
+        "end_margin": end_m,
+        "min_gap_linear": min_gap_linear,
+    }
+    st.session_state["policy"] = policy
+    return policy
+
+
+def _plot_linear_assignments(assignments: pd.DataFrame, berths_df: pd.DataFrame) -> Optional[go.Figure]:
+    if assignments is None or assignments.empty:
+        return None
+    if "mode" not in assignments.columns or "start_m" not in assignments.columns:
+        return None
+    linear_df = assignments[assignments["mode"] == "linear"].copy()
+    if linear_df.empty:
+        return None
+
+    linear_slots = (
+        berths_df.assign(type=berths_df["type"].astype(str).str.lower())
+        .loc[lambda df: df["type"].isin(["linear", "costado"])]
+        .copy()
+    )
+    if linear_slots.empty:
+        return None
+
+    groups = linear_df["berth_id"].unique().tolist()
+    fig = go.Figure()
+    for idx, group in enumerate(sorted(groups)):
+        group_assignments = linear_df[linear_df["berth_id"] == group]
+        total_length = float(linear_slots.loc[linear_slots["group_id"] == group, "length"].sum())
+        if total_length <= 0:
+            continue
+        baseline = idx * 2
+        fig.add_shape(
+            type="rect",
+            x0=0,
+            x1=total_length,
+            y0=baseline - 0.4,
+            y1=baseline + 0.4,
+            fillcolor="#f0f2f6",
+            line=dict(color="#7f8c8d"),
+            layer="below",
+        )
+        fig.add_annotation(
+            x=total_length / 2,
+            y=baseline + 0.6,
+            text=f"Tramo {group} ({total_length:.1f} m)",
+            showarrow=False,
+            font=dict(size=12, color="#34495e"),
+        )
+        for jdx, row in enumerate(group_assignments.itertuples(index=False)):
+            color = LINEAR_COLOR_CYCLE[jdx % len(LINEAR_COLOR_CYCLE)]
+            fig.add_trace(
+                go.Bar(
+                    x=[row.loa],
+                    y=[baseline],
+                    base=row.start_m,
+                    name=row.vessel_id,
+                    orientation="h",
+                    marker=dict(color=color, line=dict(color="#2c3e50", width=1)),
+                    hovertemplate=(
+                        "Barco: %{customdata[0]}<br>Inicio: %{base:.2f} m"
+                        "<br>LOA: %{x:.2f} m"
+                    ),
+                    customdata=[[row.vessel_id]],
+                    showlegend=False,
+                )
+            )
+    fig.update_layout(
+        barmode="overlay",
+        bargap=0.2,
+        title="Distribución en tramos lineales",
+        xaxis_title="Metros desde el origen del tramo",
+        yaxis=dict(
+            showticklabels=False,
+            showgrid=False,
+            zeroline=False,
+        ),
+        height=max(350, 200 * len(groups)),
+        margin=dict(l=40, r=20, t=60, b=40),
     )
-    if prioritize_by == "(ninguno)":
-        prioritize_by = None
-
-policy = {
-    "alpha": alpha,
-    "beta": beta,
-    "ukc": ukc,
-    "tide_safety": tide_safety,
-    "fender_mono": f_mono,
-    "fender_cata": f_cata,
-    "end_margin": end_m,
-    "min_gap_linear": min_gap_linear
-}
-
-# ============================
-# Datos de ejemplo (fallback)
-# ============================
+    return fig
+
+
+def _render_stats(stats: dict) -> None:
+    k1, k2, k3, k4 = st.columns(4)
+    k1.metric("Barcos", stats.get("n_vessels", 0))
+    k2.metric("Atraques", stats.get("n_berths", 0))
+    k3.metric("Asignados", stats.get("assigned", 0))
+    k4.metric("Ocupación atraques (%)", stats.get("occupancy_pct", 0))
+
+
+_ensure_policy()
+
+# ---------------------------------------------------------------------------
+# Datos demo por defecto
+# ---------------------------------------------------------------------------
 DEMO_VESSELS = """vessel_id,loa,beam,draft,type,power_kw
 B1,11.0,3.6,1.8,mono,6
 B2,14.0,4.2,2.0,mono,8
 B3,12.5,7.0,1.3,cata,10
 B4,9.5,3.2,1.5,mono,4
 B5,16.0,4.7,2.3,mono,12
 B6,10.0,3.4,1.6,mono,5
 B7,13.2,4.1,2.1,mono,9
 B8,8.8,3.0,1.2,mono,3
 """
 
 DEMO_BERTHS = """berth_id,length,slip_width,depth,fairway_width,power_kw,type,group_id
 A1,12.0,4.0,2.5,35.0,8,finger,
 A2,14.0,4.5,2.7,35.0,12,finger,
 A3,10.5,3.6,1.6,35.0,6,finger,
 A4,18.0,6.5,2.8,28.0,16,finger,
 A5,16.0,4.5,2.4,28.0,12,finger,
 T1,20.0,7.0,3.0,80.0,16,thead,
 L1,30.0,,3.0,60.0,,linear,LINEAL-1
 """
 
-# ============================
+# ---------------------------------------------------------------------------
 # Tabs principales
-# ============================
+# ---------------------------------------------------------------------------
 tabs = st.tabs(["Datos", "Optimización", "Compatibilidad", "Análisis", "Ayuda"])
 
-# ============================
-# Tab: Datos
-# ============================
+# ---------------------------------------------------------------------------
+# Tab 1: Datos
+# ---------------------------------------------------------------------------
 with tabs[0]:
     st.subheader("1) Carga tus datos de Barcos y Atraques")
 
     col1, col2 = st.columns(2)
     with col1:
-        f_vess = st.file_uploader("Barcos (CSV o Excel)", type=["csv","xls","xlsx"], key="vess")
+        f_vess = st.file_uploader("Barcos (CSV o Excel)", type=["csv", "xls", "xlsx"], key="vess")
     with col2:
-        f_berths = st.file_uploader("Atraques (CSV o Excel)", type=["csv","xls","xlsx"], key="berths")
-
-    def read_any(file):
-        if file is None:
-            return None
-        suffix = file.name.lower().split(".")[-1]
-        if suffix in ("xls","xlsx"):
-            return pd.read_excel(file)
-        return pd.read_csv(file)
-
-    if f_vess is not None:
-        vessels_df = read_any(f_vess)
-    else:
-        vessels_df = pd.read_csv(io.StringIO(DEMO_VESSELS))
+        f_berths = st.file_uploader("Atraques (CSV o Excel)", type=["csv", "xls", "xlsx"], key="berths")
 
-    if f_berths is not None:
-        berths_df = read_any(f_berths)
-    else:
-        berths_df = pd.read_csv(io.StringIO(DEMO_BERTHS))
+    vessels_df = read_any(f_vess) or pd.read_csv(io.StringIO(DEMO_VESSELS))
+    berths_df = read_any(f_berths) or pd.read_csv(io.StringIO(DEMO_BERTHS))
 
-    # Vista previa
     st.write("**Barcos** (primeras filas)")
     st.dataframe(vessels_df.head(20), use_container_width=True)
     st.write("**Atraques** (primeras filas)")
     st.dataframe(berths_df.head(20), use_container_width=True)
 
-    # Validación
     st.subheader("Validación rápida")
     validator = BerthingValidator()
     ok_v, err_v = validator.validate_vessels(vessels_df)
     ok_b, err_b = validator.validate_berths(berths_df)
 
     cols_vb = st.columns(2)
     with cols_vb[0]:
         st.markdown("**Barcos**")
         if ok_v:
             st.success("✓ Estructura de barcos válida")
         else:
             for e in err_v:
                 st.error(e)
     with cols_vb[1]:
         st.markdown("**Atraques**")
         if ok_b:
             st.success("✓ Estructura de atraques válida")
         else:
             for e in err_b:
                 st.error(e)
 
     st.info("Consejo: En tramos lineales (costado) incluye `type='linear'` y `group_id`.")
 
-    # Persistir en sesión
     st.session_state["vessels_df"] = vessels_df
     st.session_state["berths_df"] = berths_df
 
-# ============================
-# Tab: Optimización
-# ============================
+# ---------------------------------------------------------------------------
+# Tab 2: Optimización
+# ---------------------------------------------------------------------------
 with tabs[1]:
     st.subheader("2) Ejecutar optimización")
+
     vessels_df = st.session_state.get("vessels_df")
     berths_df = st.session_state.get("berths_df")
 
-    col_btn = st.columns([1,1,6])
+    col_btn = st.columns([1, 1, 6])
     with col_btn[0]:
         run = st.button("🚀 Optimizar", type="primary")
     with col_btn[1]:
         export_conf = st.button("⬇️ Exportar configuración JSON")
 
     if export_conf:
         conf = {
-            "policy": policy,
-            "time_limit": time_limit,
-            "prioritize_by": prioritize_by
+            "policy": st.session_state["policy"],
+            "time_limit": st.session_state["time_limit"],
+            "prioritize_by": st.session_state["prioritize_by"],
         }
-        st.download_button("Descargar JSON", data=json.dumps(conf, indent=2),
-                           file_name="config_berthing.json", mime="application/json")
+        st.download_button(
+            "Descargar JSON",
+            data=json.dumps(conf, indent=2),
+            file_name="config_berthing.json",
+            mime="application/json",
+        )
 
     if run:
-        try:
-            with st.spinner("Resolviendo…"):
-                assignments, unassigned, stats = solve_assignment(
-                    vessels_df, berths_df, policy=policy,
-                    time_limit=int(time_limit), prioritize_by=prioritize_by
-                )
-            st.session_state["assignments"] = assignments
-            st.session_state["unassigned"] = unassigned
-            st.session_state["stats"] = stats
-
-        except Exception as e:
-            st.error(f"Error durante la optimización: {e}")
+        if vessels_df is None or berths_df is None:
+            st.error("Carga primero los datos de barcos y atraques en la pestaña *Datos*.")
+        else:
+            try:
+                with st.spinner("Resolviendo…"):
+                    assignments, unassigned, stats = solve_assignment(
+                        vessels_df,
+                        berths_df,
+                        policy=st.session_state["policy"],
+                        time_limit=int(st.session_state["time_limit"]),
+                        prioritize_by=st.session_state["prioritize_by"],
+                    )
+                st.session_state["assignments"] = assignments
+                st.session_state["unassigned"] = unassigned
+                st.session_state["stats"] = stats
+            except Exception as exc:  # pragma: no cover - mostramos en UI
+                st.error(f"Error durante la optimización: {exc}")
 
-    # Mostrar resultados si existen
     assignments = st.session_state.get("assignments")
     unassigned = st.session_state.get("unassigned")
     stats = st.session_state.get("stats")
 
-    if assignments is not None:
-        k1, k2, k3, k4 = st.columns(4)
-        k1.metric("Barcos", stats.get("n_vessels", 0))
-        k2.metric("Atraques", stats.get("n_berths", 0))
-        k3.metric("Asignados", stats.get("assigned", 0))
-        k4.metric("Ocupación atraques (%)", stats.get("occupancy_pct", 0))
+    if assignments is not None and stats is not None:
+        _render_stats(stats)
 
         cols = st.columns(2)
         with cols[0]:
             st.markdown("### Asignaciones")
             st.dataframe(assignments, use_container_width=True)
             st.download_button(
                 "⬇️ Descargar asignaciones (CSV)",
                 data=assignments.to_csv(index=False).encode("utf-8"),
                 file_name="assignments.csv",
-                mime="text/csv"
+                mime="text/csv",
             )
         with cols[1]:
             st.markdown("### No asignados y motivo")
-            if unassigned.empty:
+            if unassigned is None or unassigned.empty:
                 st.info("Todos los barcos han sido asignados.")
             else:
                 st.dataframe(unassigned, use_container_width=True)
                 st.download_button(
                     "⬇️ Descargar motivos (CSV)",
                     data=unassigned.to_csv(index=False).encode("utf-8"),
                     file_name="unassigned_reasons.csv",
-                    mime="text/csv"
+                    mime="text/csv",
                 )
 
-        # Visualización de tramos lineales si existen
-        if "mode" in assignments.columns and (assignments["mode"] == "linear").any():
-            st.markdown("### Distribución en tramos lineales (costado)")
-            linear_df = assignments[assignments["mode"]=="linear"].copy()
-            # start_m y fin
-            linear
+        fig = _plot_linear_assignments(assignments, berths_df)
+        if fig is not None:
+            st.plotly_chart(fig, use_container_width=True)
+
+# ---------------------------------------------------------------------------
+# Tab 3: Compatibilidad
+# ---------------------------------------------------------------------------
+with tabs[2]:
+    st.subheader("3) Matriz de compatibilidad")
+    vessels_df = st.session_state.get("vessels_df")
+    berths_df = st.session_state.get("berths_df")
+
+    if st.button("Calcular compatibilidad"):
+        if vessels_df is None or berths_df is None:
+            st.error("Carga primero los datos en la pestaña *Datos*.")
+        else:
+            comp_df = calculate_compatibility_matrix(
+                vessels_df,
+                berths_df,
+                st.session_state["policy"],
+            )
+            st.session_state["compat_df"] = comp_df
+
+    comp_df = st.session_state.get("compat_df")
+    if comp_df is not None:
+        vessel_filter = st.multiselect(
+            "Filtrar por barcos",
+            options=comp_df["vessel_id"].unique().tolist(),
+        )
+        berth_filter = st.multiselect(
+            "Filtrar por atraques",
+            options=comp_df["berth_id"].unique().tolist(),
+        )
+
+        filtered = comp_df.copy()
+        if vessel_filter:
+            filtered = filtered[filtered["vessel_id"].isin(vessel_filter)]
+        if berth_filter:
+            filtered = filtered[filtered["berth_id"].isin(berth_filter)]
+
+        st.dataframe(filtered, use_container_width=True)
+
+        if not filtered.empty:
+            heatmap = (
+                filtered.assign(compatible_numeric=filtered["compatible"].astype(int))
+                .pivot(index="vessel_id", columns="berth_id", values="compatible_numeric")
+                .fillna(0)
+            )
+            fig = px.imshow(
+                heatmap,
+                color_continuous_scale=[[0.0, "#e74c3c"], [1.0, "#27ae60"]],
+                aspect="auto",
+                labels=dict(color="Compatible"),
+            )
+            fig.update_layout(margin=dict(l=40, r=20, t=40, b=40))
+            st.plotly_chart(fig, use_container_width=True)
+    else:
+        st.info("Pulsa *Calcular compatibilidad* para generar la matriz.")
+
+# ---------------------------------------------------------------------------
+# Tab 4: Análisis
+# ---------------------------------------------------------------------------
+with tabs[3]:
+    st.subheader("4) Análisis y KPIs")
+    stats = st.session_state.get("stats")
+    assignments = st.session_state.get("assignments")
+    unassigned = st.session_state.get("unassigned")
+
+    if stats is None:
+        st.info("Ejecuta primero la optimización para ver KPIs.")
+    else:
+        _render_stats(stats)
+        col_a, col_b = st.columns(2)
+        with col_a:
+            st.markdown("#### Utilización por atraque")
+            if assignments is not None and not assignments.empty:
+                utilization = (
+                    assignments[assignments["mode"] == "slot"]
+                    .groupby("berth_id")["utilization_%"]
+                    .mean()
+                    .reset_index()
+                )
+                if not utilization.empty:
+                    st.plotly_chart(
+                        px.bar(
+                            utilization,
+                            x="berth_id",
+                            y="utilization_%",
+                            labels={"berth_id": "Atraque", "utilization_%": "% utilización"},
+                            text="utilization_%",
+                        ).update_traces(texttemplate="%{text:.1f}%", textposition="outside"),
+                        use_container_width=True,
+                    )
+                else:
+                    st.info("No hay asignaciones en atraques discretos.")
+            else:
+                st.info("No hay resultados disponibles.")
+        with col_b:
+            st.markdown("#### Principales motivos de rechazo")
+            reasons = stats.get("rejection_reasons")
+            if reasons:
+                reasons_df = (
+                    pd.Series(reasons, name="conteo")
+                    .sort_values(ascending=False)
+                    .rename_axis("motivo")
+                    .reset_index()
+                )
+                st.plotly_chart(
+                    px.bar(
+                        reasons_df,
+                        x="motivo",
+                        y="conteo",
+                        text="conteo",
+                        color="motivo",
+                        color_discrete_sequence=px.colors.qualitative.D3,
+                    ).update_traces(textposition="outside"),
+                    use_container_width=True,
+                )
+            else:
+                st.info("Sin rechazos o sin motivos registrados.")
+
+        if unassigned is not None and not unassigned.empty:
+            st.markdown("#### Detalle de barcos no asignados")
+            st.dataframe(unassigned, use_container_width=True)
+
+# ---------------------------------------------------------------------------
+# Tab 5: Ayuda
+# ---------------------------------------------------------------------------
+with tabs[4]:
+    st.subheader("5) Ayuda y recomendaciones")
+    st.markdown(
+        """
+        **Pasos sugeridos**
+
+        1. Descarga las plantillas con `Descargar configuración JSON` para conservar parámetros.
+        2. Completa los datos de barcos y atraques respetando las columnas requeridas.
+        3. Lanza la optimización desde la pestaña **Optimización**.
+        4. Usa la pestaña **Compatibilidad** para revisar rápidamente qué combinaciones son viables.
+        5. En **Análisis** encontrarás métricas clave y motivos de rechazo para iterar sobre la
+           planificación.
+
+        ¿Necesitas exportar los resultados? Utiliza los botones de descarga disponibles en
+        cada pestaña.
+        """
+    )
 
EOF
)