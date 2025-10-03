# app.py — UI centrada en eficiencia de ubicación de embarcaciones
# - Soporta slots discretos (finger/thead) y tramos lineales (costado)
# - KPIs, validación, compatibilidad, descarga de resultados
# - Visualización de tramos lineales (orden/posición) con Plotly

from __future__ import annotations
import io
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from solver import (
    solve_assignment,
    calculate_compatibility_matrix,
    BerthingValidator,
    DEFAULT_POLICY
)

st.set_page_config(page_title="Optimización de Atraques", layout="wide")
st.title("🛥️ Optimización de ubicación de embarcaciones")

# ============================
# Sidebar: Parámetros y ayuda
# ============================
with st.sidebar:
    st.header("Parámetros de maniobra/seguridad")
    alpha = st.number_input("α (calle)", value=float(DEFAULT_POLICY["alpha"]), step=0.1,
                            help="W_calle ≥ α · LOA + β")
    beta = st.number_input("β (calle)", value=float(DEFAULT_POLICY["beta"]), step=0.1)
    ukc = st.number_input("UKC (margen bajo quilla, m)", value=float(DEFAULT_POLICY["ukc"]), step=0.1)
    tide_safety = st.number_input("Margen marea (m)", value=float(DEFAULT_POLICY.get("tide_safety",0.1)), step=0.1)
    f_mono = st.number_input("Defensa/lado monocasco (m)", value=float(DEFAULT_POLICY["fender_mono"]), step=0.05)
    f_cata = st.number_input("Defensa/lado catamarán (m)", value=float(DEFAULT_POLICY["fender_cata"]), step=0.05)
    end_m = st.number_input("Margen proa+popa en finger (m)", value=float(DEFAULT_POLICY["end_margin"]), step=0.1)
    min_gap_linear = st.number_input("Separación en costado (m)", value=float(DEFAULT_POLICY.get("min_gap_linear",0.4)), step=0.1,
                                     help="Hueco mínimo entre barcos en tramos lineales (costado).")

    st.divider()
    st.subheader("Ejecución")
    time_limit = st.slider("Tiempo límite del solver (s)", 5, 120, 20, 5)
    prioritize_by = st.selectbox(
        "Priorizar barcos por… (opcional)",
        options=["(ninguno)", "loa", "draft", "power_kw"],
        index=0,
        help="Ordena barcos para orientar la búsqueda (no impone prioridad dura)."
    )
    if prioritize_by == "(ninguno)":
        prioritize_by = None

policy = {
    "alpha": alpha,
    "beta": beta,
    "ukc": ukc,
    "tide_safety": tide_safety,
    "fender_mono": f_mono,
    "fender_cata": f_cata,
    "end_margin": end_m,
    "min_gap_linear": min_gap_linear
}

# ============================
# Datos de ejemplo (fallback)
# ============================
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

# ============================
# Tabs principales
# ============================
tabs = st.tabs(["Datos", "Optimización", "Compatibilidad", "Análisis", "Ayuda"])

# ============================
# Tab: Datos
# ============================
with tabs[0]:
    st.subheader("1) Carga tus datos de Barcos y Atraques")

    col1, col2 = st.columns(2)
    with col1:
        f_vess = st.file_uploader("Barcos (CSV o Excel)", type=["csv","xls","xlsx"], key="vess")
    with col2:
        f_berths = st.file_uploader("Atraques (CSV o Excel)", type=["csv","xls","xlsx"], key="berths")

    def read_any(file):
        if file is None:
            return None
        suffix = file.name.lower().split(".")[-1]
        if suffix in ("xls","xlsx"):
            return pd.read_excel(file)
        return pd.read_csv(file)

    if f_vess is not None:
        vessels_df = read_any(f_vess)
    else:
        vessels_df = pd.read_csv(io.StringIO(DEMO_VESSELS))

    if f_berths is not None:
        berths_df = read_any(f_berths)
    else:
        berths_df = pd.read_csv(io.StringIO(DEMO_BERTHS))

    # Vista previa
    st.write("**Barcos** (primeras filas)")
    st.dataframe(vessels_df.head(20), use_container_width=True)
    st.write("**Atraques** (primeras filas)")
    st.dataframe(berths_df.head(20), use_container_width=True)

    # Validación
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

    # Persistir en sesión
    st.session_state["vessels_df"] = vessels_df
    st.session_state["berths_df"] = berths_df

# ============================
# Tab: Optimización
# ============================
with tabs[1]:
    st.subheader("2) Ejecutar optimización")
    vessels_df = st.session_state.get("vessels_df")
    berths_df = st.session_state.get("berths_df")

    col_btn = st.columns([1,1,6])
    with col_btn[0]:
        run = st.button("🚀 Optimizar", type="primary")
    with col_btn[1]:
        export_conf = st.button("⬇️ Exportar configuración JSON")

    if export_conf:
        conf = {
            "policy": policy,
            "time_limit": time_limit,
            "prioritize_by": prioritize_by
        }
        st.download_button("Descargar JSON", data=json.dumps(conf, indent=2),
                           file_name="config_berthing.json", mime="application/json")

    if run:
        try:
            with st.spinner("Resolviendo…"):
                assignments, unassigned, stats = solve_assignment(
                    vessels_df, berths_df, policy=policy,
                    time_limit=int(time_limit), prioritize_by=prioritize_by
                )
            st.session_state["assignments"] = assignments
            st.session_state["unassigned"] = unassigned
            st.session_state["stats"] = stats

        except Exception as e:
            st.error(f"Error durante la optimización: {e}")

    # Mostrar resultados si existen
    assignments = st.session_state.get("assignments")
    unassigned = st.session_state.get("unassigned")
    stats = st.session_state.get("stats")

    if assignments is not None:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Barcos", stats.get("n_vessels", 0))
        k2.metric("Atraques", stats.get("n_berths", 0))
        k3.metric("Asignados", stats.get("assigned", 0))
        k4.metric("Ocupación atraques (%)", stats.get("occupancy_pct", 0))

        cols = st.columns(2)
        with cols[0]:
            st.markdown("### Asignaciones")
            st.dataframe(assignments, use_container_width=True)
            st.download_button(
                "⬇️ Descargar asignaciones (CSV)",
                data=assignments.to_csv(index=False).encode("utf-8"),
                file_name="assignments.csv",
                mime="text/csv"
            )
        with cols[1]:
            st.markdown("### No asignados y motivo")
            if unassigned.empty:
                st.info("Todos los barcos han sido asignados.")
            else:
                st.dataframe(unassigned, use_container_width=True)
                st.download_button(
                    "⬇️ Descargar motivos (CSV)",
                    data=unassigned.to_csv(index=False).encode("utf-8"),
                    file_name="unassigned_reasons.csv",
                    mime="text/csv"
                )

        # Visualización de tramos lineales si existen
        if "mode" in assignments.columns and (assignments["mode"] == "linear").any():
            st.markdown("### Distribución en tramos lineales (costado)")
            linear_df = assignments[assignments["mode"]=="linear"].copy()
            # start_m y fin
            linear
