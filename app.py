# app.py — Interfaz Streamlit mínima y estable para probar el solver
import io
import streamlit as st
import pandas as pd
from solver import (
    solve_assignment,
    calculate_compatibility_matrix,
    BerthingValidator,
    DEFAULT_POLICY
)

st.set_page_config(page_title="Optimización de Atraques", layout="wide")
st.title("🛥️ Optimización de ubicación de embarcaciones")

st.markdown("Sube tus ficheros **CSV** de Barcos y Atraques o usa el ejemplo. Pulsa **Optimizar**.")

# Datos demo por si no subes ficheros
DEMO_VESSELS = """vessel_id,loa,beam,draft,type,power_kw
B1,11.0,3.6,1.8,mono,6
B2,14.0,4.2,2.0,mono,8
B3,12.5,7.0,1.3,cata,10
B4,9.5,3.2,1.5,mono,4
"""
DEMO_BERTHS = """berth_id,length,slip_width,depth,fairway_width,power_kw,type,group_id
A1,12.0,4.0,2.5,35.0,8,finger,
A2,14.0,4.5,2.7,35.0,12,finger,
A3,10.5,3.6,1.6,35.0,6,finger,
L1,30.0,,3.0,60.0,,linear,LINEAL-1
"""

# Carga de ficheros
col1, col2 = st.columns(2)
with col1:
    v_file = st.file_uploader("Barcos (CSV)", type=["csv"])
with col2:
    b_file = st.file_uploader("Atraques (CSV)", type=["csv"])

def read_csv_or_demo(file, demo_text):
    if file is not None:
        return pd.read_csv(file)
    return pd.read_csv(io.StringIO(demo_text))

vessels_df = read_csv_or_demo(v_file, DEMO_VESSELS)
berths_df  = read_csv_or_demo(b_file, DEMO_BERTHS)

# Parámetros básicos (barra lateral)
with st.sidebar:
    st.header("Parámetros")
    alpha = st.number_input("α (calle)", value=float(DEFAULT_POLICY["alpha"]), step=0.1)
    beta  = st.number_input("β (calle)", value=float(DEFAULT_POLICY["beta"]), step=0.1)
    ukc   = st.number_input("UKC (m)", value=float(DEFAULT_POLICY["ukc"]), step=0.1)
    tide  = st.number_input("Margen marea (m)", value=float(DEFAULT_POLICY["tide_safety"]), step=0.1)
    f_m   = st.number_input("Defensa/lado mono (m)", value=float(DEFAULT_POLICY["fender_mono"]), step=0.05)
    f_c   = st.number_input("Defensa/lado cata (m)", value=float(DEFAULT_POLICY["fender_cata"]), step=0.05)
    endm  = st.number_input("Margen proa+popa (m)", value=float(DEFAULT_POLICY["end_margin"]), step=0.1)
    gap   = st.number_input("Separación costado (m)", value=float(DEFAULT_POLICY["min_gap_linear"]), step=0.1)
    tlim  = st.slider("Tiempo solver (s)", 5, 60, 20, 5)

policy = {
    "alpha": alpha, "beta": beta, "ukc": ukc, "tide_safety": tide,
    "fender_mono": f_m, "fender_cata": f_c, "end_margin": endm,
    "min_gap_linear": gap
}

# Vista previa
st.subheader("Datos de entrada")
st.write("**Barcos**")
st.dataframe(vessels_df, use_container_width=True)
st.write("**Atraques**")
st.dataframe(berths_df, use_container_width=True)

# Validación
st.subheader("Validación")
validator = BerthingValidator()
ok_v, err_v = validator.validate_vessels(vessels_df)
ok_b, err_b = validator.validate_berths(berths_df)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Barcos**")
    if ok_v: st.success("✓ Válidos")
    else:
        for e in err_v: st.error(e)
with c2:
    st.markdown("**Atraques**")
    if ok_b: st.success("✓ Válidos")
    else:
        for e in err_b: st.error(e)

# Botón de optimizar
st.subheader("Optimización")
btn = st.button("🚀 Optimizar asignación", type="primary", disabled=not(ok_v and ok_b))

if btn:
    try:
        assign, unassigned, stats = solve_assignment(vessels_df, berths_df, policy=policy, time_limit=int(tlim))
        st.success(f"Asignados: {stats['assigned']}/{stats['n_vessels']} | Ocupación: {stats['occupancy_pct']}%")

        st.markdown("### Asignaciones")
        st.dataframe(assign, use_container_width=True)
        st.download_button("⬇️ Descargar asignaciones CSV", assign.to_csv(index=False).encode("utf-8"),
                           file_name="assignments.csv", mime="text/csv")

        st.markdown("### No asignados y motivo")
        st.dataframe(unassigned, use_container_width=True)
        st.download_button("⬇️ Descargar motivos CSV", unassigned.to_csv(index=False).encode("utf-8"),
                           file_name="unassigned_reasons.csv", mime="text/csv")

        st.markdown("### Compatibilidad (resumen)")
        comp = calculate_compatibility_matrix(vessels_df, berths_df, policy)
        st.dataframe(comp.head(100), use_container_width=True)

    except Exception as e:
        st.error(f"Error durante la optimización: {e}")
