import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

# 1. Configuración de la interfaz profesional
st.set_page_config(page_title="Simulador LOU - UCV", layout="wide")

st.title("🎛️ Simulador Virtual: Vaciado y Llenado de Tanques")
st.markdown("### Laboratorio de Operaciones Unitarias - Escuela de Ingeniería Química - UCV")
st.write("Esta herramienta permite simular la dinámica de niveles basada en modelos matemáticos validados.")

# 2. Entradas en la barra lateral
st.sidebar.header("Panel de Control")

modo = st.sidebar.selectbox("Seleccione el Proceso", ["Vaciado de Tanque", "Llenado de Tanque"])

# Parámetros físicos del tanque real del LOU
diametro = st.sidebar.number_input("Diámetro del Tanque (m)", value=0.503, format="%.3f")
area_tanque = np.pi * (diametro / 2)**2

if modo == "Vaciado de Tanque":
    st.sidebar.subheader("Configuración de Vaciado")
    h0 = st.sidebar.slider("Nivel Inicial (m)", 0.0, 0.45, 0.35)
    k_vaciado = st.sidebar.number_input("Constante de Salida (K)", value=0.000295, format="%.6f")
    t_sim = st.sidebar.slider("Tiempo de Simulación (s)", 100, 2000, 1000)
else:
    st.sidebar.subheader("Configuración de Llenado")
    h0 = 0.0
    # Tu ecuación de calibración: y = 39.739x + 106.9
    lectura_rotametro = st.sidebar.number_input("Lectura del Rotámetro (cm)", value=4.0, step=0.5)
    q_lh = 39.739 * lectura_rotametro + 106.9
    q_alim = (q_lh / 1000) / 3600  # Conversión L/h a m3/s
    t_sim = st.sidebar.slider("Tiempo de Simulación (s)", 100, 2000, 1000)
    st.sidebar.info(f"Caudal calculado: {q_lh:.2f} L/h")

# 3. Ejecución de la Simulación
if st.button("🚀 Ejecutar Simulación"):
    t_eval = np.linspace(0, t_sim, 100)

    if modo == "Vaciado de Tanque":
        # dh/dt = -K*sqrt(h) / A
        def modelo(t, h):
            return -k_vaciado * np.sqrt(max(0, h)) / area_tanque
        sol = solve_ivp(modelo, [0, t_sim], [h0], t_eval=t_eval)
        h_res = sol.y[0]
    else:
        # h(t) = h0 + (Qin/A)*t
        h_res = h0 + (q_alim / area_tanque) * t_eval

    # Crear DataFrame para resultados
    df_resultados = pd.DataFrame({
        "Tiempo (s)": t_eval,
        "Nivel Simulado (m)": h_res
    })

    # 4. Visualización
    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t_eval, h_res, color='#007bff', linewidth=2, label=f"Modelo de {modo}")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Nivel (m)")
        ax.set_title(f"Perfil de Nivel vs Tiempo - {modo}")
        ax.grid(True, linestyle='--')
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("Resultados Numéricos")
        st.metric("Nivel Final", f"{h_res[-1]:.3f} m")
        
        # --- BOTÓN DE DESCARGA ---
        csv = df_resultados.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar datos (CSV)",
            data=csv,
            file_name=f'resultados_{modo.lower().replace(" ", "_")}.csv',
            mime='text/csv',
        )
        st.dataframe(df_resultados, height=400)

st.markdown("---")
st.caption("Proyecto de Grado - Ingeniería Química UCV | 2026")
