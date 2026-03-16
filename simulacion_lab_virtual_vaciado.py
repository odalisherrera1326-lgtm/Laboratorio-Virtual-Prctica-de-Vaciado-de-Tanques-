import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import r2_score

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="LOU Virtual - UCV", layout="wide")

# --- 2. ENCABEZADO INSTITUCIONAL (UCV - EIQ) ---
col_l1, col_tit, col_l2 = st.columns([1, 4, 1])

with col_l1:
    if os.path.exists("logo_ucv.png"):
        st.image("logo_ucv.png", width=110)
    else:
        st.markdown("🏛️ **UCV**")

with col_tit:
    st.markdown("<h1 style='text-align: center; color: #1a5276;'>Laboratorio Virtual: Control de Tanques</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Facultad de Ingeniería - Escuela de Ingeniería Química</h3>", unsafe_allow_html=True)

with col_l2:
    if os.path.exists("logo_quimica.png"):
        st.image("logo_quimica.png", width=110)
    else:
        st.markdown("🧪 **EIQ**")

st.markdown("---")

# --- 3. DATOS EXPERIMENTALES (REFERENCIA DE TU TESIS) ---
t_v_real = np.array([0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 580])
h_v_real = np.array([35, 32, 30, 26, 23, 20.5, 18.5, 15.5, 12.5, 11, 9, 6.5, 4.5, 2, 0.5]) / 100

# --- 4. BARRA LATERAL (CONTROL TOTAL) ---
with st.sidebar:
    st.header("⚙️ Configuración del Sistema")
    tipo_tanque = st.selectbox("Geometría del Tanque", ["Cilíndrico", "Troncocónico", "Esférico"])
    diametro = st.number_input("Diámetro Base (m)", value=0.5030, format="%.4f")
    r_base = diametro / 2

    st.markdown("---")
    st.header("🌊 Perturbación (Caudal de Entrada)")
    st.info("Aquí defines el cambio brusco en el caudal que el PID debe compensar.")
    q_inicial = st.number_input("Caudal Inicial (m³/s)", value=0.000070, format="%.7f")
    t_cambio = st.slider("Tiempo de la Perturbación (s)", 0, 1000, 300)
    q_final = st.number_input("Caudal tras Perturbación (m³/s)", value=0.000120, format="%.7f")

    st.markdown("---")
    st.header("🎮 Parámetros PID")
    sp = st.slider("Setpoint de Nivel (m)", 0.0, 0.5, 0.25)
    kp = st.number_input("Ganancia Proporcional (Kp)", value=1.5)
    ti = st.number_input("Tiempo Integral (Ti)", value=15.0)
    td = st.number_input("Tiempo Derivativo (Td)", value=1.2)
    u_bias = st.number_input("Sesgo (u0)", value=0.00037, format="%.6f")

# --- 5. FUNCIONES DE CÁLCULO ÁREA ---
def obtener_area(h):
    if tipo_tanque == "Cilíndrico":
        return np.pi * r_base**2
    elif tipo_tanque == "Troncocónico":
        radio_h = r_base + (0.45 - r_base) * (h / 1.0)
        return np.pi * radio_h**2
    else: # Esférico
        return np.pi * (2 * r_base * h - h**2) if h > 0 else 1e-5

# --- 6. PROCESAMIENTO Y SIMULACIÓN ---
if st.button("🚀 Iniciar Simulación de Control"):
    t_sim = np.arange(0, 1001, 1.0)
    h_actual = 0.20 # Partimos de un nivel inicial estable
    e_sum, e_prev = 0, 0
    
    h_hist, u_hist, q_hist = [], [], []

    # Bucle de integración numérica (Euler)
    for t in t_sim:
        # Definición de la perturbación en el tiempo
        q_in = q_inicial if t < t_cambio else q_final
        q_hist.append(q_in)

        # Algoritmo PID
        error = sp - h_actual
        e_sum += error
        u_calc = u_bias + (kp * (error + (1/ti)*e_sum + td*(error - e_prev)))
        u = np.clip(u_calc, 0, 0.01) # Saturación de la válvula
        
        # Dinámica del proceso
        area_actual = obtener_area(h_actual)
        q_out = u * np.sqrt(2 * 9.81 * h_actual) if h_actual > 0 else 0
        h_actual += ((q_in - q_out) / area_actual)
        h_actual = max(0, h_actual) # Límite físico
        
        h_hist.append(h_actual)
        u_hist.append(u)
        e_prev = error

    # --- 7. DESPLIEGUE DE RESULTADOS ---
    col_main, col_metrics = st.columns([3, 1])

    with col_main:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
        
        # Gráfico 1: Nivel vs Setpoint
        ax1.plot(t_sim, h_hist, label="Nivel Medido (PV)", color="#1f77b4", lw=2.5)
        ax1.axhline(sp, color="orange", ls="--", label=f"Setpoint (SP) = {sp}m", lw=2)
        ax1.set_ylabel("Altura (m)")
        ax1.set_title(f"Respuesta del Tanque {tipo_tanque} ante Perturbación", fontsize=14)
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)

        # Gráfico 2: La Perturbación (Qin)
        ax2.step(t_sim, q_hist, color="#d62728", label="Caudal de Entrada (Perturbación)")
        ax2.set_ylabel("Qin (m³/s)")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

        # Gráfico 3: Acción de Control
        ax3.step(t_sim, u_hist, color="#2ca02c", label="Apertura de Válvula (u)")
        ax3.set_ylabel("u (Apertura)")
        ax3.set_xlabel("Tiempo (s)")
        ax3.legend(loc="upper right")
        ax3.grid(True, alpha=0.3)
        
        st.pyplot(fig)

    with col_metrics:
        st.subheader("📊 Análisis de Datos")
        
        # Cálculo de métricas de desempeño
        h_interp = np.interp(t_v_real, t_sim, h_hist)
        precision_r2 = r2_score(h_v_real, h_interp)
        iae_error = np.sum(np.abs(sp - np.array(h_hist)))

        st.metric("Ajuste R² (vs Real)", f"{precision_r2:.4f}")
        st.metric("Error IAE", f"{iae_error:.2f}")
        
        st.markdown("---")
        st.subheader("📂 Reporte Final")
        
        # Preparación de descarga
        data_final = pd.DataFrame({
            'Tiempo_s': t_sim, 
            'Altura_m': h_hist, 
            'Qin_m3s': q_hist,
            'Apertura_u': u_hist
        })
        
        csv_buffer = data_final.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Resultados (CSV)",
            data=csv_buffer,
            file_name=f"reporte_control_{tipo_tanque.lower()}.csv",
            mime="text/csv"
        )
        
        st.success("Simulación completada con éxito.")
