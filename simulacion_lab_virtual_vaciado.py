import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. CONFIGURACIÓN E IDENTIDAD VISUAL ---
st.set_page_config(page_title="Práctica de Vaciado de Tanques - UCV", layout="wide")

col_l1, col_tit, col_l2 = st.columns([1, 4, 1])

def cargar_logo(nombre_archivo, alias):
    if os.path.exists(nombre_archivo):
        st.image(nombre_archivo, width=100)
    else:
        st.markdown(f"<div style='text-align:center; border: 1px solid #ccc; padding: 20px;'>Logo {alias}</div>", unsafe_allow_html=True)

with col_l1:
    cargar_logo("logo_ucv.png", "UCV")
with col_tit:
    st.markdown("<h1 style='text-align: center;'>Práctica de Vaciado de Tanques</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Escuela de Ingeniería Química - Universidad Central de Venezuela</p>", unsafe_allow_html=True)
with col_l2:
    cargar_logo("logo_quimica.png", "EIQ")

st.markdown("---")

# --- 2. MARCO TEÓRICO INTEGRADO ---
with st.expander("📖 Marco Teórico: Modelado de Geometría Variable", expanded=True):
    st.markdown(r"""
    El balance de masa para un tanque de sección transversal variable es:
    
    $$A(h) \frac{dh}{dt} = Q_{in} - Q_{out} \pm Q_{p}$$
    
    Donde el área superficial $A(h)$ depende de la geometría:
    * **Cilíndrico:** $A(h) = \pi R^2$ (Constante)
    * **Cónico:** $A(h) = \pi \left( \frac{R}{H} h \right)^2$
    * **Esférico:** $A(h) = \pi (2Rh - h^2)$
    
    El caudal de salida se rige por la Ley de Torricelli experimental: $Q_{out} = C_d \cdot a \cdot \sqrt{2gh}$.
    """)

# --- 3. BARRA LATERAL: PARÁMETROS EXPERIMENTALES ---
st.sidebar.header("⚙️ Configuración del Sistema")

# Selección de Proceso y Geometría
tipo_proceso = st.sidebar.selectbox("Tipo de Proceso", ["Llenado", "Vaciado"])
geometria = st.sidebar.selectbox("Geometría del Tanque", ["Cilíndrico", "Cónico", "Esférico"])
radio_max = st.sidebar.number_input("Radio Máximo (R) [m]", value=1.0)
altura_total = st.sidebar.number_input("Altura Total (H) [m]", value=3.0)

# Parámetros de Control
setpoint = st.sidebar.slider("Setpoint (Nivel Deseado) [m]", 0.1, altura_total, altura_total/2)

# Perturbación (Caudal imprevisto)
st.sidebar.divider()
st.sidebar.subheader("🌪️ Perturbación ($Q_p$)")
hay_perturbacion = st.sidebar.toggle("Activar Perturbación")
mag_pert = st.sidebar.number_input("Magnitud ($Q_p$) [m³/s]", value=0.005, format="%.4f") if hay_perturbacion else 0.0
t_pert = st.sidebar.slider("Instante de perturbación (s)", 0, 500, 100) if hay_perturbacion else 0

# Sintonización PID
st.sidebar.divider()
st.sidebar.subheader("🎮 Parámetros PID")
kp = st.sidebar.number_input("Kp", value=2.5)
ki = st.sidebar.number_input("Ki", value=0.5)
kd = st.sidebar.number_input("Kd", value=0.1)

t_sim = st.sidebar.slider("Tiempo de Simulación (s)", 60, 600, 300)

# --- 4. LÓGICA DE SIMULACIÓN (EULER) ---
def simular():
    dt = 0.5
    tiempo = np.arange(0, t_sim, dt)
    h = np.zeros(len(tiempo))
    h[0] = altura_total if tipo_proceso == "Vaciado" else 0.01
    
    err_acum = 0
    err_prev = 0
    
    for i in range(1, len(tiempo)):
        # Calcular Área Transversal según Geometría
        if geometria == "Cilíndrico":
            A_h = np.pi * (radio_max**2)
        elif geometria == "Cónico":
            A_h = np.pi * ( (radio_max / altura_total) * h[i-1] )**2
        else: # Esférico
            A_h = np.pi * (2 * radio_max * h[i-1] - h[i-1]**2)
        
        # Evitar división por cero en geometrías que se cierran en la base
        A_h = max(A_h, 0.01)

        # Controlador PID
        error = setpoint - h[i-1]
        err_acum += error * dt
        der = (error - err_prev) / dt
        u = (kp * error) + (ki * err_acum) + (kd * der)
        
        q_in = np.clip(u, 0, 0.5) # Caudal entrada máximo
        q_out = 0.6 * 0.05 * np.sqrt(2 * 9.81 * h[i-1]) if h[i-1] > 0 else 0
        q_p = mag_pert if (hay_perturbacion and tiempo[i] >= t_pert) else 0
        
        # Balance: dh/dt = (Qin - Qout + Qp) / A(h)
        dh_dt = (q_in - q_out + q_p) / A_h
        h[i] = np.clip(h[i-1] + dh_dt * dt, 0, altura_total)
        err_prev = error
        
    return tiempo, h

t, nivel = simular()
df_res = pd.DataFrame({"Tiempo (s)": t, "Nivel (m)": nivel})

# --- 5. RESULTADOS Y ANÁLISIS ---
col_g, col_d = st.columns([2, 1])

with col_g:
    st.subheader(f"Respuesta del Sistema: {geometria}")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, nivel, color='#00a65a', lw=2, label='Nivel Real (PV)')
    ax.axhline(y=setpoint, color='red', ls='--', label=f'Setpoint: {setpoint}m')
    if hay_perturbacion:
        ax.axvline(x=t_pert, color='orange', ls=':', label='Perturbación')
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Altura (m)")
    ax.legend()
    st.pyplot(fig)

with col_d:
    st.subheader("Análisis de Desempeño")
    ref = np.full(len(nivel), setpoint)
    mse = mean_squared_error(ref, nivel)
    r2 = r2_score(ref, nivel)
    n_fin = nivel[-1]
    
    st.metric("Nivel Final", f"{n_fin:.3f} m")
    st.metric("Error Residual", f"{setpoint - n_fin:.4f} m")
    st.metric("Precisión (R²)", f"{r2:.4f}")
    
    st.write("**Resumen Técnico**")
    st.table(pd.DataFrame({
        "Métrica": ["MSE", "Geometría", "Perturbación"],
        "Valor": [f"{mse:.6f}", geometria, f"{mag_pert} m³/s" if hay_perturbacion else "No"]
    }))

st.download_button("📥 Descargar Datos (CSV)", df_res.to_csv(index=False), "practica_ucv.csv")
