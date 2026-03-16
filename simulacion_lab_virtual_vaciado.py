import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. CONFIGURACIÓN E IDENTIDAD UCV ---
st.set_page_config(page_title="Práctica de Vaciado de Tanques - UCV", layout="wide")

col_l1, col_tit, col_l2 = st.columns([1, 4, 1])
def cargar_logo(archivo, nombre):
    if os.path.exists(archivo): st.image(archivo, width=100)
    else: st.markdown(f"<div style='border:1px solid #ccc;padding:10px'>{nombre}</div>", unsafe_allow_html=True)

with col_l1: cargar_logo("logo_ucv.png", "UCV")
with col_tit:
    st.markdown("<h1 style='text-align: center;'>Práctica de Vaciado de Tanques</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Unidad de Operaciones Unitarias - Facultad de Ingeniería</p>", unsafe_allow_html=True)
with col_l2: cargar_logo("logo_quimica.png", "EIQ")

# --- 2. MARCO TEÓRICO ---
with st.expander("📖 Marco Teórico y Modelado Matemático", expanded=False):
    st.markdown(r"""
    El sistema se modela mediante el balance de masa dinámico para geometrías variables:
    $$A(h) \frac{dh}{dt} = Q_{in}(u) - Q_{out}(h) \pm Q_{p}$$
    
    Donde la acción de control $u$ (salida del PID) regula el caudal de entrada $Q_{in}$ a través de la ecuación del rotámetro calibrado experimentalmente.
    """)

# --- 3. PARÁMETROS (SIDEBAR) ---
st.sidebar.header("⚙️ Parámetros de Simulación")
tipo_p = st.sidebar.selectbox("Proceso", ["Llenado", "Vaciado"])
geom = st.sidebar.selectbox("Geometría", ["Cilíndrico", "Cónico", "Esférico"])
R = st.sidebar.number_input("Radio (R) [m]", value=1.5)
H = st.sidebar.number_input("Altura (H) [m]", value=4.0)
sp = st.sidebar.slider("Setpoint [m]", 0.1, H, H/2)

st.sidebar.divider()
st.sidebar.subheader("🌪️ Perturbación ($Q_p$)")
hay_p = st.sidebar.toggle("Activar Falla/Carga")
mag_p = st.sidebar.number_input("Caudal $Q_p$ [m³/s]", value=0.02, format="%.3f") if hay_p else 0
t_p = st.sidebar.slider("Tiempo de inicio (s)", 0, 400, 100) if hay_p else 0

st.sidebar.divider()
st.sidebar.subheader("🎮 Sintonización PID")
kp, ki, kd = st.sidebar.number_input("Kp", value=2.0), st.sidebar.number_input("Ki", value=0.5), st.sidebar.number_input("Kd", value=0.1)
t_max = st.sidebar.slider("Tiempo Total (s)", 60, 600, 300)

# --- 4. LÓGICA DE SIMULACIÓN ---
def calcular():
    dt = 0.5
    t = np.arange(0, t_max, dt)
    h, u_out, error_log = np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t))
    h[0] = H if tipo_p == "Vaciado" else 0.05
    e_i, e_prev = 0, 0
    
    for i in range(1, len(t)):
        # 1. Geometría variable A(h)
        if geom == "Cilíndrico": Ah = np.pi * R**2
        elif geom == "Cónico": Ah = np.pi * ((R/H) * h[i-1])**2
        else: Ah = np.pi * (2*R*h[i-1] - h[i-1]**2)
        Ah = max(Ah, 0.01)

        # 2. PID y Caudales
        err = sp - h[i-1]
        e_i += err * dt
        e_d = (err - e_prev) / dt
        u = (kp * err) + (ki * e_i) + (kd * e_d)
        
        qin = np.clip(u, 0, 1.0) # Capacidad de bomba/válvula
        qout = 0.6 * 0.02 * np.sqrt(2 * 9.81 * h[i-1]) if h[i-1] > 0 else 0
        qp_act = mag_p if (hay_p and t[i] >= t_p) else 0
        
        # 3. Integración
        h[i] = np.clip(h[i-1] + ((qin - qout + qp_act) / Ah) * dt, 0, H)
        u_out[i], error_log[i], e_prev = qin, err, err
        
    return t, h, u_out, error_log

tiempo, nivel, esfuerzo, errores = calcular()

# --- 5. VISUALIZACIÓN MULTI-GRÁFICO ---
st.subheader("📊 Análisis Gráfico del Sistema")
col_g1, col_g2 = st.columns(2)

with col_g1:
    # Gráfico 1: Nivel (PV) vs Setpoint (SP)
    fig1, ax1 = plt.subplots()
    ax1.plot(tiempo, nivel, color='#007bff', lw=2, label='Nivel (PV)')
    ax1.axhline(y=sp, color='red', ls='--', label='Setpoint')
    ax1.set_title("Control de Nivel")
    ax1.set_ylabel("Metros (m)")
    ax1.legend()
    st.pyplot(fig1)

with col_g2:
    # Gráfico 2: Esfuerzo del Controlador (Acción de la Válvula)
    fig2, ax2 = plt.subplots()
    ax2.step(tiempo, esfuerzo, color='#ff7f0e', label='Apertura Válvula (u)')
    ax2.set_title("Esfuerzo del Controlador (Qin)")
    ax2.set_ylabel("Caudal (m³/s)")
    ax2.fill_between(tiempo, esfuerzo, alpha=0.2, color='#ff7f0e')
    st.pyplot(fig2)

# Gráfico 3: Evolución del Error (en el cuerpo central)
st.markdown("### 📉 Evolución del Error ($e = SP - PV$)")
fig3, ax3 = plt.subplots(figsize=(12, 3))
ax3.plot(tiempo, errores, color='#d62728', label='Error')
ax3.axhline(y=0, color='black', lw=1)
ax3.set_title("Desviación del Setpoint")
ax3.fill_between(tiempo, errores, color='#d62728', alpha=0.1)
st.pyplot(fig3)

# --- 6. MÉTRICAS DE DESEMPEÑO (ESTILO VIDEO) ---
st.divider()
st.header("📈 Evaluación de Desempeño")
mse = mean_squared_error(np.full(len(nivel), sp), nivel)
r2 = r2_score(np.full(len(nivel), sp), nivel)
err_r = sp - nivel[-1]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Nivel Alcanzado", f"{nivel[-1]:.3f} m")
c2.metric("Error Final", f"{err_r:.4f} m", delta=f"{err_r:.4f}", delta_color="inverse")
c3.metric("Precisión (R²)", f"{r2:.4f}")
c4.metric("MSE", f"{mse:.6f}")

st.table(pd.DataFrame({
    "Parámetro": ["Geometría Seleccionada", "Magnitud de Perturbación ($Q_p$)", "Tiempo de Estabilización Sugerido"],
    "Valor": [geom, f"{mag_p} m³/s", f"{tiempo[np.where(abs(errores) < 0.01)[0][0]] if any(abs(errores) < 0.01) else 'N/A'} s"]
}))

st.download_button("📥 Exportar Datos Experimentales", pd.DataFrame({"t":tiempo,"h":nivel,"u":esfuerzo}).to_csv(), "datos_ucv.csv")
