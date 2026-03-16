import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import os

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="LOU Virtual - UCV", layout="wide")

# --- ENCABEZADO INSTITUCIONAL ---
# Creamos 3 columnas para los logos y el título central
col_logo1, col_titulo, col_logo2 = st.columns([1, 4, 1])

with col_logo1:
    # Intenta cargar el logo de la UCV
    if os.path.exists("logo_ucv.png"):
        st.image("logo_ucv.png", width=100)
    else:
        st.write("🏛️ **UCV**")

with col_titulo:
    st.markdown("<h1 style='text-align: center;'>Laboratorio Virtual: Vaciado y Llenado de Tanques</h1>", unsafe_allow_second_party_html=True)
    st.markdown("<h3 style='text-align: center;'>Facultad de Ingeniería - Escuela de Ingeniería Química</h3>", unsafe_allow_second_party_html=True)

with col_logo2:
    # Intenta cargar el logo de la Escuela de Química o Ingeniería
    if os.path.exists("logo_quimica.png"):
        st.image("logo_quimica.png", width=100)
    else:
        st.write("🧪 **EIQ**")

st.markdown("---")

# --- EL RESTO DEL CÓDIGO (DATOS Y LÓGICA) ---

# [Aquí va el resto del código que ya teníamos: Datos experimentales, Sidebar, Simulación...]

# --- DATOS EXPERIMENTALES (MATLAB) ---
t_vaciado = np.array([0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 580])
h_vaciado = np.array([35, 32, 30, 26, 23, 20.5, 18.5, 15.5, 12.5, 11, 9, 6.5, 4.5, 2, 0.5]) / 100

t_llenado = np.array([0, 76, 152, 228, 304, 380, 456, 532, 608, 684, 760, 836, 912, 988])
h_llenado = np.array([0, 4, 7, 9.5, 13, 15, 17, 20.5, 23, 25, 28.2, 30.2, 33.5, 35]) / 100

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("⚙️ Configuración")
    modo = st.radio("Operación", ["Vaciado", "Llenado"])
    tipo_tanque = st.selectbox("Geometría", ["Cilíndrico", "Troncocónico", "Esférico"])
    diametro = st.number_input("Diámetro Base (m)", value=0.5030, format="%.4f")
    r_base = diametro / 2

    st.markdown("---")
    st.header("🎮 Parámetros PID")
    sp = st.slider("Setpoint (m)", 0.0, 0.5, 0.25)
    kp = st.number_input("Ganancia (Kp)", value=1.0)
    ti = st.number_input("Tiempo Integral (Ti)", value=20.0)
    td = st.number_input("Tiempo Derivativo (Td)", value=1.0)
    u_bias = st.number_input("Sesgo / K (u0)", value=0.00037, format="%.6f")

    if modo == "Llenado":
        st.header("🌊 Rotámetro")
        plomada = st.slider("Plomada (cm)", 0.0, 10.0, 4.0)
        q_alim = ( (39.739 * plomada + 106.9) / 1000 ) / 3600
    else:
        q_alim = 0.0

# --- BASE TEÓRICA ---
with st.expander("📚 Fundamentos: Geometría y Balance de Masa", expanded=False):
    st.latex(r"\frac{dh}{dt} = \frac{Q_{in} - Q_{out}}{A(h)}")
    st.info("Esta herramienta permite validar el modelo matemático frente a datos experimentales reales obtenidos en el laboratorio de la EIQ-UCV.")

# --- LÓGICA DE ÁREA ---
def obtener_area(h):
    if tipo_tanque == "Cilíndrico":
        return np.pi * r_base**2
    elif tipo_tanque == "Troncocónico":
        r_tope, H_total = 0.45, 1.0
        radio_h = r_base + (r_tope - r_base) * (h / H_total)
        return np.pi * radio_h**2
    else: # Esférico
        return np.pi * (2 * r_base * h - h**2) if h > 0 else 1e-5

# --- EJECUCIÓN ---
if st.button("🚀 Iniciar Simulación"):
    t_exp = t_llenado if modo == "Llenado" else t_vaciado
    h_exp = h_llenado if modo == "Llenado" else h_vaciado
    dt = 1.0
    t_full = np.arange(0, t_exp[-1] + 1, dt)
    h_actual, err_acum, e_prev = h_exp[0], 0, 0
    h_full, hist_u = [], []

    for t in t_full:
        error = sp - h_actual
        err_acum += error * dt
        u = u_bias + (kp * (error + (1/ti) * err_acum + td * (error - e_prev) / dt))
        u = np.clip(u, 0, 0.01) 
        area_v = obtener_area(h_actual)
        q_out = u * np.sqrt(2 * 9.81 * h_actual) if h_actual > 0 else 0
        h_actual += ((q_alim - q_out) / area_v) * dt
        h_actual = max(0, h_actual)
        h_full.append(h_actual)
        hist_u.append(u)
        e_prev = error

    # Gráficos
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(t_full, h_full, color="#00FF41", label="Simulado")
    ax1.scatter(t_exp, h_exp, color="red", label="Experimental")
    ax1.set_ylabel("Altura (m)")
    ax1.legend()
    
    ax2.step(t_full, hist_u, color="#00BFFF", label="Acción de Control (u)")
    ax2.set_ylabel("u")
    ax2.set_xlabel("Tiempo (s)")
    ax2.legend()
    st.pyplot(fig)

    # Tabla
    st.write("### 📊 Análisis de Desempeño")
    st.table(pd.DataFrame({
        "Métrica": ["Setpoint", "Nivel Final", "Precisión R²"],
        "Valor": [f"{sp} m", f"{h_full[-1]:.4f} m", f"{r2_score(h_exp, np.interp(t_exp, t_full, h_full)):.4f}"]
    }))
