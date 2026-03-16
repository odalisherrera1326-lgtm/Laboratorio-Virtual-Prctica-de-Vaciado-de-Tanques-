import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Importación segura de métricas
try:
    from sklearn.metrics import r2_score
except ImportError:
    st.error("Error: 'scikit-learn' no detectado. Añádelo al archivo requirements.txt.")

# --- 1. CONFIGURACIÓN Y ENCABEZADO ---
st.set_page_config(page_title="LOU Virtual - UCV", layout="wide")

# Diseño de cabecera con logos (Asegúrate de subirlos a GitHub como .png)
col_l1, col_tit, col_l2 = st.columns([1, 4, 1])

with col_l1:
    if os.path.exists("logo_ucv.png"):
        st.image("logo_ucv.png", width=110)
    else:
        st.markdown("🏛️ **UCV**")

with col_tit:
    st.markdown("<h1 style='text-align: center; color: #1a5276;'>Laboratorio Virtual: Vaciado y Llenado de Tanques</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Facultad de Ingeniería - Escuela de Ingeniería Química</h3>", unsafe_allow_html=True)

with col_l2:
    if os.path.exists("logo_quimica.png"):
        st.image("logo_quimica.png", width=110)
    else:
        st.markdown("🧪 **EIQ**")

st.markdown("---")

# --- 2. DATOS EXPERIMENTALES (Sincronizados con tus scripts de MATLAB) ---
# Datos de Vaciado
t_v_real = np.array([0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 580])
h_v_real = np.array([35, 32, 30, 26, 23, 20.5, 18.5, 15.5, 12.5, 11, 9, 6.5, 4.5, 2, 0.5]) / 100

# Datos de Llenado
t_l_real = np.array([0, 76, 152, 228, 304, 380, 456, 532, 608, 684, 760, 836, 912, 988])
h_l_real = np.array([0, 4, 7, 9.5, 13, 15, 17, 20.5, 23, 25, 28.2, 30.2, 33.5, 35]) / 100

# --- 3. BARRA LATERAL (CONFIGURACIÓN) ---
with st.sidebar:
    st.header("📋 Configuración del Sistema")
    modo = st.radio("Seleccione la Operación", ["Vaciado", "Llenado"])
    tipo_tanque = st.selectbox("Seleccione la Geometría", ["Cilíndrico", "Troncocónico", "Esférico"])
    diametro = st.number_input("Diámetro Base (m)", value=0.5030, format="%.4f")
    r_base = diametro / 2

    st.markdown("---")
    st.header("🎮 Parámetros PID")
    sp = st.slider("Setpoint (m)", 0.0, 0.5, 0.25)
    kp = st.number_input("Ganancia (Kp)", value=1.0)
    ti = st.number_input("Tiempo Integral (Ti)", value=20.0)
    td = st.number_input("Tiempo Derivativo (Td)", value=1.0)
    u_bias = st.number_input("Sesgo (u0 / K)", value=0.00037, format="%.6f")

    if modo == "Llenado":
        st.header("🌊 Entrada (Rotámetro)")
        plomada = st.slider("Plomada (cm)", 0.0, 10.0, 4.0)
        # Ecuación de calibración del rotámetro de tu script
        q_alim = ( (39.739 * plomada + 106.9) / 1000 ) / 3600
        st.info(f"Caudal entrada: {q_alim:.2e} m³/s")
    else:
        q_alim = 0.0

# --- 4. FUNDAMENTOS TEÓRICA ---
with st.expander("📚 Fundamentos: Geometría y Balance de Masa", expanded=True):
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        st.write(f"Para el proceso de **{modo.lower()}**, resolvemos la ecuación diferencial:")
        st.info("Nota: En tanques no cilíndricos, el área A(h) es variable, aumentando la complejidad del control.")
    with col_t2:
        st.latex(r"\frac{dh}{dt} = \frac{Q_{in} - Q_{out}}{A(h)}")

# --- 5. FUNCIONES DE CÁLCULO ---
def obtener_area(h):
    if tipo_tanque == "Cilíndrico":
        return np.pi * r_base**2
    elif tipo_tanque == "Troncocónico":
        r_tope, H_total = 0.45, 1.0 # Valores de diseño UCV
        radio_h = r_base + (r_tope - r_base) * (h / H_total)
        return np.pi * radio_h**2
    else: # Esférico
        return np.pi * (2 * r_base * h - h**2) if h > 0 else 1e-5

# --- 6. SIMULACIÓN Y GRÁFICOS ---
if st.button("🚀 Iniciar Simulación"):
    t_exp = t_l_real if modo == "Llenado" else t_v_real
    h_exp = h_l_real if modo == "Llenado" else h_v_real
    
    dt = 1.0
    t_full = np.arange(0, t_exp[-1] + 1, dt)
    h_actual, err_acum, e_prev = h_exp[0], 0, 0
    h_full, hist_u = [], []

    # Bucle de Control PID
    for t in t_full:
        error = sp - h_actual
        err_acum += error * dt
        derivada = (error - e_prev) / dt
        
        # Respuesta del Controlador (Acción u)
        u = u_bias + (kp * (error + (1/ti) * err_acum + td * derivada))
        u = np.clip(u, 0, 0.01) # Límite físico de la válvula
        
        area_v = obtener_area(h_actual)
        q_out = u * np.sqrt(2 * 9.81 * h_actual) if h_actual > 0 else 0
        
        dhdt = (q_alim - q_out) / area_v
        h_actual += dhdt * dt
        h_actual = max(0, h_actual)
        
        h_full.append(h_actual)
        hist_u.append(u)
        e_prev = error

    # Cálculos de Desempeño
    h_predicho = np.interp(t_exp, t_full, h_full)
    r2 = r2_score(h_exp, h_predicho)

    # --- ZONA DE GRÁFICOS ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    plt.subplots_adjust(hspace=0.25)

    # PV vs SP
    ax1.plot(t_full, h_full, label="Nivel Simulado (PV)", color="#00FF41", lw=2.5)
    ax1.scatter(t_exp, h_exp, color="red", label="Experimental MATLAB", s=20)
    ax1.axhline(sp, color="red", ls="--", label="Setpoint")
    ax1.set_ylabel("Altura (m)")
    ax1.set_title(f"Respuesta del Sistema - {tipo_tanque}")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    # Respuesta del Controlador (Gráfico pedido)
    ax2.step(t_full, hist_u, color="#00BFFF", label="Respuesta del Controlador (u)")
    ax2.set_ylabel("Apertura Válvula (u)")
    ax2.set_xlabel("Tiempo (s)")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)
    
    st.pyplot(fig)

    # --- MÉTRICAS FINALES ---
    st.write("### 📊 Análisis de Desempeño")
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Precisión R²", f"{r2:.4f}")
    col_m2.metric("Nivel Final", f"{h_full[-1]:.3f} m")
    col_m3.metric("Error Final", f"{abs(sp - h_full[-1]):.4e} m")

    if r2 > 0.98:
        st.success("✅ El modelo matemático tiene un ajuste excelente con los datos de laboratorio.")
    else:
        st.warning("⚠️ Se recomienda ajustar los parámetros PID para mejorar el seguimiento del Setpoint.")

    # Opción de descarga
    df_descarga = pd.DataFrame({"Tiempo(s)": t_full, "Nivel(m)": h_full, "Accion_Control": hist_u})
    st.download_button("📥 Descargar Resultados (CSV)", df_descarga.to_csv(index=False), "resultados_lou_ucv.csv")
