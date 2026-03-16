    import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import r2_score

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="LOU Virtual - UCV", layout="wide")

# --- 2. ENCABEZADO INSTITUCIONAL ---
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

# --- 3. DATOS EXPERIMENTALES (MATLAB) ---
# Datos de vaciado y llenado obtenidos de tus experiencias previas
t_v_real = np.array([0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 580])
h_v_real = np.array([35, 32, 30, 26, 23, 20.5, 18.5, 15.5, 12.5, 11, 9, 6.5, 4.5, 2, 0.5]) / 100

t_l_real = np.array([0, 76, 152, 228, 304, 380, 456, 532, 608, 684, 760, 836, 912, 988])
h_l_real = np.array([0, 4, 7, 9.5, 13, 15, 17, 20.5, 23, 25, 28.2, 30.2, 33.5, 35]) / 100

# --- 4. BARRA LATERAL (CONFIGURACIÓN) ---
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
    u_bias = st.number_input("Sesgo (u0)", value=0.00037, format="%.6f")

    if modo == "Llenado":
        st.header("🌊 Rotámetro")
        plomada = st.slider("Plomada (cm)", 0.0, 10.0, 4.0)
        # Ecuación de calibración de tu script de MATLAB
        q_alim = ((39.739 * plomada + 106.9) / 1000) / 3600
    else:
        q_alim = 0.0

    st.markdown("---")
    st.header("⚠️ Perturbación")
    act_perturbacion = st.checkbox("Activar Perturbación")
    t_perturbacion = st.slider("Inicio (s)", 0, 1000, 300)
    magnitud_p = st.slider("Magnitud (%)", -50, 50, 20) / 100

# --- 5. BASE TEÓRICA ---
with st.expander("📚 Fundamentos Teóricos", expanded=False):
    col_a, col_b = st.columns([2, 1])
    col_a.write(f"Simulación de la práctica de **{modo}** para la tesis de la EIQ-UCV.")
    col_b.latex(r"\frac{dh}{dt} = \frac{Q_{in} - Q_{out}}{A(h)}")

# --- 6. FUNCIONES DE CÁLCULO ---
def calc_area(h):
    if tipo_tanque == "Cilíndrico": 
        return np.pi * r_base**2
    elif tipo_tanque == "Troncocónico":
        radio_h = r_base + (0.45 - r_base) * (h / 1.0)
        return np.pi * radio_h**2
    else: 
        return np.pi * (2 * r_base * h - h**2) if h > 0 else 1e-5

# --- 7. EJECUCIÓN DE LA SIMULACIÓN ---
if st.button("🚀 Iniciar Simulación"):
    t_exp = t_l_real if modo == "Llenado" else t_v_real
    h_exp = h_l_real if modo == "Llenado" else h_v_real
    
    t_sim = np.arange(0, t_exp[-1] + 1, 1.0)
    h_actual, e_sum, e_prev = h_exp[0], 0, 0
    h_hist, u_hist = [], []

    for t in t_sim:
        # Aplicación de perturbación
        q_in_efectivo = q_alim
        if act_perturbacion and t >= t_perturbacion:
            q_in_efectivo = q_alim * (1 + magnitud_p)

        # Lógica del Controlador PID
        error = sp - h_actual
        e_sum += error
        u = u_bias + (kp * (error + (1/ti)*e_sum + td*(error - e_prev)))
        u = np.clip(u, 0, 0.01) # Límite físico de la válvula
        
        # Balance de Masa
        area = calc_area(h_actual)
        q_out = u * np.sqrt(2 * 9.81 * h_actual) if h_actual > 0 else 0
        h_actual += ((q_in_efectivo - q_out) / area)
        h_actual = max(0, h_actual)
        
        h_hist.append(h_actual)
        u_hist.append(u)
        e_prev = error

    # --- 8. VISUALIZACIÓN ---
    col_graf, col_res = st.columns([3, 1])

    with col_graf:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Gráfico de Nivel
        ax1.plot(t_sim, h_hist, color="#1f77b4", label="Simulado (PV)", lw=2)
        ax1.scatter(t_exp, h_exp, color="red", label="Experimental (MATLAB)", s=20)
        ax1.axhline(sp, color="orange", ls="--", label="Setpoint (SP)")
        ax1.set_ylabel("Altura (m)")
        ax1.set_title(f"Dinámica de {modo} - Tanque {tipo_tanque}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Gráfico de Acción de Control
        ax2.step(t_sim, u_hist, color="#2ca02c", label="Salida Controlador (u)")
        ax2.set_ylabel("Apertura Válvula")
        ax2.set_xlabel("Tiempo (s)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig)

    with col_res:
        # Análisis de Desempeño
        st.subheader("📊 Desempeño")
        h_interp = np.interp(t_exp, t_sim, h_hist)
        prec_r2 = r2_score(h_exp, h_interp)
        mse_val = np.mean((h_exp - h_interp)**2)
        
        st.metric("Precisión R²", f"{prec_r2:.4f}")
        st.metric("MSE", f"{mse_val:.6e}")
        
        # Descarga de Datos
        st.markdown("---")
        st.subheader("💾 Exportar")
        df_out = pd.DataFrame({
            'Tiempo_s': t_sim,
            'Altura_m': h_hist,
            'Control_u': u_hist
        })
        csv_data = df_out.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar CSV",
            data=csv_data,
            file_name=f"simulacion_{modo.lower()}.csv",
            mime="text/csv"
        )

    if act_perturbacion:
        st.warning(f"Simulación con perturbación del {magnitud_p*100}% a los {t_perturbacion}s.")
