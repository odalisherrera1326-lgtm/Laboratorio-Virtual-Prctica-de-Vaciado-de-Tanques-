import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import mean_squared_error

# =============================================================================
# 1. CONFIGURACIÓN E IDENTIDAD INSTITUCIONAL UCV
# =============================================================================
st.set_page_config(
    page_title="Tesis UCV - Simulador de Tanques",
    page_icon="🧪",
    layout="wide"
)

# Estilos CSS para un acabado de nivel de postgrado
st.markdown("""
    <style>
    .main { background-color: #f9fbfd; }
    [data-testid="stMetricValue"] { font-size: 2rem; color: #1a5276; font-weight: bold; }
    div.stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border-left: 8px solid #1a5276;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }
    .stButton>button {
        background-color: #1a5276;
        color: white;
        border-radius: 10px;
        font-weight: bold;
        height: 3.5em;
        width: 100%;
    }
    h1 { color: #1a5276; }
    h3 { color: #21618c; border-bottom: 2px solid #d4e6f1; padding-bottom: 8px; }
    </style>
    """, unsafe_allow_html=True)

# Encabezado con validación de logotipos
col_l1, col_tit, col_l2 = st.columns([1, 4, 1])

def cargar_imagen_institucional(ruta, etiqueta):
    if os.path.exists(ruta):
        st.image(ruta, width=110)
    else:
        st.markdown(f"<div style='border:2px dashed #ccc; padding:20px; text-align:center;'>{etiqueta}</div>", unsafe_allow_html=True)

with col_l1: cargar_imagen_institucional("logo_ucv.png", "UCV")
with col_tit:
    st.markdown("<h1 style='text-align: center;'>Laboratorio Virtual de Ingeniería Química</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Simulación de Vaciado y Llenado | Tesis de Grado - UCV</p>", unsafe_allow_html=True)
with col_l2: cargar_imagen_institucional("logo_quimica.png", "EIQ")

st.markdown("---")

# =============================================================================
# 2. MARCO TEÓRICO (FENÓMENOS DE TRANSPORTE)
# =============================================================================
with st.expander("📖 Fundamentos del Modelo Matemático", expanded=False):
    st.markdown(r"""
    La variación del nivel $h$ en un tanque esférico de radio $R$ se define por:
    $$ \pi(2Rh - h^2) \frac{dh}{dt} = Q_{in} - Q_{out} + Q_{p} $$
    Donde $Q_p$ representa la perturbación o falla externa introducida en el sistema.
    """)

# =============================================================================
# 3. PANEL DE CONTROL LATERAL (CONFIGURACIÓN)
# =============================================================================
st.sidebar.header("⚙️ Parámetros del Proceso")

with st.sidebar.container(border=True):
    regimen = st.sidebar.selectbox("🎯 Régimen", ["Llenado", "Vaciado"])
    geometria = st.sidebar.selectbox("📐 Geometría", ["Cilíndrico", "Cónico", "Esférico"])

with st.sidebar.expander("📏 Dimensiones Físicas", expanded=True):
    radio_diseno = st.number_input("Radio (R) [m]", value=1.0, step=0.1)
    # En esfera, la altura total es obligatoriamente el diámetro
    h_sug = 3.0 if geometria != "Esférico" else radio_diseno * 2
    altura_tanque = st.number_input("Altura (H) [m]", value=float(h_sug), step=0.5)
    setpoint = st.slider("Nivel Deseado (SP) [m]", 0.1, float(altura_tanque), 1.0)

# BLOQUE DE PERTURBACIÓN (Vital para la defensa)
with st.sidebar.expander("🌪️ Perturbación Experimental ($Q_p$)"):
    activar_p = st.toggle("Activar Falla/Fuga", value=True)
    magnitud_p = st.number_input("Caudal Qp [m³/s]", value=0.060, format="%.4f") if activar_p else 0.0
    inicio_p = st.slider("Instante de inicio (s)", 0, 300, 64) if activar_p else 0

with st.sidebar.expander("🎮 Configuración PID"):
    c1, c2, c3 = st.columns(3)
    kp = c1.number_input("Kp", value=2.50)
    ki = c2.number_input("Ki", value=0.50)
    kd = c3.number_input("Kd", value=0.10)
    t_sim = st.sidebar.slider("Tiempo total [s]", 60, 600, 300)

st.sidebar.markdown("---")
btn_inicio = st.sidebar.button("🚀 Iniciar Prueba Experimental", use_container_width=True)

# =============================================================================
# 4. FUNCIONES DE CÁLCULO Y DINÁMICA
# =============================================================================
def calcular_dinamica(dt, h_p, sp, geom, r, h_t, q_p_act, e_int, e_p):
    # Cálculo de Área Transversal
    if geom == "Cilíndrico":
        A = np.pi * (r**2)
    elif geom == "Cónico":
        A = np.pi * ((r/h_t) * max(h_p, 0.01))**2
    else: # Esférico
        A = np.pi * (2 * r * max(h_p, 0.01) - max(h_p, 0.01)**2)
    
    A = max(A, 0.01) # Protección contra división por cero

    # Lógica PID
    error = sp - h_p
    e_int += error * dt
    e_der = (error - e_p) / dt
    u = (kp * error) + (ki * e_int) + (kd * e_der)
    
    q_in = np.clip(u, 0, 0.6)
    q_out = 0.62 * 0.05 * np.sqrt(2 * 9.81 * h_p) if h_p > 0.01 else 0
    
    # Ecuación Diferencial con Perturbación
    dhdt = (q_in - q_out + q_p_act) / A
    h_n = np.clip(h_p + dhdt * dt, 0, h_t)
    
    return h_n, q_in, error, e_int, error

# =============================================================================
# 5. DASHBOARD DE VISUALIZACIÓN
# =============================================================================
col_graf, col_met = st.columns([2, 1])

with col_graf:
    st.subheader("🖥️ Monitor de Planta")
    visual_tanque = st.empty()
    st.subheader("📈 Respuesta del Sistema (PV)")
    visual_tendencia = st.empty()
    st.subheader("⚙️ Esfuerzo de Control (u)")
    visual_u = st.empty()

with col_met:
    st.markdown("<div class='metric-panel'>", unsafe_allow_html=True)
    st.subheader("📊 Métricas de Control")
    m_h = st.empty(); m_e = st.empty()
    st.markdown("---")
    m_mse = st.empty(); m_r2 = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)
    tabla_resumen = st.empty()
    area_descarga = st.empty()

# =============================================================================
# 6. BUCLE DE SIMULACIÓN Y RENDERIZADO (CORRECCIÓN ESFÉRICA)
# =============================================================================
if btn_inicio:
    dt = 1.0; t_rango = np.arange(0, t_sim, dt)
    h_hist, u_hist, e_hist = [], [], []
    h_actual = altura_tanque if regimen == "Vaciado" else 0.05
    err_i, err_p = 0, 0
    prog = st.progress(0)

    for i, t in enumerate(t_rango):
        # Lógica de perturbación reintegrada
        q_pert = magnitud_p if (activar_p and t >= inicio_p) else 0.0
        
        h_actual, u_val, err_val, err_i, err_p = calcular_dinamica(
            dt, h_actual, setpoint, geometria, radio_diseno, altura_tanque, q_pert, err_i, err_p
        )
        h_hist.append(h_actual); u_hist.append(u_val); e_hist.append(err_val)
        
        # --- DIBUJO PROFESIONAL DEL TANQUE ---
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(-radio_diseno*1.2, radio_diseno*1.2)
        ax.set_ylim(-0.1, altura_tanque*1.1)
        ax.set_xticks([]); ax.set_ylabel("Nivel [m]")

        # Efecto de ondas (Turbulencia)
        ondas = 0.02 * np.sin(t * 4) if (u_val > 0.05 or abs(q_pert) > 0) else 0
        h_vis = h_actual + ondas

        if geometria == "Cilíndrico":
            ax.plot([-radio_diseno, -radio_diseno, radio_diseno, radio_diseno], [altura_tanque, 0, 0, altura_tanque], color='#2c3e50', lw=4)
            ax.add_patch(plt.Rectangle((-radio_diseno, 0), 2*radio_diseno, h_vis, color='#3498db', alpha=0.6))
        elif geometria == "Cónico":
            ax.plot([-radio_diseno, 0, radio_diseno], [altura_tanque, 0, altura_tanque], color='#2c3e50', lw=4)
            r_h = (radio_diseno / altura_tanque) * h_vis
            ax.add_patch(plt.Polygon([[-r_h, h_vis], [r_h, h_vis], [0, 0]], color='#3498db', alpha=0.6))
        elif geometria == "Esférico":
            # 1. Dibujar el contorno circular del tanque
            tanque_circ = plt.Circle((0, radio_diseno), radio_diseno, color='#2c3e50', fill=False, lw=4)
            ax.add_patch(tanque_circ)
            
            # 2. DIBUJO DEL AGUA CURVA (CORRECCIÓN):
            # Usamos un "Wedge" (cuña) centrado en el centro de la esfera (0, R)
            if h_vis > 0:
                # Calculamos el ángulo basado en la altura del fluido
                # h = R - R*cos(theta) => cos(theta) = 1 - h/R
                cos_theta = np.clip(1 - (h_vis/radio_diseno), -1, 1)
                angulo_apertura = np.degrees(np.arccos(cos_theta))
                
                # El fluido se dibuja desde el fondo (270°) hacia ambos lados
                agua_curva = plt.matplotlib.patches.Wedge(
                    (0, radio_diseno), radio_diseno, 
                    270 - angulo_apertura, 270 + angulo_apertura, 
                    color='#3498db', alpha=0.6
                )
                ax.add_patch(agua_curva)

        ax.axhline(y=setpoint, color='#e74c3c', ls='--', lw=2, label="Setpoint")
        visual_tanque.pyplot(fig); plt.close(fig)

        # Gráficos de Tendencia
        f_t, ax_t = plt.subplots(figsize=(8, 3))
        ax_t.plot(h_hist, color='#2980b9', lw=2); ax_t.axhline(y=setpoint, color='red', ls='--')
        ax_t.set_xlim(0, t_sim); ax_t.set_ylim(0, altura_tanque*1.1); visual_tendencia.pyplot(f_t); plt.close(f_t)

        # Métricas
        m_h.metric("Nivel PV [m]", f"{h_actual:.3f}")
        m_e.metric("Error (SP-PV)", f"{err_val:.4f}", delta=f"{err_val:.4f}", delta_color="inverse")
        
        time.sleep(0.01); prog.progress((i+1)/len(t_rango))

    st.success("✨ Ensayo culminado. El nivel en el tanque circular ahora respeta la geometría curva.")
    area_descarga.download_button("📥 Descargar CSV", pd.DataFrame({"t":t_rango,"h":h_hist}).to_csv(), "datos_ucv.csv")
