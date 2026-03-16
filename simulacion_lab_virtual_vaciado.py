import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. CONFIGURACIÓN E IDENTIDAD VISUAL UCV ---
st.set_page_config(
    page_title="Tesis UCV - Control de Tanques",
    page_icon="🧪",
    layout="wide"
)

# Estilo CSS personalizado para un acabado profesional
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stExpander { background-color: #ffffff; border-radius: 10px; }
    h1, h2, h3 { color: #1a5276; }
    div[data-testid="stTable"] { background-color: white; border-radius: 10px; }
    /* Estilo para el panel de métricas fijas */
    .metric-panel { background-color: #eaf2f8; padding: 20px; border-radius: 15px; border: 1px solid #a9cce3;}
    </style>
    """, unsafe_allow_html=True)

# Encabezado Institucional
col_l1, col_tit, col_l2 = st.columns([1, 4, 1])

def cargar_logo(nombre_archivo, alias):
    if os.path.exists(nombre_archivo):
        st.image(nombre_archivo, width=100)
    else:
        st.markdown(f"<div style='text-align:center; border: 1px solid #ccc; padding: 20px; border-radius:10px;'>Logo {alias}</div>", unsafe_allow_html=True)

with col_l1:
    cargar_logo("logo_ucv.png", "UCV")
with col_tit:
    st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>Práctica de Vaciado de Tanques</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2em; color: #5d6d7e;'>Escuela de Ingeniería Química - Universidad Central de Venezuela</p>", unsafe_allow_html=True)
with col_l2:
    cargar_logo("logo_quimica.png", "EIQ")

st.markdown("---")

# --- 2. MARCO TEÓRICO ---
with st.expander("📖 Marco Teórico: Modelo Experimental", expanded=False):
    st.markdown(r"""
    El sistema se basa en el balance de masa dinámico para geometrías variables:
    $$A(h) \frac{dh}{dt} = Q_{in}(u) - Q_{out}(h) \pm Q_{p}$$
    
    Donde:
    * **$h(t)$**: Nivel del líquido (Variable de Proceso, PV).
    * **$A(h)$**: Área transversal variable según la geometría (Cilíndrico, Cónico, Esférico).
    * **$Q_{p}$**: Caudal de Perturbación experimental.
    """)

# --- 3. BARRA LATERAL: PARÁMETROS ---
st.sidebar.header("📋 Configuración del Ensayo")

with st.sidebar.container(border=True):
    tipo_proceso = st.selectbox("🎯 Operación", ["Llenado", "Vaciado"])
    geometria = st.selectbox("📐 Geometría", ["Cilíndrico", "Cónico", "Esférico"])

with st.sidebar.expander("📏 Dimensiones y Setpoint", expanded=True):
    radio_max = st.number_input("Radio Máximo (R) [m]", value=1.0, step=0.1)
    altura_total = st.number_input("Altura Total (H) [m]", value=3.0, step=0.5)
    setpoint = st.slider("Nivel Deseado (SP) [m]", 0.1, altura_total, altura_total/2)

with st.sidebar.expander("🌪️ Perturbación experimental ($Q_p$)"):
    hay_p = st.toggle("Activar Falla/Fuga")
    mag_p = st.number_input("Magnitud Qp [m³/s]", value=0.005, format="%.4f") if hay_p else 0.0
    t_p = st.slider("Instante de inicio (s)", 0, 500, 100) if hay_p else 0

with st.sidebar.expander("🎮 Controlador PID"):
    c1, c2, c3 = st.columns(3)
    kp = c1.number_input("Kp", value=2.5)
    ki = c2.number_input("Ki", value=0.5)
    kd = c3.number_input("Kd", value=0.1)
    t_sim = st.slider("Tiempo total [s]", 60, 600, 300)

st.sidebar.markdown("---")
btn_simular = st.sidebar.button("🚀 Iniciar Simulación y Animación", use_container_width=True)

# --- 4. LÓGICA DE SIMULACIÓN Y ANIMACIÓN ---
def simular_paso(dt, h_prev, sp, geom, r, h_t, m_p, h_p, t_p, err_acum, err_prev):
    # Área según geometría
    if geom == "Cilíndrico": A_h = np.pi * (r**2)
    elif geom == "Cónico": A_h = np.pi * ( (r / h_t) * h_prev )**2
    else: A_h = np.pi * (2 * r * h_prev - h_prev**2)
    A_h = max(A_h, 0.01) # Evitar división por cero

    # Controlador PID
    error = sp - h_prev
    err_acum += error * dt
    der = (error - err_prev) / dt
    u = (kp * error) + (ki * err_acum) + (kd * der)
    
    qin = np.clip(u, 0, 0.5) # Capacidad máxima
    q_out = 0.6 * 0.05 * np.sqrt(2 * 9.81 * h_prev) if h_prev > 0 else 0
    
    # Balance de Masa (Euler)
    dh_dt = (qin - q_out + m_p) / A_h
    h_new = np.clip(h_prev + dh_dt * dt, 0, h_t)
    
    return h_new, qin, error, err_acum, err_prev

# Layout principal: Visualización | Análisis (Fijo)
col_main, col_analysis = st.columns([2, 1])

with col_main:
    st.subheader("Visualización del Proceso")
    tanque_plot = st.empty()
    grafico_control = st.empty()

# Panel de Análisis (Fijo a la derecha)
with col_analysis:
    st.markdown("<div class='metric-panel'>", unsafe_allow_html=True)
    st.subheader("📊 Análisis de Desempeño (Tiempo Real)")
    metrica_nivel = st.empty()
    metrica_error = st.empty()
    st.markdown("---")
    metrica_r2 = st.empty()
    metrica_mse = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("### 📝 Resumen del Ensayo")
    tabla_resumen = st.empty()
    st.markdown("---")
    boton_descarga = st.empty()

# Lógica al presionar el botón
if btn_simular:
    dt = 1.0 # Paso de tiempo para la animación
    tiempo_anim = np.arange(0, t_sim, dt)
    h_anim, u_anim, e_anim = [], [], []
    
    h_actual = altura_total if tipo_proceso == "Vaciado" else 0.01
    err_acum, err_prev = 0, 0
    
    progress_bar = st.progress(0)

    # Bucle de Animación
    for i, t in enumerate(tiempo_anim):
        # 1. Simular un paso
        p_act = mag_p if (hay_p and t >= t_p) else 0.0
        h_actual, qin, error, err_acum, err_prev = simular_paso(dt, h_actual, setpoint, geometria, radio_max, altura_total, p_act, hay_p, t_p, err_acum, err_prev)
        
        # 2. Guardar datos
        h_anim.append(h_actual)
        u_anim.append(qin)
        e_anim.append(error)
        
        # 3. Actualizar Visualización (Columna Izquierda)
        # --- Esquema del Tanque ---
        fig_t, ax_t = plt.subplots(figsize=(4, 4))
        ax_t.set_xlim(-1.2*radio_max, 1.2*radio_max)
        ax_t.set_ylim(0, 1.1*altura_total)
        ax_t.set_title(f"Esquema: Tanque {geometria}")
        ax_t.set_xticks([]); ax_t.set_ylabel("Altura (m)")
        
        if geometria == "Cilíndrico":
            rect_w = plt.Rectangle((-radio_max, 0), 2*radio_max, h_actual, color='#1e88e5', alpha=0.8)
            ax_t.add_patch(rect_w)
            ax_t.plot([-radio_max, -radio_max, radio_max, radio_max], [altura_total, 0, 0, altura_total], color='black', lw=2)
        elif geometria == "Cónico":
            r_w = (radio_max / altura_total) * h_actual
            pts_w = np.array([[-r_w, h_actual], [r_w, h_actual], [0, 0]])
            ax_t.fill(pts_w[:,0], pts_w[:,1], color='#1e88e5', alpha=0.8)
            ax_t.plot([-radio_max, 0, radio_max], [altura_total, 0, altura_total], color='black', lw=2)
            
        ax_t.axhline(y=setpoint, color='red', ls='--', lw=1, label="SP")
        tanque_plot.pyplot(fig_t)
        plt.close(fig_t)
        
        # --- Gráfico de Control ---
        fig_c, ax_c = plt.subplots(figsize=(7, 3.5))
        ax_c.plot(tiempo_anim[:len(h_anim)], h_anim, color='#1e88e5', lw=2.5, label="Nivel (PV)")
        ax_c.axhline(y=setpoint, color='red', ls='--', label="Setpoint")
        if hay_p and t >= t_p:
            ax_c.axvline(x=t_p, color='orange', ls=':', label="Perturbación")
        ax_c.set_xlim(0, t_sim); ax_c.set_ylim(0, 1.1*altura_total)
        ax_c.set_xlabel("Tiempo (s)"); ax_c.set_ylabel("Altura (m)")
        ax_c.legend(loc="lower right"); ax_c.grid(True, alpha=0.2)
        grafico_control.pyplot(fig_c)
        plt.close(fig_c)

        # 4. Actualizar Análisis (Columna Derecha - TIEMPO REAL)
        metrica_nivel.metric("Nivel Actual", f"{h_actual:.3f} m")
        metrica_error.metric("Error (SP-PV)", f"{error:.4f} m", delta=f"{error:.4f}", delta_color="inverse")
        
        # Cálculos de scikit-learn en tiempo real
        h_vec = np.array(h_anim)
        ref_vec = np.full(len(h_vec), setpoint)
        
        if len(h_vec) > 1: # Necesitamos al menos 2 puntos para R2
            mse_rt = mean_squared_error(ref_vec, h_vec)
            r2_rt = r2_score(ref_vec, h_vec)
            
            metrica_r2.metric("Precisión (R²)", f"{r2_rt:.4f}")
            metrica_mse.metric("Error Cuadrático (MSE)", f"{mse_rt:.6f}")
        
        # Actualizar tabla resumen
        tabla_resumen.table(pd.DataFrame({
            "Variable Experimental": ["Tiempo Transcurrido", "Magnitud $Q_p$", "Regímen"],
            "Valor": [f"{t} s", f"{p_act:.4f} m³/s", f"{tipo_proceso}"]
        }))
        
        # Control de velocidad
        time.sleep(0.01)
        progress_bar.progress((i + 1) / len(tiempo_anim))

    # --- 5. RESULTADOS FINALES ---
    # Botón de descarga al terminar
    boton_descarga.download_button(
        label="📥 Descargar Reporte Experimental (CSV)",
        data=pd.DataFrame({"Tiempo":tiempo_anim, "Nivel":h_anim, "Error":e_anim}).to_csv(index=False),
        file_name=f"simulacion_ucv_{geometria.lower()}.csv",
        use_container_width=True
    )
    st.success("🎉 Ensayo completado. Datos listos para el informe.")
else:
    st.info("👈 Configura los parámetros en la barra lateral y haz clic en 'Iniciar Simulación' para ver la animación y el análisis.")
