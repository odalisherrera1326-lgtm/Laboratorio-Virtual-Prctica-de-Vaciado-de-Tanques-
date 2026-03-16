import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. CONFIGURACIÓN E IDENTIDAD VISUAL ---
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
    st.markdown("<p style='text-align: center; color: #7f8c8d;'>Proyecto de Grado</p>", unsafe_allow_html=True)
with col_l2:
    cargar_logo("logo_quimica.png", "EIQ")

st.markdown("---")

# --- 2. MARCO TEÓRICO INTEGRADO ---
with st.expander("📖 Marco Teórico: Balance de Masa Dinámico", expanded=False):
    st.markdown(r"""
    El comportamiento del nivel ($h$) en el tiempo se rige por el balance de masa global:
    $$A(h) \frac{dh}{dt} = Q_{in}(u) - Q_{out}(h) \pm Q_{p}$$
    
    Donde el área superficial $A(h)$ depende de la geometría:
    * **Cilíndrico:** $A(h) = \pi R^2$ (Constante)
    * **Cónico:** $A(h) = \pi \left( \frac{R}{H} h \right)^2$
    * **Esférico:** $A(h) = \pi (2Rh - h^2)$
    
    El controlador PID ajusta la acción de control $u$ para regular $Q_{in}$:
    $$u(t) = K_p e(t) + K_i \int e(t)dt + K_d \frac{de(t)}{dt}$$
    Donde $e(t) = Setpoint (SP) - h(t)$.
    """)

# --- 3. BARRA LATERAL: PARÁMETROS EXPERIMENTALES ---
st.sidebar.header("📋 Configuración del Ensayo")

with st.sidebar.container(border=True):
    tipo_proceso = st.selectbox("🎯 Operación", ["Llenado", "Vaciado"])
    geometria = st.selectbox("📐 Geometría", ["Cilíndrico", "Cónico", "Esférico"])

with st.sidebar.expander("📏 Dimensiones y Setpoint", expanded=True):
    radio_max = st.number_input("Radio Máximo (R) [m]", value=1.0, step=0.1)
    altura_total = st.number_input("Altura Total (H) [m]", value=3.0, step=0.5)
    setpoint = st.slider("Nivel Deseado (SP) [m]", 0.1, altura_total, altura_total/2)

with st.sidebar.expander("🌪️ Perturbación ($Q_p$)"):
    hay_perturbacion = st.toggle("Simular Falla/Fuga")
    mag_pert = st.number_input("Magnitud Qp [m³/s]", value=0.005, format="%.4f") if hay_perturbacion else 0.0
    t_pert = st.slider("Instante de inicio (s)", 0, 500, 100) if hay_perturbacion else 0

with st.sidebar.expander("🎮 Controlador PID"):
    col_p, col_i, col_d = st.columns(3)
    kp = col_p.number_input("Kp", value=2.5)
    ki = col_i.number_input("Ki", value=0.5)
    kd = col_d.number_input("Kd", value=0.1)
    t_sim = st.slider("Tiempo total [s]", 60, 600, 300)

st.sidebar.markdown("---")
if st.sidebar.button("🚀 Iniciar Simulación y Animación", use_container_width=True):
    st.session_state['simular'] = True
else:
    if 'simular' not in st.session_state:
        st.session_state['simular'] = False

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
    
    # Perturbación
    q_p = m_p if h_p else 0
    
    # Balance de Masa (Euler)
    dh_dt = (qin - q_out + q_p) / A_h
    h_new = np.clip(h_prev + dh_dt * dt, 0, h_t)
    
    return h_new, qin, error, err_acum, err_prev

if st.session_state['simular']:
    # Inicialización de la Animación
    dt = 1.0 # Paso de tiempo para la animación (más lento para que se vea)
    tiempo_anim = np.arange(0, t_sim, dt)
    h_anim = []
    u_anim = []
    e_anim = []
    
    h_actual = altura_total if tipo_proceso == "Vaciado" else 0.01
    err_acum, err_prev = 0, 0
    
    # Contenedores para la actualización dinámica
    col_vis, col_status = st.columns([1.5, 1])
    with col_vis:
        st.subheader("Visualización del Proceso")
        tanque_plot = st.empty()
        termometro = st.empty()
        
    with col_status:
        st.subheader("Estado en Tiempo Real")
        metrica_nivel = st.empty()
        metrica_error = st.empty()
        grafico_control = st.empty()

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Bucle de Animación
    for i, t in enumerate(tiempo_anim):
        # 1. Simular un paso
        p_act = hay_perturbacion and t >= t_pert
        h_actual, qin, error, err_acum, err_prev = simular_paso(dt, h_actual, setpoint, geometria, radio_max, altura_total, mag_pert, p_act, t_pert, err_acum, err_prev)
        
        # 2. Guardar datos
        h_anim.append(h_actual)
        u_anim.append(qin)
        e_anim.append(error)
        
        # 3. Actualizar Visualización
        # --- Esquema del Tanque ---
        fig_tanque, ax_t = plt.subplots(figsize=(4, 5))
        ax_t.set_xlim(-1.2*radio_max, 1.2*radio_max)
        ax_t.set_ylim(0, 1.1*altura_total)
        ax_t.set_title(f"Tanque {geometria}")
        ax_t.set_xticks([])
        ax_t.set_ylabel("Altura (m)")
        
        # Dibujar forma del tanque (esquema)
        if geometria == "Cilíndrico":
            rect_t = plt.Rectangle((-radio_max, 0), 2*radio_max, altura_total, color='lightgray', alpha=0.5)
            ax_t.add_patch(rect_t)
            rect_w = plt.Rectangle((-radio_max, 0), 2*radio_max, h_actual, color='#1e88e5', alpha=0.8)
            ax_t.add_patch(rect_w)
        elif geometria == "Cónico":
            # Esquema simplificado
            pts_t = np.array([[-radio_max, altura_total], [radio_max, altura_total], [0, 0]])
            ax_t.fill(pts_t[:,0], pts_t[:,1], color='lightgray', alpha=0.5)
            # Nivel de agua cónico
            r_w = (radio_max / altura_total) * h_actual
            pts_w = np.array([[-r_w, h_actual], [r_w, h_actual], [0, 0]])
            ax_t.fill(pts_w[:,0], pts_w[:,1], color='#1e88e5', alpha=0.8)
        
        ax_t.axhline(y=setpoint, color='red', ls='--', lw=1, label="SP")
        tanque_plot.pyplot(fig_tanque)
        plt.close(fig_tanque)
        
        # --- Termómetro ---
        with termometro:
            st.write(f"**Nivel:**")
            st.progress(h_actual / altura_total)

        # 4. Actualizar Estado
        metrica_nivel.metric("Nivel Actual", f"{h_actual:.3f} m")
        metrica_error.metric("Error (SP-PV)", f"{error:.4f} m", delta=f"{error:.4f}", delta_color="inverse")
        
        # --- Gráfico de Control Dinámico ---
        fig_c, ax_c = plt.subplots(figsize=(6, 3))
        ax_c.plot(tiempo_anim[:len(h_anim)], h_anim, color='#1e88e5', lw=2, label="Nivel (PV)")
        ax_c.axhline(y=setpoint, color='red', ls='--', label="Setpoint")
        if hay_perturbacion and t >= t_pert:
            ax_c.axvline(x=t_pert, color='orange', ls=':', label="Perturbación")
        ax_c.set_xlim(0, t_sim)
        ax_c.set_ylim(0, 1.1*altura_total)
        ax_c.set_xlabel("Tiempo (s)")
        ax_c.legend(loc="lower right")
        ax_c.grid(True, alpha=0.2)
        grafico_control.pyplot(fig_c)
        plt.close(fig_c)
        
        # Controlar la velocidad de la animación
        time.sleep(0.01) # Pequeña pausa para que se vea
        progress_bar.progress((i + 1) / len(tiempo_anim))
        status_text.text(f"Simulando segundo: {t}s / {t_sim}s")

    # --- 5. ANÁLISIS DE DESEMPEÑO FINAL (ESTILO VIDEO) ---
    st.divider()
    st.header("📊 Análisis de Desempeño Final")
    
    h_final_vec = np.array(h_anim)
    ref = np.full(len(h_final_vec), setpoint)
    mse = mean_squared_error(ref, h_final_vec)
    r2 = r2_score(ref, h_final_vec)
    n_fin = h_final_vec[-1]
    err_r = setpoint - n_fin
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Setpoint (SP)", f"{setpoint} m")
    c2.metric("Nivel Final (PV)", f"{n_fin:.3f} m")
    c3.metric("Error Residual", f"{err_r:.4f} m", delta=f"{err_r:.4f}", delta_color="inverse")
    c4.metric("Precisión R²", f"{r2:.4f}")
    
    st.write("**Resumen del Ensayo Técnico**")
    resumen = pd.DataFrame({
        "Métrica de Tesis": ["Error Cuadrático (MSE)", "Geometría Seleccionada", "Perturbación Aplicada ($Q_p$)", "Régimen"],
        "Valor": [f"{mse:.8f}", geometria, f"{mag_pert:.4f} m³/s" if hay_perturbacion else "NO", tipo_proceso]
    })
    st.table(resumen)
    
    # Exportación
    st.download_button(
        label="📥 Descargar Reporte de Datos (CSV)",
        data=pd.DataFrame({"Tiempo":tiempo_anim, "Nivel":h_anim, "Error":e_anim}).to_csv(index=False),
        file_name=f"simulacion_ucv_{geometria.lower()}.csv",
        use_container_width=True
    )
    
    st.success("🎉 Simulación y animación completadas con éxito.")
else:
    st.info("👈 Configura los parámetros en la barra lateral y haz clic en 'Iniciar Simulación' para ver la animación.")
