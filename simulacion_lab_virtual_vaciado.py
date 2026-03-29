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
    page_title="Tesis UCV - Simulación Dinámica",
    page_icon="🧪",
    layout="wide"
)

# Inicialización del estado de simulación para el Reset
if 'ejecutando' not in st.session_state:
    st.session_state.ejecutando = False

# Estilos CSS Avanzados para una interfaz profesional y académica
st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    [data-testid="stMetricValue"] { font-size: 1.8rem; color: #1a5276; font-weight: bold; }
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
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #154360;
        box-shadow: 0 4px 15px rgba(26,82,118,0.3);
    }
    /* Estilo para el botón de Reset (color rojo UCV) */
    div.stButton > button:first-child[kind="secondary"] {
        background-color: #943126;
        color: white;
        border: none;
    }
    h1 { color: #1a5276; text-align: center; }
    h3 { color: #21618c; border-bottom: 2px solid #d4e6f1; padding-bottom: 8px; }
    .instruccion-box {
        background-color: #e8f4f8;
        padding: 30px;
        border-radius: 15px;
        border: 1px dashed #1a5276;
        text-align: center;
        color: #1a5276;
        font-size: 1.2em;
    }
    </style>
    """, unsafe_allow_html=True)

# Encabezado Institucional: Escuela de Ingeniería Química
col_l1, col_tit, col_l2 = st.columns([1, 4.5, 1.5])

def render_logo_institucional(ruta, nombre):
    if os.path.exists(ruta):
        st.image(ruta, width=110)
    else:
        st.markdown(f"<div style='border:1px solid #ccc; padding:10px;'>{nombre}</div>", unsafe_allow_html=True)

with col_l1: 
    #izquierda
    render_logo_institucional("logo_ucv.png", "UCV")

with col_tit:
    st.markdown("<h1>Práctica Virtual: Balance en estado no estacionario</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #5d6d7e;'>Escuela de Ingeniería Química | Facultad de Ingeniería - UCV</p>", unsafe_allow_html=True)

with col_l2: 
    # Aquí llamamos directamente a st.image para darle un tamaño mayor (ejemplo: 180)
    if os.path.exists("logo_quimica.png"):
        st.image("logo_quimica.png", width=180) 
    else:
        st.markdown("<div style='border:1px solid #ccc; padding:10px;'>EIQ</div>", unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# 2. MARCO TEÓRICO: BALANCE DE MASA Y TORRICELLI
# =============================================================================
with st.expander("📖 Marco Teórico: Ecuaciones de Conservación y Descarga", expanded=False):
    st.markdown(r"""
    La dinámica del sistema se describe mediante el **Balance Global de Masa** para un volumen de control con densidad constante ($\rho$):
    
    $$ \frac{dV}{dt} = Q_{in} - Q_{out} \pm Q_{p} $$
    
    Considerando que el volumen es función del nivel ($V = \int A(h)dh$), aplicamos la regla de la cadena para obtener la ecuación general de vaciado/llenado válida para **cualquier área transversal $A(h)$**:
    
    $$ A(h) \frac{dh}{dt} = Q_{in} - (C_d \cdot a \cdot \sqrt{2gh}) \pm Q_{p} $$
    
    Donde:
    * **$A(h)$**: Área de la sección transversal en función de la altura (m²).
    * **$Q_{in}$**: Flujo de entrada controlado (m³/s).
    * **$Q_{out}$**: Flujo de salida basado en la **Ley de Torricelli** (m³/s).
    * **$C_d$**: Coeficiente de descarga (adimensional).
    * **$a$**: Área del orificio de salida (m²).
    * **$Q_{p}$**: Flujo de perturbación o falla (m³/s).
    """)

# =============================================================================
# 3. BARRA LATERAL: PARÁMETROS TÉCNICOS
# =============================================================================
st.sidebar.header("⚙️ Configuración del Sistema")

with st.sidebar.container(border=True):
    op_tipo = st.sidebar.selectbox("🎯 Operación Principal", ["Llenado", "Vaciado"])
    geom_tanque = st.sidebar.selectbox("📐 Geometría del Equipo", ["Cilíndrico", "Cónico", "Esférico"])

with st.sidebar.expander("📏 Especificaciones del Tanque", expanded=True):
    r_max = st.number_input("Radio de Diseño (R) [m]", value=1.0, min_value=0.1, step=0.1)
    h_sug = 3.0 if geom_tanque != "Esférico" else r_max * 2
    h_total = st.number_input("Altura de Diseño (H) [m]", value=float(h_sug), min_value=0.1, step=0.5)
    sp_nivel = st.slider("Consigna de Nivel (Setpoint) [m]", 0.1, float(h_total), float(h_total/2))

with st.sidebar.expander("🌪️ Escenario de Perturbación ($Q_p$)"):
    p_activa = st.toggle("Simular Falla/Fuga Externas", value=True)
    p_magnitud = st.number_input("Magnitud Qp [m³/s]", value=0.045, format="%.4f") if p_activa else 0.0
    p_tiempo = st.slider("Inicio de perturbación [s]", 0, 500, 80) if p_activa else 0

with st.sidebar.expander("🎮 Parámetros del Controlador PID"):
    c1, c2, c3 = st.columns(3)
    kp_val = c1.number_input("Kp", value=2.6)
    ki_val = c2.number_input("Ki", value=0.5)
    kd_val = c3.number_input("Kd", value=0.1)
    tiempo_ensayo = st.sidebar.slider("Tiempo de simulación [s]", 60, 600, 300)

st.sidebar.markdown("---")
# Botones de Control
col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    iniciar_sim = st.button("🚀 Iniciar", use_container_width=True)
with col_btn2:
    btn_reset = st.button("🔄 Reset", use_container_width=True, type="secondary")

if btn_reset:
    st.session_state.ejecutando = False
    st.rerun()

# =============================================================================
# 4. LÓGICA DE CÁLCULO: MÉTODO DE EULER
# =============================================================================
def resolver_sistema(dt, h_prev, sp, geom, r, h_t, q_p_val, e_sum, e_prev):
    if geom == "Cilíndrico":
        area_h = np.pi * (r**2)
    elif geom == "Cónico":
        area_h = np.pi * ((r/h_t) * max(h_prev, 0.01))**2
    else: # Esférico
        area_h = np.pi * (2 * r * max(h_prev, 0.01) - max(h_prev, 0.01)**2)
    
    area_h = max(area_h, 0.01) 

    err = sp - h_prev
    e_sum += err * dt
    e_der = (err - e_prev) / dt
    u_control = (kp_val * err) + (ki_val * e_sum) + (kd_val * e_der)
    
    q_entrada = np.clip(u_control, 0, 0.6)
    q_salida = 0.61 * 0.04 * np.sqrt(2 * 9.81 * h_prev) if h_prev > 0.005 else 0
    
    dh_dt = (q_entrada - q_salida + q_p_val) / area_h
    h_next = np.clip(h_prev + dh_dt * dt, 0, h_t)
    
    return h_next, q_entrada, err, e_sum, err

# =============================================================================
# 5. ESTRUCTURA DE LA INTERFAZ (DASHBOARD)
# =============================================================================
col_graf, col_met = st.columns([2, 1])

with col_graf:
    st.subheader("🖥️ Monitor del Proceso")
    placeholder_tanque = st.empty()
    st.subheader("📊 Tendencia Temporal de Nivel")
    placeholder_grafico = st.empty()
    st.subheader("⚙️ Acción del Controlador (Caudal de Entrada)")
    placeholder_u = st.empty()

with col_met:
    st.markdown("<div class='metric-panel'>", unsafe_allow_html=True)
    st.subheader("📊 Métricas de Control")
    m_h = st.empty(); m_e = st.empty(); m_mse = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)
    tabla_resumen = st.empty()
    area_descarga = st.empty()

# =============================================================================
# 6. BUCLE DE SIMULACIÓN Y CONTROL DE ESTADOS (CORREGIDO PARA TIEMPO REAL)
# =============================================================================
if iniciar_sim:
    st.session_state.ejecutando = True

if not st.session_state.ejecutando:
    # Mensaje de espera con el estilo de tu tesis
    placeholder_tanque.markdown("""
        <div class='instruccion-box'>
            <h3>⚠️ Sistema en Espera</h3>
            <p>Por favor, ajuste los parámetros en la barra lateral y presione <b>'Iniciar'</b> para comenzar el experimento virtual.</p>
        </div>
    """, unsafe_allow_html=True)
else:
    # 1. Preparación de variables de estado
    dt = 1.0 
    vector_t = np.arange(0, tiempo_ensayo, dt)
    h_log, u_log = [], []
    h_corrida = h_total if op_tipo == "Vaciado" else 0.05
    err_int, err_pasado = 0, 0
    
    barra_p = st.progress(0)

    # 2. Bucle principal de cálculo y renderizado
    for i, t_act in enumerate(vector_t):
        # Lógica de perturbación
        q_p_inst = p_magnitud if (p_activa and t_act >= p_tiempo) else 0.0
        
        # Resolución numérica (Euler + PID)
        h_corrida, u_inst, e_inst, err_int, err_pasado = resolver_sistema(
            dt, h_corrida, sp_nivel, geom_tanque, r_max, h_total, q_p_inst, err_int, err_pasado
        )
        
        # Almacenamiento para las gráficas de tendencia
        h_log.append(h_corrida)
        u_log.append(u_inst)
        
        # --- ACTUALIZACIÓN DE GRÁFICAS EN TIEMPO REAL ---
        
        # A. Dibujo del Tanque Animado
        fig_t, ax_t = plt.subplots(figsize=(5, 4))
        ax_t.set_xlim(-r_max*1.2, r_max*1.2)
        ax_t.set_ylim(-0.1, h_total*1.1)
        ax_t.set_xticks([]); ax_t.set_ylabel("Nivel [m]")
        
        # Simulación de oleaje visual si hay flujo de entrada
        h_vis = h_corrida + (0.02 * np.sin(t_act * 4) if u_inst > 0.05 else 0)
        
        if geom_tanque == "Cilíndrico":
            ax_t.add_patch(plt.Rectangle((-r_max, 0), 2*r_max, h_vis, color='#3498db', alpha=0.6))
            ax_t.plot([-r_max, -r_max, r_max, r_max], [h_total, 0, 0, h_total], color='#2c3e50', lw=3)
        elif geom_tanque == "Cónico":
            r_h = (r_max / h_total) * h_vis
            ax_t.add_patch(plt.Polygon([[-r_h, h_vis], [r_h, h_vis], [0, 0]], color='#3498db', alpha=0.6))
            ax_t.plot([-r_max, 0, r_max], [h_total, 0, h_total], color='#2c3e50', lw=3)
        elif geom_tanque == "Esférico":
            ax_t.add_patch(plt.Circle((0, r_max), r_max, color='#2c3e50', fill=False, lw=3))
            if h_vis > 0:
                ang_w = np.degrees(np.arccos(np.clip(1 - (h_vis/r_max), -1, 1)))
                ax_t.add_patch(plt.matplotlib.patches.Wedge((0, r_max), r_max, 270-ang_w, 270+ang_w, color='#3498db', alpha=0.6))

        ax_t.axhline(y=sp_nivel, color='red', ls='--', label=f"SP: {sp_nivel}m")
        placeholder_tanque.pyplot(fig_t)
        plt.close(fig_t) # <--- Muy importante para que no se sature la memoria

        # B. Gráfica de Tendencia de Nivel
        fig_tr, ax_tr = plt.subplots(figsize=(8, 3.5))
        # Graficamos el tiempo transcurrido hasta el momento i
        ax_tr.plot(vector_t[:i+1], h_log, color='#2980b9', lw=2, label="Nivel (PV)")
        ax_tr.axhline(y=sp_nivel, color='red', ls='--', alpha=0.5)
        ax_tr.set_xlim(0, tiempo_ensayo) # Mantiene el eje X fijo para ver el avance
        ax_tr.set_ylim(0, h_total*1.1)
        ax_tr.set_xlabel("Tiempo [s]"); ax_tr.set_ylabel("Altura [m]")
        ax_tr.grid(True, alpha=0.2)
        placeholder_grafico.pyplot(fig_tr)
        plt.close(fig_tr)

        # C. Acción del Controlador
        fig_u, ax_u = plt.subplots(figsize=(8, 2.5))
        ax_u.step(vector_t[:i+1], u_log, color='#e67e22', where='post')
        ax_u.set_xlim(0, tiempo_ensayo)
        ax_u.set_ylim(0, 0.7)
        ax_u.set_ylabel("u [m³/s]")
        placeholder_u.pyplot(fig_u)
        plt.close(fig_u)

        # Actualización de métricas de texto
        m_h.metric("Nivel PV [m]", f"{h_corrida:.3f}")
        m_e.metric("Error de Control", f"{e_inst:.4f} m")
        
        # Pausa breve para permitir que Streamlit renderice
        time.sleep(0.01) 
        barra_p.progress((i+1)/len(vector_t))

    # --- FINALIZACIÓN ---
    st.success(f"✨ Simulación de {geom_tanque} completada.")
    st.balloons()
    
    df_descarga = pd.DataFrame({"Tiempo [s]": vector_t, "Nivel [m]": h_log, "Caudal [m3/s]": u_log})
    area_descarga.download_button(
        "📥 Descargar Datos del Ensayo (CSV)", 
        df_descarga.to_csv(index=False), 
        "resultados_simulacion_ucv.csv", 
        use_container_width=True)
    
