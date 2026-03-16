import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. CONFIGURACIÓN E IDENTIDAD VISUAL UCV ---
# Configuración inicial de la página para asegurar el layout ancho y el título de la pestaña
st.set_page_config(
    page_title="Tesis UCV - Control de Tanques",
    page_icon="🧪",
    layout="wide"
)

# Estilo CSS personalizado para un acabado profesional institucional
# Se definen colores, sombras y radios de borde para una interfaz moderna
st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    
    /* Estilo para el panel de métricas de desempeño */
    [data-testid="stMetricValue"] { font-size: 1.8rem; color: #1a5276; font-weight: bold; }
    div.stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #1a5276;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Personalización de botones de simulación */
    .stButton>button {
        background-color: #1a5276;
        color: white;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #154360;
        box-shadow: 0 4px 12px rgba(26,82,118,0.3);
        transform: translateY(-1px);
    }
    
    /* Contenedor del panel de análisis lateral */
    .metric-panel {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #e0e6ed;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
    }
    
    /* Tipografía institucional */
    h1 { color: #1a5276; text-shadow: 1px 1px 2px rgba(0,0,0,0.1); }
    h2, h3 { color: #21618c; border-bottom: 2px solid #d4e6f1; padding-bottom: 10px; }
    hr { border: 1px solid #d4e6f1; }
    
    /* Estilo para las tablas de datos en tiempo real */
    div[data-testid="stTable"] { background-color: white; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# Encabezado Institucional: UCV - EIQ
col_l1, col_tit, col_l2 = st.columns([1, 4, 1])

def cargar_logo(nombre_archivo, alias):
    """Función para cargar logos institucionales con manejo de errores de ruta"""
    if os.path.exists(nombre_archivo):
        st.image(nombre_archivo, width=110)
    else:
        st.markdown(f"<div style='text-align:center; border: 2px dashed #bdc3c7; padding: 25px; border-radius:15px; color:#7f8c8d; background:#fff;'>Logo {alias}</div>", unsafe_allow_html=True)

with col_l1:
    cargar_logo("logo_ucv.png", "UCV")
with col_tit:
    st.markdown("<h1 style='text-align: center; margin-top: 10px;'>Práctica de Vaciado de Tanques</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.1em; color: #566573; font-style: italic;'>Escuela de Ingeniería Química - Universidad Central de Venezuela</p>", unsafe_allow_html=True)
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
    * **$Q_{p}$**: Caudal de Perturbación experimental (Fugas o Fallas).
    * **$u$**: Acción de control (Apertura de válvula de entrada).
    
    **Nota Geometría Esférica:** El área se calcula como $A(h) = \pi(2Rh - h^2)$, lo que implica una fuerte no linealidad en el proceso.
    """)

# --- 3. BARRA LATERAL: PARÁMETROS COMPLETOS DEL ENSAYO ---
st.sidebar.header("⚙️ Configuración del Ensayo")

with st.sidebar.container(border=True):
    tipo_proceso = st.sidebar.selectbox("🎯 Operación", ["Llenado", "Vaciado"])
    geometria = st.sidebar.selectbox("📐 Geometría", ["Cilíndrico", "Cónico", "Esférico"])

with st.sidebar.expander("📏 Dimensiones y Setpoint", expanded=True):
    radio_max = st.number_input("Radio Máximo (R) [m]", value=1.0, step=0.1)
    # Cálculo automático de la altura máxima para esferas (Diámetro)
    alt_sug = 3.0 if geometria != "Esférico" else radio_max * 2
    altura_total = st.number_input("Altura Total (H) [m]", value=float(alt_sug), step=0.5)
    setpoint = st.slider("Nivel Deseado (SP) [m]", 0.1, float(altura_total), float(altura_total/2))

with st.sidebar.expander("🌪️ Perturbación experimental ($Q_p$)"):
    hay_p = st.toggle("Activar Falla/Fuga")
    mag_p = st.number_input("Magnitud Qp [m³/s]", value=0.005, format="%.4f") if hay_p else 0.0
    t_p = st.slider("Instante de inicio (s)", 0, 500, 100) if hay_p else 0

with st.sidebar.expander("🎮 Controlador PID"):
    c1, c2, c3 = st.columns(3)
    kp = c1.number_input("Kp", value=2.5)
    ki = c2.number_input("Ki", value=0.5)
    kd = c3.number_input("Kd", value=0.1)
    t_sim = st.sidebar.slider("Tiempo total de simulación [s]", 60, 600, 300)

st.sidebar.markdown("---")
btn_simular = st.sidebar.button("🚀 Iniciar Simulación y Animación", use_container_width=True)

# --- 4. LÓGICA DE SIMULACIÓN Y CÁLCULOS DINÁMICOS ---
def simular_paso(dt, h_prev, sp, geom, r, h_t, m_p, h_p, t_p, err_acum, err_prev):
    """Ejecuta un paso de integración numérica usando el método de Euler"""
    # Cálculo de área con blindaje contra divisiones por cero
    if geom == "Cilíndrico": 
        A_h = np.pi * (r**2)
    elif geom == "Cónico": 
        A_h = np.pi * ( (r / h_t) * max(h_prev, 0.01) )**2
    else: # Geometría Esférica
        A_h = np.pi * (2 * r * max(h_prev, 0.01) - max(h_prev, 0.01)**2)
    
    A_h = max(A_h, 0.01) # Seguridad matemática

    # Algoritmo del Controlador PID
    error = sp - h_prev
    err_acum += error * dt
    der = (error - err_prev) / dt
    u = (kp * error) + (ki * err_acum) + (kd * der)
    
    # Restricciones físicas de la válvula
    qin = np.clip(u, 0, 0.5) 
    # Salida por gravedad (Ley de Torricelli simplificada)
    q_out = 0.6 * 0.05 * np.sqrt(2 * 9.81 * h_prev) if h_prev > 0.001 else 0
    
    # Balance de Masa
    dh_dt = (qin - q_out + m_p) / A_h
    h_new = np.clip(h_prev + dh_dt * dt, 0, h_t)
    
    return h_new, qin, error, err_acum, err_prev

# Configuración del Layout Principal
col_main, col_analysis = st.columns([2, 1])

with col_main:
    st.subheader("🖥️ Monitor del Proceso en Tiempo Real")
    tanque_plot = st.empty()
    grafico_control = st.empty()
    st.subheader("⚙️ Acción del Controlador (Esfuerzo $u$)")
    grafico_u = st.empty()

with col_analysis:
    st.markdown("<div class='metric-panel'>", unsafe_allow_html=True)
    st.subheader("📊 Análisis de Desempeño")
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

# Lógica de ejecución al presionar el botón de inicio
if btn_simular:
    dt = 1.0 # Paso de tiempo para la visualización
    tiempo_anim = np.arange(0, t_sim, dt)
    h_anim, u_anim, e_anim = [], [], []
    
    # Condición inicial según el proceso seleccionado
    h_actual = altura_total if tipo_proceso == "Vaciado" else 0.05
    err_acum, err_prev = 0, 0
    progress_bar = st.progress(0)

    for i, t in enumerate(tiempo_anim):
        # Aplicar perturbación si corresponde
        p_act = mag_p if (hay_p and t >= t_p) else 0.0
        
        # Simular paso actual
        h_actual, qin, error, err_acum, err_prev = simular_paso(dt, h_actual, setpoint, geometria, radio_max, altura_total, p_act, hay_p, t_p, err_acum, err_prev)
        
        h_anim.append(h_actual)
        u_anim.append(qin)
        e_anim.append(error)
        
        # --- 1. Visualización del Tanque ---
        fig_t, ax_t = plt.subplots(figsize=(4, 4))
        ax_t.set_xlim(-1.2*radio_max, 1.2*radio_max)
        ax_t.set_ylim(0, 1.1*altura_total)
        ax_t.set_xticks([]); ax_t.set_ylabel("Altura [m]")
        
        if geometria == "Esférico":
            circulo = plt.Circle((0, radio_max), radio_max, color='#2c3e50', fill=False, lw=3)
            ax_t.add_patch(circulo)
            # Dibujar el nivel de agua en la esfera
            ax_t.axhspan(0, h_actual, xmin=0.25, xmax=0.75, color='#3498db', alpha=0.6)
        else:
            ax_t.plot([-radio_max, -radio_max, radio_max, radio_max], [altura_total, 0, 0, altura_total], color='#2c3e50', lw=3)
            ax_t.add_patch(plt.Rectangle((-radio_max, 0), 2*radio_max, h_actual, color='#3498db', alpha=0.6))
        
        ax_t.axhline(y=setpoint, color='#e74c3c', ls='--', lw=2, label="Setpoint")
        tanque_plot.pyplot(fig_t)
        plt.close(fig_t)
        
        # --- 2. Gráfico de Control (PV vs SP) ---
        fig_c, ax_c = plt.subplots(figsize=(8, 3.5))
        ax_c.plot(tiempo_anim[:len(h_anim)], h_anim, color='#2980b9', lw=2.5, label="Nivel (PV)")
        ax_c.axhline(y=setpoint, color='#e74c3c', ls='--', label="Setpoint")
        ax_c.set_xlim(0, t_sim); ax_c.set_ylim(0, 1.1*altura_total)
        ax_c.grid(True, alpha=0.3); ax_c.legend(loc="lower right")
        grafico_control.pyplot(fig_c)
        plt.close(fig_c)

        # --- 3. TOQUE MAESTRO: Gráfico u Dinámico ---
        fig_u, ax_u = plt.subplots(figsize=(8, 2.5))
        ax_u.step(tiempo_anim[:len(u_anim)], u_anim, color='#e67e22', lw=2, label="Válvula (u)")
        ax_u.fill_between(tiempo_anim[:len(u_anim)], u_anim, step="pre", color='#e67e22', alpha=0.1)
        # Escala adaptativa para que la acción siempre sea visible
        u_max_val = max(u_anim) if u_anim else 0.5
        ax_u.set_ylim(0, max(0.6, u_max_val + 0.1))
        ax_u.set_xlim(0, t_sim); ax_u.grid(True, alpha=0.3); ax_u.legend(loc="upper right")
        grafico_u.pyplot(fig_u)
        plt.close(fig_u)

        # --- 4. Actualización de Análisis en Tiempo Real ---
        metrica_nivel.metric("Nivel Actual", f"{h_actual:.3f} m")
        metrica_error.metric("Error (SP-PV)", f"{error:.4f} m", delta=f"{error:.4f}", delta_color="inverse")
        
        if len(h_anim) > 1:
            mse_val = mean_squared_error(np.full(len(h_anim), setpoint), h_anim)
            r2_val = r2_score(np.full(len(h_anim), setpoint), h_anim)
            metrica_r2.metric("Precisión (R²)", f"{r2_val:.4f}")
            metrica_mse.metric("Error (MSE)", f"{mse_val:.6f}")
        
        tabla_resumen.table(pd.DataFrame({
            "Variable": ["Tiempo Transcurrido", "Fuga Qp", "Régimen"],
            "Valor": [f"{t} s", f"{p_act:.4f} m³/s", f"{tipo_proceso}"]
        }))
        
        time.sleep(0.01)
        progress_bar.progress((i + 1) / len(tiempo_anim))

    # --- 5. RESULTADOS FINALES Y DESCARGA ---
    boton_descarga.download_button(
        label="📥 Descargar Reporte Experimental (CSV)",
        data=pd.DataFrame({"Tiempo":tiempo_anim, "Nivel_PV":h_anim, "Accion_u":u_anim}).to_csv(index=False),
        file_name=f"sim_ucv_{geometria.lower()}.csv",
        use_container_width=True
    )
    st.success("✨ Ensayo completado con éxito. Datos listos para el informe.")
else:
    st.info("💡 Configure los parámetros en la barra lateral y presione 'Iniciar Simulación' para comenzar el ensayo.")
