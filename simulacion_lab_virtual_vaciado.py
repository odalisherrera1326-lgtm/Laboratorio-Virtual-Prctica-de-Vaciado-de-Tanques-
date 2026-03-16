import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import mean_squared_error, r2_score

# =============================================================================
# 1. CONFIGURACIÓN DEL ENTORNO Y ESTÉTICA INSTITUCIONAL (UCV)
# =============================================================================
# Establecemos la configuración de la página para un despliegue tipo Dashboard profesional.
# El layout 'wide' permite que los gráficos de control se vean con mayor resolución.
st.set_page_config(
    page_title="Tesis UCV - Simulación Control de Nivel",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Definición de Estilos CSS Personalizados para cumplir con la identidad visual de la Facultad.
# Se incluyen sombras (box-shadow) y bordes redondeados para un acabado moderno.
st.markdown("""
    <style>
    .main { background-color: #f8fbfc; }
    body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    
    /* Diseño de las tarjetas de métricas para análisis de desempeño */
    [data-testid="stMetricValue"] { font-size: 2rem; color: #1a5276; font-weight: 800; }
    div.stMetric {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        border-left: 8px solid #1a5276;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    div.stMetric:hover { transform: scale(1.02); }
    
    /* Personalización de botones de simulación (Color Azul UCV) */
    .stButton>button {
        background-color: #1a5276;
        color: white;
        border-radius: 12px;
        border: none;
        height: 3.5em;
        font-size: 1.1em;
        font-weight: bold;
        width: 100%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background-color: #154360;
        box-shadow: 0 6px 15px rgba(26,82,118,0.4);
    }
    
    /* Contenedores de información y análisis lateral */
    .metric-panel {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 20px;
        border: 1px solid #e1e8ed;
        box-shadow: 0 10px 35px rgba(0,0,0,0.05);
    }
    
    h1 { color: #1a5276; font-weight: 900; margin-bottom: 5px; }
    h2, h3 { color: #21618c; border-bottom: 2px solid #d4e6f1; padding-bottom: 12px; }
    </style>
    """, unsafe_allow_html=True)

# Bloque de Identidad Institucional: Escuela de Ingeniería Química
col_l1, col_tit, col_l2 = st.columns([1, 4, 1])

def mostrar_logo(ruta_archivo, texto_fallback):
    """Renderiza el logo institucional o un cuadro de texto si el archivo no existe."""
    if os.path.exists(ruta_archivo):
        st.image(ruta_archivo, width=120)
    else:
        st.markdown(f"<div style='text-align:center; border: 2px dashed #bdc3c7; padding: 30px; border-radius:15px; color:#95a5a6; background:#fff;'>[ Logo {texto_fallback} ]</div>", unsafe_allow_html=True)

with col_l1:
    mostrar_logo("logo_ucv.png", "UCV")
with col_tit:
    st.markdown("<h1 style='text-align: center;'>Simulador Virtual de Operaciones Unitarias</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.25em; color: #5d6d7e;'>Diseño de Práctica: Vaciado y Llenado de Tanques Geométricos</p>", unsafe_allow_html=True)
with col_l2:
    mostrar_logo("logo_quimica.png", "Química")

st.markdown("---")

# =============================================================================
# 2. FUNDAMENTOS MATEMÁTICOS Y MARCO TEÓRICO
# =============================================================================
# Este bloque es crucial para la validez académica de la tesis.
with st.expander("📖 Ver Ecuaciones de Diseño y Modelado Físico", expanded=False):
    st.info("El sistema modela el cambio de inventario de masa en un sistema abierto.")
    st.latex(r"A(h) \frac{dh}{dt} = Q_{in} - Q_{out}")
    st.markdown("""
    **Cálculo del Área Transversal $A(h)$ según la Geometría:**
    * **Cilíndrica:** Área constante $A = \pi R^2$.
    * **Cónica:** Área variable proporcional al cuadrado de la altura.
    * **Esférica:** $A(h) = \pi (2Rh - h^2)$. Requiere especial atención en los límites físicos.
    
    **Algoritmo de Control (PID):**
    Se utiliza una implementación discreta para calcular la apertura de la válvula de entrada.
    """)

# =============================================================================
# 3. CONFIGURACIÓN DEL EXPERIMENTO (ENTRADAS DE USUARIO)
# =============================================================================
st.sidebar.header("🔧 Parámetros del Proceso")

# Selección de la naturaleza del experimento
with st.sidebar.container(border=True):
    modo_op = st.sidebar.selectbox("🎯 Tipo de Ensayo", ["Llenado", "Vaciado"])
    forma_tanque = st.sidebar.selectbox("📐 Geometría del Recipiente", ["Cilíndrico", "Cónico", "Esférico"])

# Definición de las dimensiones físicas de la unidad
with st.sidebar.expander("📏 Dimensiones del Tanque", expanded=True):
    r_diseno = st.number_input("Radio de Diseño (R) [m]", value=1.0, min_value=0.1, step=0.1)
    
    # Ajuste dinámico de altura para el caso esférico (H = 2R)
    h_sugerida = 3.0 if forma_tanque != "Esférico" else r_diseno * 2
    h_maxima = st.number_input("Altura Máxima (H) [m]", value=float(h_sugerida), min_value=0.1, step=0.5)
    
    # Consigna de nivel para el sistema de control
    consigna = st.slider("Setpoint de Nivel (SP) [m]", 0.1, float(h_maxima), float(h_maxima/2))

# Configuración del controlador de la planta
with st.sidebar.expander("🎮 Ajuste del Controlador PID"):
    st.caption("Sintonía de las ganancias del lazo de control.")
    kp_val = st.number_input("Proporcional (Kp)", value=2.8, format="%.2f")
    ki_val = st.number_input("Integral (Ki)", value=0.6, format="%.2f")
    kd_val = st.number_input("Derivativo (Kd)", value=0.15, format="%.2f")
    tiempo_ensayo = st.sidebar.slider("Tiempo de Simulación [s]", 60, 600, 300)

st.sidebar.markdown("---")
if st.sidebar.button("▶️ Iniciar Prueba Experimental", use_container_width=True):
    ejecutar_simulacion = True
else:
    ejecutar_simulacion = False

# =============================================================================
# 4. FUNCIONES DE CÁLCULO Y DINÁMICA DE SISTEMAS
# =============================================================================
def resolver_dinamica(dt, h_prev, sp, geom, radio, h_total, error_int, error_der_prev):
    """
    Función núcleo que resuelve la física del problema y la acción de control.
    Implementa el método de integración numérica de Euler.
    """
    # 4.1 Cálculo del área transversal según la altura actual h
    if geom == "Cilíndrico":
        area_act = np.pi * (radio**2)
    elif geom == "Cónico":
        # Relación de semejanza de triángulos para el cono
        area_act = np.pi * ( (radio / h_total) * max(h_prev, 0.01) )**2
    else: # Geometría Esférica
        # Ecuación del área de un círculo a una altura h en la esfera
        area_act = np.pi * (2 * radio * max(h_prev, 0.01) - max(h_prev, 0.01)**2)
    
    # Protección de división por cero para estabilidad numérica
    area_act = max(area_act, 0.005)

    # 4.2 Lógica del Controlador PID
    err = sp - h_prev
    error_int += err * dt
    err_derivativo = (err - error_der_prev) / dt
    
    # Señal de salida u (esfuerzo del controlador)
    u_control = (kp_val * err) + (ki_val * error_int) + (kd_val * err_derivativo)
    
    # 4.3 Fenómenos de Entrada y Salida
    q_entrada = np.clip(u_control, 0, 0.55) # Capacidad máxima de la válvula
    # Ecuación de descarga por gravedad (Bernoulli simplificado)
    q_salida = 0.6 * 0.045 * np.sqrt(2 * 9.81 * h_prev) if h_prev > 0.005 else 0
    
    # 4.4 Balance de Masa: dh/dt = (Qin - Qout) / A
    tasa_cambio = (q_entrada - q_salida) / area_act
    h_nueva = np.clip(h_prev + tasa_cambio * dt, 0, h_total)
    
    return h_nueva, q_entrada, err, error_int, err

# =============================================================================
# 5. ESTRUCTURA DEL PANEL DE RESULTADOS (DASHBOARD)
# =============================================================================
col_viz, col_data = st.columns([2, 1])

with col_viz:
    st.subheader("🔭 Monitoreo en Tiempo Real")
    placeholder_tanque = st.empty()
    st.subheader("📈 Respuesta Dinámica (PV vs SP)")
    placeholder_grafico_nivel = st.empty()
    st.subheader("⚡ Señal del Controlador (Esfuerzo u)")
    placeholder_grafico_u = st.empty()

with col_data:
    st.markdown("<div class='metric-panel'>", unsafe_allow_html=True)
    st.subheader("📊 Indicadores Técnicos")
    met_h = st.empty()
    met_e = st.empty()
    st.markdown("---")
    met_mse = st.empty()
    met_r2 = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("### 📄 Reporte de la Corrida")
    tabla_info = st.empty()
    st.markdown("---")
    descarga_btn_area = st.empty()

# =============================================================================
# 6. BUCLE PRINCIPAL DE LA SIMULACIÓN
# =============================================================================
if ejecutar_simulacion:
    # Inicialización de variables de estado y vectores de datos
    paso_t = 1.0
    vector_tiempo = np.arange(0, tiempo_ensayo, paso_t)
    data_h, data_u, data_e = [], [], []
    
    # Condición de inicio según el tipo de proceso
    nivel_actual = h_maxima if modo_op == "Vaciado" else 0.05
    e_integral, e_pasado = 0, 0
    barra_progreso = st.progress(0)

    for i, t in enumerate(vector_tiempo):
        # Resolver el siguiente paso de tiempo
        nivel_actual, u_accion, error_v, e_integral, e_pasado = resolver_dinamica(
            paso_t, nivel_actual, consigna, forma_tanque, r_diseno, h_maxima, e_integral, e_pasado
        )
        
        # Almacenamiento de resultados para graficación
        data_h.append(nivel_actual); data_u.append(u_accion); data_e.append(error_v)
        
        # --- 6.1 RENDERIZADO DEL TANQUE Y AGUA (GEOMETRÍA DINÁMICA) ---
        fig_t, ax_t = plt.subplots(figsize=(4, 5))
        ax_t.set_xlim(-r_diseno*1.3, r_diseno*1.3)
        ax_t.set_ylim(-0.1, h_maxima*1.1)
        ax_t.set_xticks([]); ax_t.set_ylabel("Nivel de Líquido [m]")

        # Efecto de ondas en la superficie para mayor realismo visual
        ondas = 0.025 * np.sin(t * 3.0) if u_accion > 0.02 else 0
        h_render = nivel_actual + ondas

        if forma_tanque == "Cilíndrico":
            ax_t.plot([-r_diseno, -r_diseno, r_diseno, r_diseno], [h_maxima, 0, 0, h_maxima], color='#2c3e50', lw=4)
            ax_t.add_patch(plt.Rectangle((-r_diseno, 0), 2*r_diseno, h_render, color='#3498db', alpha=0.6))
        elif forma_tanque == "Cónico":
            ax_t.plot([-r_diseno, 0, r_maxima], [h_maxima, 0, h_maxima], color='#2c3e50', lw=4) # Error corregido r_maxima -> r_diseno
            r_h = (r_diseno / h_maxima) * h_render
            ax_t.add_patch(plt.Polygon([[-r_h, h_render], [r_h, h_render], [0, 0]], color='#3498db', alpha=0.6))
        elif forma_tanque == "Esférico":
            ax_t.add_patch(plt.Circle((0, r_diseno), r_diseno, color='#2c3e50', fill=False, lw=4))
            if h_render > 0:
                rad_clip = np.degrees(np.arccos(np.clip(1 - (h_render/r_diseno), -1, 1)))
                ax_t.add_patch(plt.matplotlib.patches.Wedge((0, r_diseno), r_diseno, 270-rad_clip, 270+rad_clip, color='#3498db', alpha=0.6))

        ax_t.axhline(y=consigna, color='#e74c3c', ls='--', lw=2.5, label="Setpoint")
        placeholder_tanque.pyplot(fig_t); plt.close(fig_t)

        # --- 6.2 ACTUALIZACIÓN DE GRÁFICOS DE TENDENCIA ---
        fig_n, ax_n = plt.subplots(figsize=(9, 3.5))
        ax_n.plot(data_h, color='#2980b9', lw=3, label="Nivel Medido (PV)")
        ax_n.axhline(y=consigna, color='#c0392b', ls='--', lw=2, label="Setpoint")
        ax_n.set_xlim(0, tiempo_ensayo); ax_n.set_ylim(0, h_maxima*1.1); ax_n.grid(True, alpha=0.2)
        ax_n.legend(loc="upper right"); placeholder_grafico_nivel.pyplot(fig_n); plt.close(fig_n)

        # Gráfico de la acción de control (Escala Dinámica)
        fig_u, ax_u = plt.subplots(figsize=(9, 2.8))
        ax_u.step(range(len(data_u)), data_u, color='#e67e22', lw=2, label="Válvula (u)")
        ax_u.fill_between(range(len(data_u)), data_u, step="pre", color='#e67e22', alpha=0.15)
        ax_u.set_ylim(0, max(0.65, max(data_u)+0.1)); ax_u.set_xlim(0, tiempo_ensayo); ax_u.grid(True, alpha=0.2)
        placeholder_grafico_u.pyplot(fig_u); plt.close(fig_u)

        # --- 6.3 ACTUALIZACIÓN DE MÉTRICAS Y TABLAS ---
        met_h.metric("Nivel PV [m]", f"{nivel_actual:.3f}")
        met_e.metric("Error de Control", f"{error_v:.4f}", delta=f"{error_v:.4f}", delta_color="inverse")
        
        if len(data_h) > 5:
            val_mse = mean_squared_error(np.full(len(data_h), consigna), data_h)
            met_mse.metric("Precisión (MSE)", f"{val_mse:.6f}")
        
        tabla_info.table(pd.DataFrame({
            "Parámetro": ["Tiempo Transcurrido", "Estado de Válvula", "Operación"],
            "Valor": [f"{t} s", f"{u_accion:.4f}", f"{modo_op}"]
        }))
        
        time.sleep(0.01) # Simulación de tiempo real
        barra_progreso.progress((i + 1) / len(vector_tiempo))

    st.success("✨ Simulación Finalizada Correctamente. Datos listos para exportación.")
    
    # Opción de exportación de datos para análisis externo (Excel/Python)
    df_final = pd.DataFrame({"Tiempo": vector_tiempo, "Nivel_m": data_h, "Apertura_u": data_u})
    descarga_btn_area.download_button(
        "📥 Descargar Datos del Ensayo (CSV)", 
        df_final.to_csv(index=False), 
        "ensayo_ucv_iq.csv", 
        use_container_width=True
    )
else:
    st.info("💡 Configure los parámetros de la planta en la barra lateral y presione 'Iniciar Prueba' para comenzar el experimento virtual.")
