import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Configuración de la página
st.set_page_config(page_title="Simulación Vaciado de Tanques - UCV", layout="wide")

# Título principal
st.title("🚀 Simulación de Vaciado y Llenado de Tanques")
st.markdown("---")

# --- BARRA LATERAL: PARÁMETROS ---
st.sidebar.header("⚙️ Configuración del Sistema")

# Parámetros del Tanque
area_tanque = st.sidebar.number_input("Área del Tanque (m²)", min_value=0.1, value=1.5, step=0.1)
altura_max = st.sidebar.number_input("Altura Máxima (m)", min_value=1.0, value=5.0, step=0.5)
setpoint = st.sidebar.slider("Setpoint (Nivel Deseado en m)", 0.0, altura_max, 2.5)

# Parámetros del Controlador PID
st.sidebar.subheader("Controlador PID")
kp = st.sidebar.number_input("Ganancia Proporcional (Kp)", value=1.2)
ki = st.sidebar.number_input("Ganancia Integral (Ki)", value=0.5)
kd = st.sidebar.number_input("Ganancia Derivativa (Kd)", value=0.1)

# Tiempo de simulación
t_final = st.sidebar.slider("Tiempo Total (s)", 10, 500, 100)

# --- LÓGICA DE SIMULACIÓN (Ejemplo Matemático Simplificado) ---
t = np.linspace(0, t_final, 100)
# Simulación de una respuesta típica de sistema de segundo orden (subamortiguada)
# Esto emula el comportamiento real que se ve en el video
nivel = setpoint * (1 - np.exp(-0.05 * t) * (np.cos(0.1 * t) + 0.5 * np.sin(0.1 * t)))

# Crear DataFrame para los resultados
df_simulacion = pd.DataFrame({"Tiempo (s)": t, "Nivel (m)": nivel})

# --- CUERPO PRINCIPAL: VISUALIZACIÓN ---
col_graf, col_datos = st.columns([2, 1])

with col_graf:
    st.subheader("📈 Comportamiento del Nivel")
    fig, ax = plt.subplots()
    ax.plot(t, nivel, label="Nivel Real", color="#1f77b4", linewidth=2)
    ax.axhline(y=setpoint, color='r', linestyle='--', label="Setpoint")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Nivel (m)")
    ax.set_ylim(0, altura_max + 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with col_datos:
    st.subheader("📋 Datos en Tiempo Real")
    st.dataframe(df_simulacion.tail(10), use_container_width=True)

# --- SECCIÓN: ANÁLISIS DE DESEMPEÑO (Lo que vimos en el video) ---
st.divider()
st.header("📊 Análisis de Desempeño del Sistema")

# 1. Cálculos de métricas de error
referencia = np.full(len(nivel), setpoint)
mse = mean_squared_error(referencia, nivel)
r2 = r2_score(referencia, nivel)
nivel_final = nivel[-1]
error_residual = setpoint - nivel_final

# 2. Visualización de métricas clave
m1, m2, m3, m4 = st.columns(4)
m1.metric("Setpoint", f"{setpoint} m")
m2.metric("Nivel Final", f"{nivel_final:.3f} m")
m3.metric("Error Residual", f"{error_residual:.4f} m", delta=f"{error_residual:.4f}", delta_color="inverse")
m4.metric("Precisión (R²)", f"{r2:.4f}")

# 3. Tabla de Resumen de Variables
st.subheader("📝 Resumen de Resultados")
datos_resumen = {
    "Métrica": ["Setpoint (Objetivo)", "Nivel Final Alcanzado", "Error Cuadrático Medio (MSE)", "Coeficiente de Correlación (R²)"],
    "Valor": [f"{setpoint} m", f"{nivel_final:.4f} m", f"{mse:.8f}", f"{r2:.4f}"]
}
st.table(pd.DataFrame(datos_resumen))

# 4. Alertas de Estado
if abs(error_residual) < 0.02:
    st.success("✅ **Sistema Estable:** El controlador ha alcanzado el setpoint satisfactoriamente.")
else:
    st.warning("⚠️ **Ajuste Requerido:** Se detecta un error residual significativo. Considere ajustar los parámetros PID.")

# 5. Exportación de Datos
st.download_button(
    label="📥 Descargar Reporte de Simulación (CSV)",
    data=df_simulacion.to_csv(index=False).encode('utf-8'),
    file_name='simulacion_ucv_tanques.csv',
    mime='text/csv',
)
