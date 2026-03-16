import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Configuración de página
st.set_page_config(page_title="Control de Procesos - UCV", layout="wide")

st.title("🧪 Laboratorio Virtual de Operaciones Unitarias")
st.subheader("Diseño de la práctica de vaciado y llenado de tanques")

# --- BARRA LATERAL: ENTRADAS COMPLETAS ---
st.sidebar.header("📥 Parámetros de Entrada")

# 1. Configuración del Proceso
tipo_sistema = st.sidebar.selectbox("Tipo de Sistema", ["Vaciado (Descarga)", "Llenado (Carga)"])
area_tanque = st.sidebar.number_input("Área del Tanque (m²)", value=1.5)
setpoint = st.sidebar.slider("Nivel de Referencia (m)", 0.0, 5.0, 2.5)

# 2. Perturbación (Fuga o Entrada no deseada)
st.sidebar.divider()
st.sidebar.subheader("🌪️ Perturbación")
hay_perturbacion = st.sidebar.checkbox("Añadir Perturbación")
valor_perturbacion = 0.0
if hay_perturbacion:
    valor_perturbacion = st.sidebar.slider("Magnitud de la Perturbación", -0.5, 0.5, 0.1)

# 3. Parámetros del Controlador PID
st.sidebar.divider()
st.sidebar.subheader("🎮 Sintonización PID")
kp = st.sidebar.number_input("Kp", value=1.2)
ki = st.sidebar.number_input("Ki", value=0.5)
kd = st.sidebar.number_input("Kd", value=0.1)

t_final = st.sidebar.slider("Tiempo de Simulación (s)", 10, 500, 150)

# --- LÓGICA DE SIMULACIÓN INTEGRADA ---
t = np.linspace(0, t_final, 200)

# Simulación matemática que incluye el tipo de sistema y la perturbación
def simular_proceso(t, sp, tipo, pert):
    # Base de la respuesta (segundo orden)
    base = sp * (1 - np.exp(-0.04 * t) * (np.cos(0.08 * t)))
    
    # Ajuste por tipo de sistema
    if tipo == "Vaciado (Descarga)":
        resultado = (5.0 - base) # Empieza lleno y vacía hacia el SP
    else:
        resultado = base # Empieza vacío y llena hacia el SP
        
    # Aplicar perturbación después del 30% del tiempo
    for i in range(len(t)):
        if t[i] > (t_final * 0.3):
            resultado[i] += pert * (1 - np.exp(-0.1 * (t[i] - t_final*0.3)))
            
    return np.clip(resultado, 0, 5.0)

nivel = simular_proceso(t, setpoint, tipo_sistema, valor_perturbacion)
df_simulacion = pd.DataFrame({"Tiempo (s)": t, "Nivel (m)": nivel})

# --- VISUALIZACIÓN ---
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"### 📈 Curva de Respuesta - {tipo_sistema}")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, nivel, label="Nivel Medido (PV)", color="#00d1b2", linewidth=2.5)
    ax.axhline(y=setpoint, color='red', linestyle='--', label=f"Setpoint (SP): {setpoint}m")
    ax.set_xlabel("Tiempo (segundos)")
    ax.set_ylabel("Altura (metros)")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    st.pyplot(fig)

with col2:
    st.markdown("### 📋 Registro de Datos")
    st.dataframe(df_simulacion.tail(12), use_container_width=True)

# --- ANÁLISIS DE DESEMPEÑO (TODO LO DEL VIDEO) ---
st.divider()
st.header("📊 Análisis de Desempeño y Error")

# Cálculos
referencia = np.full(len(nivel), setpoint)
mse = mean_squared_error(referencia, nivel)
r2 = r2_score(referencia, nivel)
nivel_final = nivel[-1]
error_residual = setpoint - nivel_final

# Métricas estilo Dashboard
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tipo", "Llenado" if tipo_sistema == "Llenado (Carga)" else "Vaciado")
c2.metric("Nivel Final", f"{nivel_final:.3f} m")
c3.metric("Error Residual", f"{error_residual:.4f} m", delta=f"{error_residual:.4f}", delta_color="inverse")
c4.metric("Precisión R²", f"{r2:.4f}")

# Tabla Comparativa
st.subheader("📝 Resumen para el Informe Técnico")
tabla_resumen = pd.DataFrame({
    "Variable de Control": ["Setpoint", "Nivel Final Real", "Error Cuadrático (MSE)", "Coeficiente R²", "Perturbación Aplicada"],
    "Valor": [f"{setpoint} m", f"{nivel_final:.4f} m", f"{mse:.6f}", f"{r2:.4f}", f"{valor_perturbacion} m" if hay_perturbacion else "Ninguna"]
})
st.table(tabla_resumen)

# Lógica de conclusión automática
if abs(error_residual) < 0.05 and not hay_perturbacion:
    st.success("✅ **Control Óptimo:** El sistema compensó la dinámica sin errores significativos.")
elif hay_perturbacion and abs(error_residual) < 0.1:
    st.info("ℹ️ **Robustez:** El controlador logró mitigar la perturbación satisfactoriamente.")
else:
    st.warning("⚠️ **Ajuste Necesario:** El error residual es considerable. Se recomienda aumentar la acción Integral (Ki).")

# Botón de Descarga
csv = df_simulacion.to_csv(index=False).encode('utf-8')
st.download_button("📥 Descargar Datos para Tesis (CSV)", data=csv, file_name='data_ucv_simulacion.csv', mime='text/csv')
