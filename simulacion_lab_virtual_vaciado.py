import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Configuración de página
st.set_page_config(page_title="Laboratorio Virtual UCV", layout="wide")

# Título actualizado
st.title("🧪 Laboratorio Virtual: Práctica de Vaciado de Tanques")
st.markdown("### Escuela de Ingeniería Química - UCV")

# --- SECCIÓN TEÓRICA ---
with st.expander("📚 Fundamentos: Geometría y Balance de Masa"):
    st.write("En esta práctica, estudiamos cómo la forma del tanque afecta la dinámica del nivel.")
    st.latex(r"\frac{dh}{dt} = \frac{Q_{in} - Q_{out}}{A(h)}")
    st.info("Nota: En tanques no cilíndricos, el área $A(h)$ no es constante, lo que genera un sistema no lineal.")

# --- BARRA LATERAL ---
st.sidebar.header("⚙️ Configuración del Sistema")
tipo_tanque = st.sidebar.selectbox("Seleccione la Geometría", 
                                   ["Cilíndrico", "Troncocónico", "Esférico"])

r_base = st.sidebar.number_input("Radio base/esfera (m)", value=0.25, min_value=0.01)

# Parámetros según geometría
if tipo_tanque == "Troncocónico":
    r_tope = st.sidebar.number_input("Radio tope (m)", value=0.45, min_value=0.01)
    H_total = st.sidebar.number_input("Altura total (m)", value=1.0, min_value=0.1)
elif tipo_tanque == "Esférico":
    H_total = 2 * r_base
else:
    H_total = 1.0

st.sidebar.header("🎮 Parámetros de Control (PID)")
sp = st.sidebar.slider("Setpoint (m)", 0.0, float(H_total), 0.6)
kp = st.sidebar.number_input("Kp", value=0.08, format="%.4f")
ti = st.sidebar.number_input("Ti", value=25.0)
td = st.sidebar.number_input("Td", value=2.0)

# --- SIMULACIÓN ---
if st.button("🚀 Iniciar Simulación"):
    
    def obtener_area(h):
        if tipo_tanque == "Cilíndrico":
            return np.pi * r_base**2
        elif tipo_tanque == "Troncocónico":
            radio_h = r_base + (r_tope - r_base) * (h / H_total)
            return np.pi * radio_h**2
        elif tipo_tanque == "Esférico":
            return np.pi * (2 * r_base * h - h**2) if h > 0 else 1e-5

    # Parámetros temporales
    t_final, t_steps = 800, 400
    t_eval = np.linspace(0, t_final, t_steps)
    dt = t_final / t_steps
    
    h_actual = 0.2
    hist_h, hist_e, hist_u = [], [], []
    err_acum, u_prev = 0, 0

    for t in t_eval:
        error = sp - h_actual
        err_acum += error * dt
        derivada = (error - u_prev) / dt
        
        # Acción PID
        u = kp * (error + (1/ti) * err_acum + td * derivada)
        u = np.clip(u, 0, 0.01)
        
        q_out = u * np.sqrt(2 * 9.81 * h_actual) if h_actual > 0 else 0
        area = obtener_area(h_actual)
        
        dhdt = (0.0002 - q_out) / area
        h_actual += dhdt * dt
        h_actual = max(0, min(h_actual, H_total))
        
        hist_h.append(h_actual)
        hist_e.append(error)
        hist_u.append(u)
        u_prev = error

    # --- RESULTADOS ---
    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t_eval, hist_h, label="Nivel (PV)", color="#1f77b4", lw=2)
        ax.axhline(sp, color="red", ls="--", label="Setpoint")
        ax.set_title(f"Respuesta del Sistema - Tanque {tipo_tanque}")
        ax.set_ylabel("Altura (m)")
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("📊 Análisis de Desempeño")
        
        # Cálculo de métricas
        mse = np.mean(np.square(hist_e))
        error_final = hist_e[-1]
        
        # Tabla comparativa de métricas
        metricas = {
            "Variable": ["Setpoint", "Nivel Final", "Error Residual", "MSE (Precisión)"],
            "Valor": [f"{sp} m", f"{hist_h[-1]:.4f} m", f"{error_final:.4f} m", f"{mse:.6f}"]
        }
        st.table(pd.DataFrame(metricas))

        if mse < 0.001:
            st.success("🎯 Control de alta precisión")
        else:
            st.warning("⚠️ Se sugiere re-sintonizar el PID")

    # Botón para descargar resultados
    df_descarga = pd.DataFrame({"Tiempo(s)": t_eval, "Nivel(m)": hist_h, "Error(m)": hist_e})
    st.download_button("📥 Descargar CSV para el informe", df_descarga.to_csv(index=False), "datos_ucv.csv")

st.markdown("---")
st.caption("Unidad de Operaciones Unitarias - EIQ UCV")
