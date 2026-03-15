import numpy as np
import IPython.display as display
from matplotlib import pyplot as plt
import io
import base64

ys = 200 + np.random.randn(100)
x = [x for x in range(len(ys))]

fig = plt.figure(figsize=(4, 3), facecolor='w')
plt.plot(x, ys, '-')
plt.fill_between(x, ys, 195, where=(ys > 195), facecolor='g', alpha=0.6)
plt.title("Sample Visualization", fontsize=10)

data = io.BytesIO()
plt.savefig(data)
image = F"data:image/png;base64,{base64.b64encode(data.getvalue()).decode()}"
alt = "Sample Visualization"
display.display(display.Markdown(F"""![{alt}]({image})"""))
plt.close(fig)


import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Configuración de la página
st.set_page_config(page_title="Simulador LOU - UCV", layout="wide")

st.title("Práctica Virtual: Laboratorio de Operaciones Unitarias - UCV")
st.markdown("---")

# --- BARRA LATERAL (ENTRADAS) ---
st.sidebar.header("Parámetros del Sistema")

# Dimensiones reales del tanque (Tabla 4 y 5)
diametro = st.sidebar.number_input("Diámetro del Tanque (m)", value=0.503, format="%.3f")
area_tanque = np.pi * (diametro / 2)**2

modo = st.sidebar.radio("Seleccione el proceso:", ["Vaciado", "Llenado"])

if modo == "Vaciado":
    st.sidebar.subheader("Configuración de Vaciado")
    h0 = st.sidebar.slider("Nivel Inicial (m)", 0.0, 0.40, 0.35)
    # Constante K identificada en tus pruebas de MATLAB
    k_vaciado = st.sidebar.number_input("Constante K de Salida", value=0.000295, format="%.6f")
    t_final = st.sidebar.number_input("Tiempo de simulación (s)", value=1000)

else:
    st.sidebar.subheader("Configuración de Llenado")
    # Entrada según la plomada del rotámetro
    altura_plomada = st.sidebar.number_input("Altura de Plomada (cm)", value=4.0, step=0.5)

    # Ecuación de calibración: y = 39.739x + 106.9 (Caudal en L/h)
    q_lh = 39.739 * altura_plomada + 106.9
    q_alim = (q_lh / 1000) / 3600  # Conversión a m3/s

    st.sidebar.info(f"Caudal calculado: {q_lh:.2f} L/h")

# --- LÓGICA DE SIMULACIÓN ---
if st.button("Ejecutar Simulación"):
    t_eval = np.linspace(0, 1000, 100)

    if modo == "Vaciado":
        # EDO: dh/dt = -K*sqrt(h) / Area
        def edo_vaciado(t, h):
            return -k_vaciado * np.sqrt(max(0, h)) / area_tanque

        sol = solve_ivp(edo_vaciado, [0, t_final], [h0], t_eval=t_eval)
        t_plot, h_plot = sol.t, sol.y[0]

    else:
        # Modelo de llenado: h(t) = h0 + (Qalim/Area)*t
        h_plot = 0 + (q_alim / area_tanque) * t_eval
        t_plot = t_eval

    # --- GRÁFICA ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_plot, h_plot, 'b-', linewidth=2, label=f"Modelo de {modo}")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Nivel (m)")
    ax.set_title(f"Dinámica del Tanque - Proceso de {modo}")
    ax.grid(True, linestyle='--')
    ax.legend()

    st.pyplot(fig)

    # Mostrar datos en tabla
    df_resultados = {"Tiempo (s)": t_plot, "Nivel (m)": h_plot}
    st.dataframe(df_resultados)
