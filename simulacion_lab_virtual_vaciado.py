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

"""Los notebooks de Colab ejecutan código en los servidores alojados en la nube de Google, lo que significa que puedes aprovechar al máximo el hardware de Google, incluidas las <a href="#using-accelerated-hardware">GPU y TPU</a>, independientemente de la potencia de tu máquina. Lo único que necesitas es un navegador.

Por ejemplo, si estás esperando que el código de <strong>pandas</strong> termine de ejecutarse y quieres ir más rápido, puedes cambiar a un entorno de ejecución de GPU y usar bibliotecas como <a href="https://rapids.ai/cudf-pandas">RAPIDS cuDF</a> que proporcionan aceleración sin cambios de código.

Para obtener más información sobre la aceleración de pandas en Colab, consulta la <a href="https://colab.research.google.com/github/rapidsai-community/showcase/blob/main/getting_started_tutorials/cudf_pandas_colab_demo.ipynb">guía de 10 minutos</a> o
 la <a href="https://colab.research.google.com/github/rapidsai-community/showcase/blob/main/getting_started_tutorials/cudf_pandas_stocks_demo.ipynb">demostración sobre el análisis de datos del mercado de valores de EE.UU.</a>.

<div class="markdown-google-sans">

## Aprendizaje automático
</div>

Colab te permite importar un conjunto de datos de imágenes, entrenar un clasificador de imágenes en él y evaluar el modelo con solo <a href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/quickstart/beginner.ipynb">unas pocas líneas de código</a>.

Entre los usos que se la da a Colab en la comunidad de aprendizaje automático, se encuentran los siguientes:
- Introducción a TensorFlow
- Desarrollo y entrenamiento de redes neuronales
- Experimentación con TPU
- Diseminación de investigación de IA
- Creación de instructivos

Para ver notebooks de Colab de ejemplo que muestran los usos del aprendizaje automático, consulta los <a href="#machine-learning-examples">ejemplos</a> que se incluyen a continuación.

<div class="markdown-google-sans">

## Más recursos

### Cómo trabajar con notebooks en Colab

</div>

- [Descripción general de Colab](/notebooks/basic_features_overview.ipynb)
- [Guía para usar Markdown](/notebooks/markdown_guide.ipynb)
- [Cómo importar bibliotecas y luego instalar dependencias](/notebooks/snippets/importing_libraries.ipynb)
- [Cómo guardar y cargar notebooks en GitHub](https://colab.research.google.com/github/googlecolab/colabtools/blob/main/notebooks/colab-github-demo.ipynb)
- [Formularios interactivos](/notebooks/forms.ipynb)
- [Widgets interactivos](/notebooks/widgets.ipynb)

<div class="markdown-google-sans">

<a name="working-with-data"></a>
### Cómo trabajar con datos
</div>

- [Cómo cargar datos: Drive, Hojas de cálculo y Google Cloud Storage](/notebooks/io.ipynb)
- [Gráficos: visualización de datos](/notebooks/charts.ipynb)
- [Cómo comenzar a usar BigQuery](/notebooks/bigquery.ipynb)

<div class="markdown-google-sans">

### Aprendizaje automático

<div>

Estos son algunos de los notebooks relacionados con el aprendizaje automático; incluido el curso de aprendizaje automático en línea de Google. Para obtener más información, consulta el <a href="https://developers.google.com/machine-learning/crash-course/">sitio web del curso completo</a>.
- [Introducción a Pandas DataFrame](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/pandas_dataframe_ultraquick_tutorial.ipynb)
- [Introducción a RAPIDS cuDF para acelerar Pandas](https://nvda.ws/rapids-cudf)
- [Comenzar a usar el modo de acelerador de cuML](https://colab.research.google.com/github/rapidsai-community/showcase/blob/main/getting_started_tutorials/cuml_sklearn_colab_demo.ipynb)

<div class="markdown-google-sans">

<a name="using-accelerated-hardware"></a>
### Uso de aceleración de hardware
</div>

- [Entrena una CNN para clasificar dígitos escritos a mano en el conjunto de datos MNIST con la API de Flax NNX](https://colab.research.google.com/github/google/flax/blob/main/docs_nnx/mnist_tutorial.ipynb)
- [Entrena un Vision Transformer &#40;ViT&#41; para la clasificación de imágenes con JAX](https://colab.research.google.com/github/jax-ml/jax-ai-stack/blob/main/docs/source/JAX_Vision_transformer.ipynb)
- [Clasificación de texto con un modelo de lenguaje de Transformer con JAX](https://colab.research.google.com/github/jax-ml/jax-ai-stack/blob/main/docs/source/JAX_transformer_text_classification.ipynb)

<div class="markdown-google-sans">

<a name="machine-learning-examples"></a>

### Ejemplos destacados

</div>

- <a href="https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html">Entrena un modelo de lenguaje miniGPT con JAX AI Stack</a>
- <a href="https://github.com/google/tunix/blob/main/examples/qlora_gemma.ipynb">Ajuste de LoRA o QLoRA para LLM con Tunix</a>
- <a href="https://keras.io/examples/keras_recipes/parameter_efficient_finetuning_of_gemma_with_lora_and_qlora/">Ajuste eficiente de parámetros de Gemma con LoRA y QLoRA</a>
- <a href="https://keras.io/keras_hub/guides/hugging_face_keras_integration/">Cargando puntos de control de Transformers de Hugging Face</a>
- <a href="https://keras.io/guides/int8_quantization_in_keras/">Cuantización de números enteros de 8 bits en Keras</a>
- <a href="https://keras.io/examples/keras_recipes/float8_training_and_inference_with_transformer/">Entrenamiento e inferencia de Float8 con un modelo de Transformer simple</a>
- <a href="https://keras.io/keras_hub/guides/transformer_pretraining/">Entrenamiento previo de un Transformer desde cero con KerasHub</a>
- <a href="https://keras.io/examples/vision/mnist_convnet/">Red convolucional simple basada en MNIST</a>
- <a href="https://keras.io/examples/vision/image_classification_from_scratch/">Clasificación de imágenes desde cero con Keras 3</a>
- <a href="https://keras.io/keras_hub/guides/classification_with_keras_hub/">Clasificación de imágenes con KerasHub</a>
"""

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
