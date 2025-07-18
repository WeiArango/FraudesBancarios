import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

t1, t2 = st.columns([0.3, 0.7])
t1.image('foto.jpeg', width = 300)
t2.title('Censo Porcino Colombia')
#t2.markdown('## **Audífonos Bluetooth** \n### **Fidelidad de sonido**')

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Leer CSV
df = pd.read_csv("archivos/Tendencia_Crecimiento_Porcicultura.csv")
df.columns = df.columns.str.strip()

st.markdown("## **Datos Por Departamento**")
st.dataframe(df.set_index("DEPARTAMENTO"))

# 🎨 Controles en la barra lateral
st.sidebar.header("🎨 Personalización de fuente")
fontsize = st.sidebar.slider("Tamaño de fuente", min_value=12, max_value=24, value=18)
fontfamily = st.sidebar.selectbox("Familia de fuente", ["sans-serif", "serif", "monospace", "Arial", "Times New Roman"])
fontweight = st.sidebar.selectbox("Grosor del texto", ["normal", "bold", "light", "medium", "heavy"])

st.sidebar.header("📏 Tamaño de la gráfica")
fig_width = st.sidebar.slider("Ancho (pulgadas)", min_value=6, max_value=20, value=16)
fig_height = st.sidebar.slider("Alto (pulgadas)", min_value=4, max_value=15, value=10)

# Actualizar estilo global de matplotlib
plt.rcParams.update({
    'font.size': fontsize,
    'font.family': fontfamily,
    'axes.titlesize': fontsize + 4,
    'axes.labelsize': fontsize + 2,
    'xtick.labelsize': fontsize,
    'ytick.labelsize': fontsize,
    'legend.fontsize': fontsize
})

# Menú de selección de departamento
st.markdown("## **Selecciona un Departamento 🎨**")
departamento = st.selectbox("", df["DEPARTAMENTO"].unique())

def graficar_tendencia_lineal(df, departamento, width, height):
    datos = df[df['DEPARTAMENTO'] == departamento].squeeze()
    anios = ['2016', '2017', '2019', '2022', '2025']
    x = np.arange(len(anios))

    total_cerdos = [datos[f'Total_Cerdos_{a}'] for a in anios]
    total_predios = [datos[f'Total_Predios_Porcinos_{a}'] for a in anios]

    fig, ax1 = plt.subplots(figsize=(width, height))

    # Gráfica de línea para cerdos
    ax1.plot(x, total_cerdos, marker='o', color='steelblue', label='Total Cerdos', linewidth=2)
    ax1.set_xlabel('Año', fontsize=fontsize, fontweight=fontweight)
    ax1.set_ylabel('Total Cerdos', color='steelblue', fontsize=fontsize, fontweight=fontweight)
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(anios, fontsize=fontsize, rotation=45, ha='right')
    
    # Gráfica de línea para predios
    ax2 = ax1.twinx()
    ax2.plot(x, total_predios, marker='o', color='darkorange', label='Total Predios Porcinos', linewidth=2)
    ax2.set_ylabel('Total Predios Porcinos', color='darkorange', fontsize=fontsize, fontweight=fontweight)
    ax2.tick_params(axis='y', labelcolor='darkorange')

    # Anotaciones
    for i, val in enumerate(total_cerdos):
        ax1.annotate(f'{val:,}', xy=(x[i], val), xytext=(0, 15), textcoords='offset points',
                     ha='center', fontsize=fontsize - 2, color='steelblue', fontweight=fontweight)

    for i, val in enumerate(total_predios):
        ax2.annotate(f'{val:,}', xy=(x[i], val), xytext=(0, 5), textcoords='offset points',
                     ha='center', fontsize=fontsize - 2, color='darkorange', fontweight=fontweight)

    plt.title(f'Tendencia Lineal de Cerdos y Predios en {departamento}',
              fontsize=fontsize + 4, fontweight=fontweight, fontfamily=fontfamily)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    fig.text(0.5, -0.1, 'Fuente: ICA', ha='center', fontsize=fontsize - 1, fontstyle='italic')
    
    return fig

def graficar_barras_comparativas(df, departamento, width, height):
    datos = df[df['DEPARTAMENTO'] == departamento].squeeze()
    anios = ['2016', '2017', '2019', '2022', '2025']
    x = np.arange(len(anios))
    bar_width = 0.35

    total_cerdos = [datos[f'Total_Cerdos_{a}'] for a in anios]
    total_predios = [datos[f'Total_Predios_Porcinos_{a}'] for a in anios]

    fig, ax = plt.subplots(figsize=(width, height))

    # Barras para cerdos
    bars1 = ax.bar(x - bar_width/2, total_cerdos, bar_width, 
                   color='steelblue', label='Total Cerdos')
    
    # Barras para predios
    bars2 = ax.bar(x + bar_width/2, total_predios, bar_width, 
                   color='darkorange', label='Total Predios Porcinos')

    ax.set_xlabel('Año', fontsize=fontsize, fontweight=fontweight)
    ax.set_ylabel('Cantidad', fontsize=fontsize, fontweight=fontweight)
    ax.set_xticks(x)
    ax.set_xticklabels(anios, fontsize=fontsize, rotation=45, ha='right')
    ax.legend(fontsize=fontsize)

    # Añadir etiquetas a las barras
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:,}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=fontsize - 2)

    autolabel(bars1)
    autolabel(bars2)

    plt.title(f'Comparación de Cerdos y Predios en {departamento}',
              fontsize=fontsize + 4, fontweight=fontweight, fontfamily=fontfamily)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    fig.text(0.5, -0.1, 'Fuente: ICA', ha='center', fontsize=fontsize - 1, fontstyle='italic')
    
    return fig

# Crear pestañas para las gráficas
tab1, tab2 = st.tabs(["📈 Gráfica Lineal", "📊 Gráfica de Barras"])

with tab1:
    st.pyplot(graficar_tendencia_lineal(df, departamento, fig_width, fig_height))

with tab2:
    st.pyplot(graficar_barras_comparativas(df, departamento, fig_width, fig_height))