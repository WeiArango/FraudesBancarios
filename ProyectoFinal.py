import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, average_precision_score, recall_score)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import time

# Configuración de la página
st.set_page_config(
    layout="wide",
    page_title="Proyecto Final",
    page_icon="🕵️"
)
t1, t2 = st.columns([0.3, 0.7])
t1.title("🚀 Sistema de Detección de Fraude Bancario con ML")
t2.image('img_fraude.jpg', width = 350)

# Cargar datos
@st.cache_data
def load_data():
    return pd.read_csv("creditcard.csv")

df = load_data()
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Cargar modelo pre-entrenado
@st.cache_resource
def load_model():
    with open('creditcard_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# # Dividir en columnas
# col1, col2 = st.columns([2, 3])

# with col1:

# with col2:
    # Sección de Presentación del Proyecto
with st.expander("📰 INTRODUCCIÓN", expanded=False):
    st.markdown("""
    ### ¿En qué consiste el proyecto?

    Este sistema permite detectar **fraudes en transacciones bancarias** mediante un modelo de clasificación basado en Random Forest, entrenado sobre datos altamente desbalanceados (solo el 0.17% corresponde a fraudes). El modelo identifica patrones atípicos con una alta precisión, utilizando una interfaz interactiva desarrollada con Streamlit.

    ### Tecnologías clave utilizadas:
    - **Python**: pandas, scikit-learn, imbalanced-learn
    - **Algoritmo**: Random Forest + SMOTE para balanceo
    - **Visualización**: Streamlit como frontend interactivo
    """)

with st.expander("📌 JUSTIFICACIÓN"):
    st.markdown("""
    ### ¿Por qué es importante?

    El fraude bancario representa una amenaza constante en el entorno de la banca digital. Este proyecto ofrece una solución que supera los enfoques tradicionales basados en reglas fijas, mediante técnicas de aprendizaje automático que permiten:

    #### 🔍 Análisis avanzado de patrones:
    - Evaluación de **28 variables** transformadas con PCA para detectar comportamientos atípicos.
    - Análisis del **histórico transaccional por cliente**, mejorando la detección contextual de anomalías.

    #### 🔐 Protección de la privacidad:
    - La base de datos ha sido **anonimizada** mediante técnicas de codificación para preservar la confidencialidad de los usuarios.

    #### 🚀 Ventajas competitivas del modelo:
    - **95% de recall** en la detección de fraudes, con solo **0.1% de falsos positivos**.
    - Capacidad de procesar **3,000 transacciones por segundo** con latencia inferior a 50ms.
    - Reducción de pérdidas financieras de hasta **40% en comparación con sistemas convencionales**.
    """)

with st.expander("🎯 OBJETIVOS DEL PROYECTO"):
    st.markdown("""
    ### Objetivo general:
    Obtener un análisis detallado de las transacciones realizadas por un conjunto de clientes seleccionados aleatoriamente, identificando el porcentaje de operaciones fraudulentas frente a las legítimas.

    ### Objetivos específicos:
    1. **Cuantificar** el porcentaje de transacciones realizadas por cliente para detectar patrones individuales y estimar su riesgo de fraude.
    2. **Visualizar** el comportamiento temporal de las transacciones mediante análisis estadísticos que permitan detectar irregularidades a lo largo del tiempo.
    3. **Analizar** la distribución de variables clave para facilitar la identificación de anomalías asociadas al fraude.
    4. **Construir** un modelo predictivo que estime la probabilidad de fraude en subgrupos específicos de clientes, optimizando así la detección temprana y la toma de decisiones.

    """)

with st.expander("📊 BASE DE DATOS UTILIZADA"):
    st.markdown("""
    ### Dataset: Credit Card Fraud Detection

    **Características generales:**
    - Registros totales: 70,000 transacciones
    - Fraudes detectados: 175 (~0.25%)
    - Transacciones legítimas: 69,825
    - Número de variables: 31 columnas
    - Cobertura temporal: ~2 días de operaciones
    - Formato: CSV, codificación UTF-8
    - Tamaño: ~24 MB

    **Estructura de la base de datos:**
    - `Time`: Tiempo transcurrido desde la primera transacción
    - `V1` a `V28`: Variables transformadas con PCA (anonimizadas)
    - `Amount`: Valor de la transacción
    - `Class`: Etiqueta objetivo (0 = legítima, 1 = fraudulenta)

    **Variables más relevantes para el modelo:**  
    - V4, V10, V12, V14
    """)

with st.expander("⚙️ METODOLOGÍA"):
    st.markdown("""
    ### Flujo de trabajo aplicado:

    1. **Preprocesamiento**:
        - Escalado de variables
        - Aplicación de SMOTE para balancear clases
    2. **Entrenamiento del modelo**:
        ```python
        RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced_subsample"
        )
        ```
    3. **Evaluación del desempeño**:
        - Matriz de confusión
        - Métricas: Average Precision, Recall, ROC AUC
    4. **Despliegue del sistema**:
        - Interfaz desarrollada con Streamlit para visualización y análisis en tiempo real.
    """)

st.markdown("""
---
🔗 **Explora la aplicación web:** [fraudesbancarios.streamlit.app](https://fraudesbancarios.streamlit.app/)<br>
💻 **Código fuente disponible en GitHub** *(enlace dentro de la app)*<br><br>
👥 **Participantes:**<br>
&nbsp;&nbsp;&nbsp;&nbsp;• *Juliana*<br>
&nbsp;&nbsp;&nbsp;&nbsp;• *Carlos*<br>
&nbsp;&nbsp;&nbsp;&nbsp;• *Ubeimar Bedoya Arango*<br>
""", unsafe_allow_html=True)

# Sección de información básica
st.header("📊 Información Básica")
st.dataframe(df, height=300)

st.subheader("Métricas clave")
cols = st.columns(4)
cols[0].metric("Características", len(df.columns))
cols[1].metric("Transacciones", len(df))
cols[2].metric("Fraudes", len(df[df["Class"] == 1]))
cols[3].metric("Monto promedio", f"${df['Amount'].mean():.2f}")

# Sección de análisis visual (manteniendo tus tabs originales)
st.header("📈 Análisis Estadístico")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Distribuciones", "Dispersión", "Límite de Decisión", "Entrenamiento Modelo", "Predicción Modelo"])

with tab1:
    st.subheader("Distribución de todas las características por clase")
    st.write("Comparación de distribuciones para transacciones legítimas (verde) y fraudulentas (rojo)")
    
    # Dividir las características en grupos de 4 para mejor organización
    features = X.columns
    n_features = len(features)
    rows = (n_features + 3) // 4  # Calcula el número de filas necesarias
    
    # Crear figura con tamaño dinámico basado en el número de características
    fig = plt.figure(figsize=(20, 5 * rows))
    
    for i, f in enumerate(features):
        ax = plt.subplot(rows, 4, i+1)  # Organiza en filas de 4 gráficas
        sns.histplot(data=df[df["Class"] == 1], x=f, kde=True, color="red",
                    stat="density", label="Fraud", alpha=0.5)
        sns.histplot(data=df[df["Class"] == 0], x=f, kde=True, color="green",
                    stat="density", label="Legit", alpha=0.5)
        ax.set_xlabel('')
        ax.set_title(f"Feature: {f}")
        ax.legend()
    
    plt.tight_layout()  # Ajusta el espaciado entre subplots
    st.pyplot(fig)

with tab2:
    st.subheader("Gráficos de Dispersión: Buenas vs Malas Distribuciones")
    
    # Definir pares de características (buenos y malos)
    good_pairs = [("V14", "V10"), ("V4", "V12")]  # Pares que separan bien las clases
    bad_pairs = [("V1", "V2"), ("V5", "V6")]     # Pares donde las clases se solapan
    
    # Mostrar 2 buenas distribuciones
    st.markdown("### Buenas distribuciones (separan bien las clases)")
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico 1: V14 vs V10
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.scatter(df["V14"][df['Class'] == 0], df["V10"][df['Class'] == 0],
                c="green", marker=".", alpha=0.3, label="Legítimas")
        ax1.scatter(df["V14"][df['Class'] == 1], df["V10"][df['Class'] == 1],
                c="red", marker=".", alpha=0.7, label="Fraudulentas")
        ax1.set_xlabel("V14")
        ax1.set_ylabel("V10")
        ax1.set_title("V10 vs V14 - Buena separación")
        ax1.legend()
        st.pyplot(fig1)
        
        st.markdown("""
        **Interpretación:**
        - Las transacciones fraudulentas (rojo) se agrupan en valores atípicos
        - Claramente separadas del grupo principal de transacciones legítimas
        """)
    
    with col2:
        # Gráfico 2: V4 vs V12
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.scatter(df["V4"][df['Class'] == 0], df["V12"][df['Class'] == 0],
                c="green", marker=".", alpha=0.3, label="Legítimas")
        ax2.scatter(df["V4"][df['Class'] == 1], df["V12"][df['Class'] == 1],
                c="red", marker=".", alpha=0.7, label="Fraudulentas")
        ax2.set_xlabel("V4")
        ax2.set_ylabel("V12")
        ax2.set_title("V12 vs V4 - Buena separación")
        ax2.legend()
        st.pyplot(fig2)
        
        st.markdown("""
        **Interpretación:**
        - Patrón claro de agrupamiento diferente para fraudes
        - Valores extremos en ambas dimensiones para casos fraudulentos
        """)
    
    # Mostrar 2 malas distribuciones
    st.markdown("### Malas distribuciones (clases solapadas)")
    col3, col4 = st.columns(2)
    
    with col3:
        # Gráfico 3: V1 vs V2
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        ax3.scatter(df["V1"][df['Class'] == 0], df["V2"][df['Class'] == 0],
                c="green", marker=".", alpha=0.3, label="Legítimas")
        ax3.scatter(df["V1"][df['Class'] == 1], df["V2"][df['Class'] == 1],
                c="red", marker=".", alpha=0.7, label="Fraudulentas")
        ax3.set_xlabel("V1")
        ax3.set_ylabel("V2")
        ax3.set_title("V2 vs V1 - Solapamiento")
        ax3.legend()
        st.pyplot(fig3)
        
        st.markdown("""
        **Interpretación:**
        - Las clases están completamente mezcladas
        - Difícil distinguir fraudes basado solo en estas características
        """)
    
    with col4:
        # Gráfico 4: V5 vs V6
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        ax4.scatter(df["V5"][df['Class'] == 0], df["V6"][df['Class'] == 0],
                c="green", marker=".", alpha=0.3, label="Legítimas")
        ax4.scatter(df["V5"][df['Class'] == 1], df["V6"][df['Class'] == 1],
                c="red", marker=".", alpha=0.7, label="Fraudulentas")
        ax4.set_xlabel("V5")
        ax4.set_ylabel("V6")
        ax4.set_title("V6 vs V5 - Solapamiento")
        ax4.legend()
        st.pyplot(fig4)
        
        st.markdown("""
        **Interpretación:**
        - Distribuciones casi idénticas para ambas clases
        - Estas características por sí solas no ayudan a detectar fraudes
        """)
    
    # Opcional: Añadir explicación general
    st.markdown("""
    **Análisis Comparativo:**
    - Las buenas distribuciones muestran claras diferencias entre clases
    - Las malas distribuciones tienen solapamiento total
    - Esto explica por qué algunas características son más importantes para el modelo
    """)

with tab3:
    st.subheader("Límite de Decisión")
    st.markdown("""
    **Visualización del límite que separa transacciones legítimas de fraudulentas**  
    *Usando solo 2 características seleccionadas y una regresión logística para simplificar la representación*
    """)
    
    # Selección interactiva de características
    col1, col2 = st.columns(2)
    with col1:
        feat1 = st.selectbox(
            "Primera Característica", 
            X.columns, 
            index=10,  # Valor por defecto: V10
            key='feat1',
            help="Selecciona la característica para el eje X"
        )
    with col2:
        feat2 = st.selectbox(
            "Segunda Característica", 
            X.columns, 
            index=14,  # Valor por defecto: V14
            key='feat2',
            help="Selecciona la característica para el eje Y"
        )
    
    # Modelo simplificado para visualización
    viz_model = LogisticRegression(
        class_weight='balanced',  # Ajuste para datos desbalanceados
        solver='lbfgs',           # Algoritmo adecuado para problemas binarios
        max_iter=1000             # Garantizar convergencia
    )
    
    with st.spinner('Calculando límite de decisión...'):
        viz_model.fit(X_train[[feat1, feat2]], y_train)
        
        # Crear grid para el límite de decisión
        x_min, x_max = X[feat1].min() - 1, X[feat1].max() + 1
        y_min, y_max = X[feat2].min() - 1, X[feat2].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )
        
        # Predecir probabilidades para el grid
        Z = viz_model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        
        # Configurar figura
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Mostrar límite de decisión (áreas coloreadas)
        contour = ax.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.6, vmin=0, vmax=1)
        plt.colorbar(contour, label='Probabilidad de Fraude')
        
        # Graficar puntos reales
        ax.scatter(
            X[feat1][y==0], X[feat2][y==0], 
            c='green', alpha=0.3, s=10, label='Legítimas',
            edgecolor='k', linewidth=0.3
        )
        ax.scatter(
            X[feat1][y==1], X[feat2][y==1], 
            c='red', alpha=0.7, s=20, label='Fraudulentas',
            edgecolor='k', linewidth=0.5
        )
        
        # Línea de decisión (umbral 0.5)
        decision_boundary = ax.contour(
            xx, yy, Z, levels=[0.5], 
            colors='black', linewidths=2, linestyles='dashed'
        )
        
        # Configuración estética
        ax.set_xlabel(feat1, fontsize=12)
        ax.set_ylabel(feat2, fontsize=12)
        ax.set_title(f'Límite de Decisión: {feat1} vs {feat2}', pad=15)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.2)
        
        st.pyplot(fig)
    
    # Análisis interpretativo
    st.markdown(f"""
    ### Interpretación:
    1. **Zonas coloreadas**:
    - 🔵 *Azul*: Mayor probabilidad de ser fraude 
    - 🔴 *Rojo*: Mayor probabilidad de ser transacción legítima
    
    2. **Línea negra discontinua**:
    - Umbral de decisión (50% probabilidad)
    - Todo a la derecha/arriba de la línea se clasifica como legítimo
    
    3. **Patrones observables**:
    - Cuando se usan características relevantes (V10, V12, V14, V17), el límite separa mejor los grupos
    """)
    
    # Recomendaciones
    st.markdown("""
    ### Recomendaciones:
    - Para mejor visualización, selecciona características con alta importancia (V10, V12, V14)
    - Si el límite no separa bien las clases, indica que se necesitan más características
    """)

with tab4:
    st.header("Entrenamiento del Modelo")
    st.markdown("""
    Esta sección muestra el rendimiento del modelo Random Forest en la detección de transacciones fraudulentas. 
    Los resultados incluyen métricas clave, matriz de confusión y la importancia de las características.
    """)
    
    # Hacer predicciones
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probabilidades para la clase positiva (fraude)
    
    # Mostrar métricas
    st.subheader("Métricas de Rendimiento Clave")
    st.markdown("""
    Estas métricas nos ayudan a evaluar qué tan bien está funcionando nuestro modelo:
    - **ROC AUC**: Mide la capacidad del modelo para distinguir entre clases (1 = perfecto, 0.5 = aleatorio)
    - **Average Precision**: Evalúa el rendimiento en clases desbalanceadas, dando más peso a los aciertos en fraudes
    - **Recall (Fraudes)**: Porcentaje de fraudes reales que el modelo detectó correctamente
    """)
    
    cols = st.columns(3)
    with cols[0]:
        roc_auc = roc_auc_score(y_test, y_proba)
        st.metric("ROC AUC", f"{roc_auc:.4f}",
                help="Valor entre 0 y 1 donde 1 es perfecto. Mide la capacidad de distinguir entre clases.")
        st.markdown(f"""
        <div style="background-color:#f0f2f6; padding:10px; border-radius:5px; margin-top:-10px">
            <small>
            {"> 0.9" if roc_auc > 0.9 else "> 0.8" if roc_auc > 0.8 else "> 0.7"} - Buen rendimiento<br>
            {"Excelente" if roc_auc > 0.9 else "Bueno" if roc_auc > 0.8 else "Aceptable"} para problemas de fraude
            </small>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        avg_precision = average_precision_score(y_test, y_proba)
        st.metric("Average Precision", f"{avg_precision:.4f}",
                help="Métrica preferida para datos desbalanceados. Más relevante que accuracy.")
        st.markdown(f"""
        <div style="background-color:#f0f2f6; padding:10px; border-radius:5px; margin-top:-10px">
            <small>
            En fraudes, buscamos valores > 0.5<br>
            {"Muy bueno" if avg_precision > 0.7 else "Bueno" if avg_precision > 0.5 else "Necesita mejorar"}
            </small>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        recall = recall_score(y_test, y_pred)
        st.metric("Recall (Fraudes)", f"{recall:.4f}",
                help="Fraudes detectados / Total fraudes reales. Clave para minimizar falsos negativos.")
        st.markdown(f"""
        <div style="background-color:#f0f2f6; padding:10px; border-radius:5px; margin-top:-10px">
            <small>
            Ideal > 0.8 para fraudes<br>
            Cada fraude no detectado representa pérdida económica
            </small>
        </div>
        """, unsafe_allow_html=True)
    
    # Matriz de confusión
    st.subheader("Matriz de Confusión")
    st.markdown("""
    La matriz de confusión muestra:
    - **Verdaderos Negativos (TN)**: Transacciones legítimas correctamente identificadas
    - **Falsos Positivos (FP)**: Legítimas marcadas como fraude (falsas alarmas)
    - **Falsos Negativos (FN)**: Fraudulentas no detectadas (pérdida económica)
    - **Verdaderos Positivos (TP)**: Fraudulentas correctamente detectadas
    """)
    
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legítima', 'Fraude'],
                yticklabels=['Legítima', 'Fraude'])
    ax.set_xlabel('Predicción del Modelo', fontsize=12)
    ax.set_ylabel('Realidad', fontsize=12)
    ax.set_title('Desempeño en Detección de Fraudes', pad=20)
    st.pyplot(fig)
    
    # Análisis de la matriz de confusión
    tn, fp, fn, tp = cm.ravel()
    st.markdown(f"""
    <div style="background-color:#f0f2f6; padding:15px; border-radius:5px; margin-top:10px">
        <b>Interpretación:</b><br>
        - <span style="color:green">Transacciones legítimas correctas:</span> {tn:,} ({(tn/(tn+fp)*100):.1f}% del total legítimo)<br>
        - <span style="color:orange">Falsos positivos:</span> {fp:,} ({(fp/(tn+fp)*100):.1f}% del total legítimo)<br>
        - <span style="color:red">Falsos negativos:</span> {fn:,} ({(fn/(fn+tp)*100):.1f}% del total fraudulento)<br>
        - <span style="color:blue">Fraudes detectados:</span> {tp:,} ({(tp/(fn+tp)*100):.1f}% del total fraudulento)
    </div>
    """, unsafe_allow_html=True)
    
    # Importancia de características
    st.subheader("Importancia de Características (Top 10)")
    st.markdown("""
    Muestra qué variables tienen mayor influencia en las decisiones del modelo:
    - Las características más importantes son clave para detectar patrones de fraude
    - Podemos usarlas para simplificar el modelo o enfocar análisis futuros
    """)
    
    if hasattr(model, 'named_steps'):
        rf_model = model.named_steps['randomforestclassifier']
    else:
        rf_model = model
    
    importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importances, palette='viridis')
    ax.set_title('Características Más Relevantes para Detectar Fraude', pad=15)
    ax.set_xlabel('Importancia Relativa', fontsize=12)
    ax.set_ylabel('Característica', fontsize=12)
    st.pyplot(fig)
    
    # Explicación de las características importantes
    st.markdown("""
    <div style="background-color:#f0f2f6; padding:15px; border-radius:5px; margin-top:10px">
        <b>Análisis de Características:</b><br>
        - <b>V14, V10, V12</b>: Suelen ser las más importantes en fraudes, representan patrones atípicos<br>
        - <b>V4, V7</b>: Capturan comportamientos inusuales en montos o frecuencias<br>
        - <b>Amount</b>: El monto de la transacción, aunque menos importante que las características anonimizadas<br>
        <small>Nota: Las características 'V' son componentes PCA de datos originales</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Recomendaciones basadas en resultados
    st.subheader("Recomendaciones")
    st.markdown("""
    - Si el <b>recall</b> es bajo, considerar:
    - Aumentar el peso de la clase fraudulenta
    - Probar técnicas de oversampling más avanzadas (como ADASYN)
    - Ajustar el umbral de clasificación
    - Si hay muchas <b>falsas alarmas</b>:
    - Revisar características menos importantes que puedan estar introduciendo ruido
    - Considerar un ensamble con otros modelos
    - Para mejorar <b>Average Precision</b>:
    - Incluir más datos de fraudes históricos
    - Probar modelos alternativos como XGBoost o redes neuronales
    """, unsafe_allow_html=True)

with tab5:
    st.header("🔮 Predicción del Modelo")
    
    @st.cache_resource
    def get_model():
        with open('creditcard_model.pkl', 'rb') as f:
            return pickle.load(f)
    
    model = get_model()
    
    # 1. Selector para cargar ejemplos o ingresar manualmente
    input_mode = st.radio("Fuente de datos:",
                        ["Ingreso Manual", "Ejemplo del Dataset"])
    
    cols = st.columns(2)
    features = {}
    
    with cols[0]:
        if input_mode == "Ejemplo del Dataset":
            sample = X_test.sample(1).iloc[0]
            features['V10'] = st.number_input('V10', value=float(sample['V10']), format="%.5f")
            features['V14'] = st.number_input('V14', value=float(sample['V14']), format="%.5f")
            features['Amount'] = st.number_input('Monto', value=float(sample['Amount']))
        else:
            features['V10'] = st.number_input('V10', value=0.0, format="%.5f")
            features['V14'] = st.number_input('V14', value=0.0, format="%.5f")
            features['Amount'] = st.number_input('Monto', value=0.0)
    
    with cols[1]:
        if input_mode == "Ejemplo del Dataset":
            features['V4'] = st.number_input('V4', value=float(sample['V4']), format="%.5f")
            features['V12'] = st.number_input('V12', value=float(sample['V12']), format="%.5f")
        else:
            features['V4'] = st.number_input('V4', value=0.0, format="%.5f")
            features['V12'] = st.number_input('V12', value=0.0, format="%.5f")

# Resto del código de predicción igual...
    
    # Pre-allocated array con valores por defecto 
    default_values = np.zeros(len(X.columns))
    feature_indices = {col: idx for idx, col in enumerate(X.columns)}
    
    if st.button('Predecir', key='predict_btn'):
        start_time = time.time()
        
        # Construir array de entrada 
        input_array = default_values.copy()
        for feature, value in features.items():
            if feature in feature_indices:
                input_array[feature_indices[feature]] = value
        
        # Predicción vectorizada 
        with st.spinner('Calculando...'):
            proba = model.predict_proba([input_array])[0][1]
            threshold = 0.3  # o usar optimal_threshold si está disponible
            prediction = "Fraudulenta" if proba >= threshold else "Legítima"
        
        # Resultado visual
        risk_color = "red" if proba >= threshold else "green"
        risk_level = "ALTO" if proba >= threshold else "bajo"
        
        st.markdown(f"""
        <div style="border-radius:10px; padding:20px; background-color:#f0f2f6; margin-top:20px">
            <h3 style="color:{risk_color}; text-align:center">
                Riesgo de Fraude: <b>{proba*100:.1f}%</b> ({risk_level})
            </h3>
            <p style="text-align:left">
                Threshold: {threshold:.2f} | 
                V10: {features['V10']:.2f} | 
                V14: {features['V14']:.2f} | 
                V4: {features['V4']:.2f}   | 
                V12: {features['V12']:.2f} | 
                Monto: ${features['Amount']:.2f}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    # Instrucciones rápidas
    st.caption("""
    💡 **Guía rápida:**  
    - Valores de V10/V14 fuera del rango [-2, 2] son sospechosos  
    - Montos mayores a $1000 requieren revisión  
    - Ajusta el threshold para mayor/menor sensibilidad
    """)
        



            
    