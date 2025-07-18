# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 10:02:41 2025

@author: jahop
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Portafolio Profesional IA", layout="wide")
st.title("📘 Portafolio Interactivo - Aplicaciones de IA en Procesos Organizacionales")

# Información del autor
st.markdown("""
**Autor:** Javier Horacio Pérez Ricárdez  
**Teléfono:** +52 561056 4095  
**Empresa:** La Corona Jabones
""")

# Tabs principales
tabs = st.tabs([
    "🏭 Simulación de Producción",
    "📈 Análisis Predictivo",
    "📊 Visualización de Datos",
    "📤 Automatización de Reportes",
    "🧪 Control de Calidad"
])

# -------------------- 1. Simulación de Producción --------------------
with tabs[0]:
    st.header("🏭 1. Simulación de una Línea de Producción")

    st.markdown("Ajusta los parámetros de la producción para simular la eficiencia de una línea de ensamble.")

    num_maquinas = st.slider("Número de máquinas en operación", 1, 10, 5)
    tasa_fallo = st.slider("Probabilidad de falla por máquina (%)", 0, 50, 10)
    tiempo_produccion = st.slider("Tiempo promedio de producción por unidad (segundos)", 1, 30, 5)
    unidades = st.number_input("Unidades a producir", min_value=1, value=100)

    if st.button("Simular producción"):
        fallos = np.random.binomial(n=1, p=tasa_fallo / 100, size=(unidades, num_maquinas))
        fallos_por_unidad = fallos.sum(axis=1)
        tiempo_total = tiempo_produccion * (1 + 0.1 * fallos_por_unidad)
        df_sim = pd.DataFrame({
            "Unidad": np.arange(1, unidades + 1),
            "Fallos": fallos_por_unidad,
            "Tiempo Total (s)": tiempo_total
        })
        st.dataframe(df_sim.head(10))
        st.success(f"Tiempo promedio de producción: {df_sim['Tiempo Total (s)'].mean():.2f} segundos")
        fig = px.histogram(df_sim, x="Tiempo Total (s)", nbins=30, title="Distribución del Tiempo de Producción")
        st.plotly_chart(fig, use_container_width=True)

# -------------------- 2. Predictivo --------------------
with tabs[1]:
    st.header("📈 2. Análisis Predictivo")
    uploaded_file = st.file_uploader("Carga un archivo CSV (Publicidad, Ventas)", type="csv")
    if uploaded_file:
        df_pred = pd.read_csv(uploaded_file)
    else:
        X = np.random.rand(50, 1) * 100
        y = 3 * X.flatten() + np.random.randn(50) * 10 + 50
        df_pred = pd.DataFrame({'Publicidad': X.flatten(), 'Ventas': y})

    st.dataframe(df_pred.head())

    modelo = LinearRegression().fit(df_pred[['Publicidad']], df_pred['Ventas'])
    user_gasto = st.slider("Gasto en publicidad para predecir", 0, 150, 50)
    y_pred = modelo.predict([[user_gasto]])[0]
    st.metric("Predicción de ventas", f"${y_pred:.2f}")

    fig = px.scatter(df_pred, x='Publicidad', y='Ventas', trendline="ols")
    fig.add_vline(x=user_gasto, line_dash="dot", line_color="green")
    st.plotly_chart(fig, use_container_width=True)

# -------------------- 3. Visualización --------------------
with tabs[2]:
    st.header("📊 3. Visualización Interactiva de Inventario")
    productos = ['Jabón Clásico', 'Jabón Suave', 'Jabón Neutro', 'Jabón Antibacterial']
    inventario = {p: st.number_input(f"Inventario de {p}", min_value=0, value=np.random.randint(100, 800)) for p in productos}
    df_inv = pd.DataFrame(list(inventario.items()), columns=["Producto", "Inventario"])
    st.dataframe(df_inv)
    filtro = st.multiselect("Filtrar productos:", productos, default=productos)
    df_filtrado = df_inv[df_inv["Producto"].isin(filtro)]
    fig = px.bar(df_filtrado, x="Producto", y="Inventario", text="Inventario", color="Producto")
    st.plotly_chart(fig, use_container_width=True)

# -------------------- 4. Automatización --------------------
with tabs[3]:
    st.header("📤 4. Genera tu propio reporte")
    with st.form("form_reporte"):
        dep = st.text_input("Departamento")
        emp = st.number_input("Número de empleados", min_value=0)
        sat = st.slider("Satisfacción", 0.0, 1.0, 0.8)
        submitted = st.form_submit_button("Agregar al reporte")
        if submitted and dep:
            if "reporte_df" not in st.session_state:
                st.session_state.reporte_df = pd.DataFrame(columns=["Departamento", "Empleados", "Satisfacción"])
            st.session_state.reporte_df.loc[len(st.session_state.reporte_df)] = [dep, emp, sat]

    if "reporte_df" in st.session_state:
        st.dataframe(st.session_state.reporte_df)
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            st.session_state.reporte_df.to_excel(writer, index=False)
        st.download_button("📥 Descargar reporte Excel", data=output.getvalue(),
                           file_name="reporte_personalizado.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -------------------- 5. Control de Calidad --------------------
with tabs[4]:
    st.header("🧪 5. Control de Calidad Dinámico")
    num_lotes = st.slider("Número de lotes", 10, 50, 30)
    media_esperada = st.number_input("Media esperada", value=100.0)
    desviacion = st.number_input("Desviación estándar", value=5.0)
    mediciones = np.random.normal(loc=media_esperada, scale=desviacion, size=num_lotes)
    if num_lotes >= 20:
        mediciones[10] = media_esperada + 4 * desviacion
        mediciones[25] = media_esperada - 4 * desviacion
    df_calidad = pd.DataFrame({'Lote': range(1, num_lotes+1), 'Medición': mediciones})
    LIC = media_esperada - 3 * desviacion
    LSC = media_esperada + 3 * desviacion
    df_calidad['Fuera de control'] = (df_calidad['Medición'] > LSC) | (df_calidad['Medición'] < LIC)
    st.dataframe(df_calidad)
    fig = px.line(df_calidad, x='Lote', y='Medición', markers=True,
                  color=df_calidad['Fuera de control'].map({True: '⚠️ Fuera de control', False: '✔️ OK'}),
                  color_discrete_map={'✔️ OK': 'blue', '⚠️ Fuera de control': 'red'})
    fig.add_hline(y=media_esperada, line_dash='dot', line_color='green', annotation_text='Media')
    fig.add_hline(y=LSC, line_dash='dash', line_color='red', annotation_text='LSC')
    fig.add_hline(y=LIC, line_dash='dash', line_color='red', annotation_text='LIC')
    st.plotly_chart(fig, use_container_width=True)
