import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Previsão de Consumo de Energia", page_icon="⚡", layout="wide"
)

st.title("⚡ Painel de Previsão de Consumo de Energia Industrial")
st.markdown(
    "Este aplicativo utiliza um modelo de **Regressão Linear** treinado com dados simulados para prever o consumo de energia com base em variáveis operacionais da fábrica."
)


# 1. Geração do Dataset e Treinamento do Modelo (com cache para otimização)
@st.cache_data
def carregar_dados_e_modelo():
    np.random.seed(42)
    n_registros = 100

    velocidade = np.random.uniform(5.0, 30.0, n_registros)
    maquinas = np.random.randint(1, 15, n_registros)
    temp_amb = np.random.uniform(15.0, 35.0, n_registros)

    # Simulação de consumo de energia com uma relação linear + ruído
    consumo = (
        (12.5 * velocidade)
        + (45.0 * maquinas)
        + (3.2 * temp_amb)
        + np.random.normal(0, 10, n_registros)
    )

    df = pd.DataFrame(
        {
            "Velocidade_Esteira": velocidade,
            "Maquinas_Ativas": maquinas,
            "Temp_Ambiente": temp_amb,
            "Consumo_kWh": consumo,
        }
    )

    # Separando variáveis preditoras (X) e alvo (y)
    X = df[["Velocidade_Esteira", "Maquinas_Ativas", "Temp_Ambiente"]]
    y = df["Consumo_kWh"]

    # Treinamento do modelo de Regressão Linear
    model = LinearRegression()
    model.fit(X, y)

    return df, model


df, model = carregar_dados_e_modelo()

# 2. Barra lateral (Sidebar) com controles numéricos
st.sidebar.header("Painel de Controle")
st.sidebar.markdown("Ajuste os parâmetros operacionais abaixo:")

vel_input = st.sidebar.slider(
    "Velocidade da Esteira (m/min)",
    float(df["Velocidade_Esteira"].min()),
    float(df["Velocidade_Esteira"].max()),
    float(df["Velocidade_Esteira"].mean()),
)

maq_input = st.sidebar.slider(
    "Máquinas Ativas",
    int(df["Maquinas_Ativas"].min()),
    int(df["Maquinas_Ativas"].max()),
    int(df["Maquinas_Ativas"].mean()),
)

temp_input = st.sidebar.slider(
    "Temperatura Ambiente (°C)",
    float(df["Temp_Ambiente"].min()),
    float(df["Temp_Ambiente"].max()),
    float(df["Temp_Ambiente"].mean()),
)

# 3. Realizar a previsão com base na entrada do usuário
entrada_usuario = np.array([[vel_input, maq_input, temp_input]])
consumo_previsto = model.predict(entrada_usuario)[0]

# 4. Layout Principal
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Resultado da Previsão")
    st.metric(
        label="Consumo Estimado de Energia", value=f"{consumo_previsto:.2f} kWh"
    )

    st.info(
        "💡 **Como usar:** Mova os seletores na barra lateral para simular diferentes cenários industriais e observar a variação do consumo em tempo real."
    )

with col2:
    st.subheader("Contribuição das Variáveis (Coeficientes)")

    # Gráfico de barras usando Matplotlib
    features = ["Velocidade Esteira", "Máquinas Ativas", "Temp. Ambiente"]
    coefs = model.coef_

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    bars = ax.bar(features, coefs, color=colors)

    ax.set_ylabel("Peso no Modelo (Coeficiente)")
    ax.set_title("Impacto de Cada Parâmetro no Consumo")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Rótulos de dados nas barras
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    st.pyplot(fig)

# Expansor para visualizar os dados brutos de treino
with st.expander("🔍 Visualizar Dataset de Treinamento (Primeiras 20 linhas)"):
    st.dataframe(df.head(20))
