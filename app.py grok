import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(
    page_title="Previsão de Consumo de Energia Industrial",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Painel de Previsão de Consumo de Energia Industrial")
st.markdown("---")

# ============================================================
# 1. Geração do Dataset Sintético
# ============================================================
@st.cache_data
def gerar_dataset(n_registros=50):
    np.random.seed(42)
    
    velocidade = np.random.uniform(1.0, 12.0, n_registros)          # m/s
    maquinas = np.random.randint(1, 25, n_registros)                # quantidade
    temp_ambiente = np.random.uniform(15.0, 40.0, n_registros)     # °C
    
    # Fórmula realista + ruído
    consumo = (
        8.5 * velocidade +
        12.0 * maquinas +
        1.8 * temp_ambiente +
        np.random.normal(0, 15, n_registros)
    )
    consumo = np.clip(consumo, 50, None)  # evita valores negativos irreais
    
    df = pd.DataFrame({
        "Velocidade_Esteira": np.round(velocidade, 2),
        "Maquinas_Ativas": maquinas,
        "Temp_Ambiente": np.round(temp_ambiente, 1),
        "Consumo_kWh": np.round(consumo, 1)
    })
    return df

df = gerar_dataset(50)

# ============================================================
# 2. Treinamento do Modelo de Regressão
# ============================================================
@st.cache_resource
def treinar_modelo(dataframe):
    X = dataframe[["Velocidade_Esteira", "Maquinas_Ativas", "Temp_Ambiente"]]
    y = dataframe["Consumo_kWh"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    
    score = modelo.score(X_test, y_test)
    return modelo, score, X.columns.tolist()

modelo, r2_score, feature_names = treinar_modelo(df)

# ============================================================
# 3. Sidebar – Controles de Entrada
# ============================================================
st.sidebar.header("⚙️ Parâmetros de Entrada")
st.sidebar.markdown("Ajuste os valores abaixo para obter a previsão:")

velocidade_input = st.sidebar.number_input(
    "Velocidade da Esteira (m/s)",
    min_value=0.5,
    max_value=15.0,
    value=5.0,
    step=0.1,
    format="%.1f"
)

maquinas_input = st.sidebar.number_input(
    "Máquinas Ativas",
    min_value=1,
    max_value=30,
    value=10,
    step=1
)

temp_input = st.sidebar.number_input(
    "Temperatura Ambiente (°C)",
    min_value=10.0,
    max_value=45.0,
    value=25.0,
    step=0.5,
    format="%.1f"
)

st.sidebar.markdown("---")
st.sidebar.info(f"**R² do modelo:** {r2_score:.3f}")

# ============================================================
# 4. Previsão
# ============================================================
entrada = np.array([[velocidade_input, maquinas_input, temp_input]])
consumo_previsto = modelo.predict(entrada)[0]

# Contribuição de cada variável (coeficiente × valor de entrada)
coeficientes = modelo.coef_
contribuicoes = coeficientes * entrada[0]
nomes_features = ["Velocidade Esteira", "Máquinas Ativas", "Temp. Ambiente"]

# ============================================================
# 5. Tela Principal
# ============================================================
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📊 Resultado da Previsão")
    st.metric(
        label="Consumo Estimado de Energia",
        value=f"{consumo_previsto:,.1f} kWh",
        delta=None
    )
    
    st.markdown("### Detalhes da Predição")
    st.write(f"**Velocidade da Esteira:** {velocidade_input:.1f} m/s")
    st.write(f"**Máquinas Ativas:** {maquinas_input}")
    st.write(f"**Temperatura Ambiente:** {temp_input:.1f} °C")
    
    st.markdown("---")
    st.caption("Modelo: Regressão Linear (scikit-learn)")

with col2:
    st.subheader("📈 Contribuição de cada Variável")
    
    # Gráfico de barras com Matplotlib
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    cores = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    barras = ax.barh(nomes_features, contribuicoes, color=cores, edgecolor="black", height=0.6)
    
    # Adiciona valores nas barras
    for barra, valor in zip(barras, contribuicoes):
        largura = barra.get_width()
        ax.text(
            largura + (0.5 if largura >= 0 else -0.5),
            barra.get_y() + barra.get_height()/2,
            f"{valor:.1f}",
            va="center",
            ha="left" if largura >= 0 else "right",
            fontsize=10,
            fontweight="bold"
        )
    
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Contribuição para o Consumo (kWh)", fontsize=11)
    ax.set_title("Impacto de cada Variável na Previsão", fontsize=13, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    
    st.pyplot(fig)

# ============================================================
# 6. Dataset de Treinamento (opcional – expansível)
# ============================================================
with st.expander("📁 Ver Dataset de Treinamento (50 registros)"):
    st.dataframe(df, use_container_width=True)
    st.caption("Dataset gerado sinteticamente para fins de demonstração.")
