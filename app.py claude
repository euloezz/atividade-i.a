"""
Painel de previsão de consumo de energia industrial.

Executar com:
    pip install streamlit pandas numpy scikit-learn matplotlib
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Previsão de Consumo Industrial", page_icon="⚡", layout="wide")

FEATURES = ["Velocidade_Esteira", "Maquinas_Ativas", "Temp_Ambiente"]
TARGET = "Consumo_kWh"


# ---------------------------------------------------------------------------
# 1) Dataset sintético
# ---------------------------------------------------------------------------
@st.cache_data
def gerar_dataset(n: int = 40, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    velocidade = rng.uniform(0.5, 3.0, n).round(2)       # m/s
    maquinas = rng.integers(2, 21, n)                     # unidades
    temperatura = rng.uniform(15, 40, n).round(1)         # °C

    # Relação "verdadeira" + ruído
    consumo = (
        50
        + 35 * velocidade
        + 22 * maquinas
        + 3.5 * temperatura
        + rng.normal(0, 12, n)
    ).round(2)

    return pd.DataFrame(
        {
            "Velocidade_Esteira": velocidade,
            "Maquinas_Ativas": maquinas,
            "Temp_Ambiente": temperatura,
            "Consumo_kWh": consumo,
        }
    )


# ---------------------------------------------------------------------------
# 2) Treinamento do modelo
# ---------------------------------------------------------------------------
@st.cache_resource
def treinar_modelo(df: pd.DataFrame):
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

    modelo = LinearRegression().fit(X_tr, y_tr)
    pred = modelo.predict(X_te)
    metricas = {"r2": r2_score(y_te, pred), "mae": mean_absolute_error(y_te, pred)}

    # Reajusta com todos os dados para uso final no painel
    modelo_final = LinearRegression().fit(X, y)
    return modelo_final, metricas


df = gerar_dataset()
modelo, metricas = treinar_modelo(df)

# ---------------------------------------------------------------------------
# 3) Sidebar: parâmetros de entrada
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Parâmetros de entrada")

velocidade = st.sidebar.number_input(
    "Velocidade da esteira (m/s)",
    min_value=0.0, max_value=5.0, value=1.5, step=0.1,
)
maquinas = st.sidebar.number_input(
    "Máquinas ativas (un.)",
    min_value=0, max_value=50, value=10, step=1,
)
temperatura = st.sidebar.number_input(
    "Temperatura ambiente (°C)",
    min_value=-10.0, max_value=60.0, value=25.0, step=0.5,
)

entrada = pd.DataFrame([[velocidade, maquinas, temperatura]], columns=FEATURES)
previsao = float(modelo.predict(entrada)[0])

# ---------------------------------------------------------------------------
# 4) Tela principal
# ---------------------------------------------------------------------------
st.title("⚡ Previsão de Consumo de Energia Industrial")
st.caption("Modelo de Regressão Linear (scikit-learn) treinado com dados sintéticos.")

col1, col2, col3 = st.columns(3)
col1.metric("Consumo previsto", f"{previsao:,.1f} kWh")
col2.metric("R² (teste)", f"{metricas['r2']:.3f}")
col3.metric("Erro médio absoluto (teste)", f"{metricas['mae']:.1f} kWh")

st.divider()
st.subheader("📊 Contribuição de cada variável no modelo")

coefs = pd.Series(modelo.coef_, index=FEATURES)
contribuicao_atual = coefs * entrada.iloc[0]                 # coef × valor informado
importancia_padronizada = coefs * df[FEATURES].std()         # coef × desvio-padrão

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].bar(FEATURES, importancia_padronizada.values, color="#4C78A8")
axes[0].set_title("Importância geral (coeficiente × desvio-padrão)")
axes[0].set_ylabel("kWh por 1 desvio-padrão")

axes[1].bar(FEATURES, contribuicao_atual.values, color="#F58518")
axes[1].set_title("Contribuição na previsão atual (coeficiente × entrada)")
axes[1].set_ylabel("kWh")

for ax in axes:
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

fig.tight_layout()
st.pyplot(fig)

st.caption(
    f"Consumo base do modelo (intercepto): {modelo.intercept_:,.1f} kWh. "
    "Previsão = intercepto + soma das contribuições à direita."
)

with st.expander("Ver dataset de treino"):
    st.dataframe(df, use_container_width=True)
