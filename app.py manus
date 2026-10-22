import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Previsão de Consumo Industrial",
    page_icon="⚡",
    layout="wide",
)


@st.cache_data
def criar_dataset() -> pd.DataFrame:
    """Gera um conjunto sintético reproduzível com 30 observações."""
    rng = np.random.default_rng(42)
    velocidade = rng.uniform(20, 100, 30)  # metros por minuto
    maquinas = rng.integers(1, 11, 30)  # máquinas ativas
    temperatura = rng.uniform(15, 40, 30)  # graus Celsius
    ruido = rng.normal(0, 5, 30)

    # Relação sintética: as variáveis elevam o consumo, com uma pequena variação aleatória.
    consumo = (
        18
        + 0.72 * velocidade
        + 12.5 * maquinas
        + 1.8 * temperatura
        + ruido
    )

    return pd.DataFrame(
        {
            "Velocidade_Esteira": velocidade,
            "Maquinas_Ativas": maquinas,
            "Temp_Ambiente": temperatura,
            "Consumo_kWh": consumo,
        }
    )


df = criar_dataset()
features = ["Velocidade_Esteira", "Maquinas_Ativas", "Temp_Ambiente"]
alvo = "Consumo_kWh"

# Padronizar os atributos faz com que os pesos do modelo sejam comparáveis
# e permite calcular contribuições relativas ao cenário médio do dataset.
X = df[features]
y = df[alvo]
modelo = make_pipeline(StandardScaler(), LinearRegression())
modelo.fit(X, y)

st.title("⚡ Painel de Previsão de Consumo de Energia Industrial")
st.write(
    "Informe as condições de operação na barra lateral para estimar o consumo. "
    "O modelo foi treinado com um dataset sintético de 30 registros."
)

st.sidebar.header("Parâmetros de entrada")
velocidade_input = st.sidebar.number_input(
    "Velocidade da esteira (m/min)",
    min_value=0.0,
    max_value=150.0,
    value=60.0,
    step=1.0,
)
maquinas_input = st.sidebar.number_input(
    "Máquinas ativas",
    min_value=0,
    max_value=20,
    value=5,
    step=1,
)
temperatura_input = st.sidebar.number_input(
    "Temperatura ambiente (°C)",
    min_value=-10.0,
    max_value=60.0,
    value=25.0,
    step=1.0,
)

entrada = pd.DataFrame(
    [[velocidade_input, maquinas_input, temperatura_input]],
    columns=features,
)
previsao = float(modelo.predict(entrada)[0])

st.subheader("Consumo previsto")
st.metric(
    label="Consumo estimado",
    value=f"{previsao:,.1f} kWh",
    help="Estimativa calculada pelo modelo de regressão linear.",
)

# Contribuição de cada variável para a diferença entre a previsão atual e
# a previsão no cenário médio do dataset. As contribuições somam essa diferença.
scaler = modelo.named_steps["standardscaler"]
regressor = modelo.named_steps["linearregression"]
entrada_padronizada = scaler.transform(entrada)[0]
contribuicoes = regressor.coef_ * entrada_padronizada

st.subheader("Contribuição das variáveis para a previsão")
st.caption(
    "As barras mostram quanto cada parâmetro desloca a previsão em relação "
    "ao cenário médio dos dados de treinamento. Valores negativos reduzem a estimativa."
)

fig, ax = plt.subplots(figsize=(9, 4.5))
cores = ["#2E8B57" if valor >= 0 else "#D9534F" for valor in contribuicoes]
barras = ax.bar(features, contribuicoes, color=cores)
ax.axhline(0, color="#333333", linewidth=0.8)
ax.set_ylabel("Contribuição para a previsão (kWh)")
ax.set_title("Impacto de cada variável em relação ao cenário médio")
ax.tick_params(axis="x", rotation=12)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.bar_label(barras, fmt="%+.1f", padding=3)
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

with st.expander("Visualizar dataset de treinamento"):
    st.dataframe(df.round(2), use_container_width=True)
    st.caption(f"Registros: {len(df)} | Algoritmo: Regressão Linear")
