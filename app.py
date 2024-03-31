from decimal import Decimal
from io import BytesIO
from math import log10, floor

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st


################################################################################
# ESTADO DA SESSÃO

if "dados" not in st.session_state:
    st.session_state["dados"] = pd.DataFrame()

dados = st.session_state["dados"]


#############################################################
# PARÂMETROS DE ENTRADA

with st.sidebar:
    st.header("MP-resuldados", divider=True)
    st.subheader("Número de divisões:")
    col1, col2 = st.columns(2)
    with col1:
        h = st.number_input(
            "horizontal", value=180, min_value=100, max_value=300, step=10
        )
    with col2:
        v = st.number_input(
            "vertical", value=280, min_value=100, max_value=300, step=10
        )

    titulo = st.text_input(
        label="título do gráfico",
        value="Papel milimetrado",
    )

    xlabel = st.text_input(
        label="nome do eixo horizontal (unidade)",
        value="eixo horizontal",
    )

    ylabel = st.text_input(
        label="nome do eixo vertical (unidade)",
        value="eixo vertical",
    )

    rot = st.slider(
        label="rotação dos números da escala horizontal",
        value=0,
        min_value=-90,
        max_value=90,
        step=1,
    )


#############################################################
# GERAR O PAPEL


def gravar_dados(dados):
    dados.fillna(0, inplace=True)
    st.session_state["dados"] = dados


def escala(h, v, dados):
    if not dados.empty:
        # cálculo do delta
        delta_x = dados["x"].max() - dados["x"].min()
        dados_y_max = (dados["y"] + dados["erro"]).max()
        dados_y_min = (dados["y"] - dados["erro"]).min()
        delta_y = dados_y_max - dados_y_min

        # cálculo da escala natural
        escala_nat_x = delta_x / h
        escala_nat_y = delta_y / v

        # função auxiliar para o cálculo da escala boa
        def fexp(number):
            (sign, digits, exponent) = Decimal(number).as_tuple()
            return len(digits) + exponent - 1

        def fman(number):
            return float(Decimal(number).scaleb(-fexp(number)).normalize())

        # cálculo da escala boa
        def escala_boa(num):
            exp = 10 ** fexp(num)
            man = fman(num)
            if man == 5 or man == 2 or man == 1:
                return num
            if man > 5:
                return 10 * exp
            if man > 2:
                return 5 * exp
            if man > 1:
                return 2 * exp
            return exp

        escala_boa_x = escala_boa(escala_nat_x)
        escala_boa_y = escala_boa(escala_nat_y)

        delta_bom_x = escala_boa_x * h
        delta_bom_y = escala_boa_y * v

        # cálculo dos limites
        limite_x = [
            round(dados["x"].min() - (delta_bom_x - delta_x) / 2, 10),
            round(dados["x"].max() + (delta_bom_x - delta_x) / 2, 10),
        ]
        limite_y = [
            round(dados_y_min - (delta_bom_y - delta_y) / 2, 10),
            round(dados_y_max + (delta_bom_y - delta_y) / 2, 10),
        ]

        # cálculo dos limites bons
        def limite_bom(esc, num):
            esc_cm = 10 * esc
            return round(num / esc_cm) * esc_cm

        v_limite_bom = np.vectorize(limite_bom)

        limite_bom_x = [
            round(limite_bom(escala_boa_x, limite_x[1]) - h * escala_boa_x, 10),
            round(limite_bom(escala_boa_x, limite_x[1]), 10),
        ]

        limite_bom_y = [
            round(limite_bom(escala_boa_y, limite_y[1]) - v * escala_boa_y, 10),
            round(limite_bom(escala_boa_y, limite_y[1]), 10),
        ]

        # cálculo das divisões
        div_x = [
            round(limite_bom_x[0] + escala_boa_x * 10 * i, 10)
            for i in range(0, int(h / 10 + 1))
        ]
        div_y = [
            round(limite_bom_y[0] + escala_boa_y * 10 * i, 10)
            for i in range(0, int(v / 10 + 1))
        ]

        # conversão para mm
        dados_mm = pd.DataFrame()
        dados_mm["x_mm"] = v_limite_bom(0.05, (dados["x"] - div_x[0]) / escala_boa_x)
        dados_mm["y_mm"] = v_limite_bom(0.05, (dados["y"] - div_y[0]) / escala_boa_y)
        dados_mm["erro_mm"] = v_limite_bom(
            0.05, dados["erro"] / escala_boa_y
        )  # tamanho da incerteza

    else:
        delta_x = None
        delta_y = None
        escala_nat_x = None
        escala_nat_y = None
        escala_boa_x = None
        escala_boa_y = None
        delta_bom_x = None
        delta_bom_y = None
        limite_x = None
        limite_y = None
        limite_bom_x = None
        limite_bom_y = None
        div_x = np.arange(0, h + 1, 10)
        div_y = np.arange(0, v + 1, 10)
        dados_mm = None

    return (
        h,
        v,
        delta_x,
        delta_y,
        escala_nat_x,
        escala_nat_y,
        escala_boa_x,
        escala_boa_y,
        delta_bom_x,
        delta_bom_y,
        limite_x,
        limite_y,
        limite_bom_x,
        limite_bom_y,
        div_x,
        div_y,
        dados_mm,
    )

def paralelogramo(dados):
    x = dados['x']
    y = dados['y']
    y_err = dados['erro']
    
    modelo = LinearRegression().fit(np.array(x).reshape(-1, 1), y)
    a = modelo.coef_[0]
    y_reta = modelo.predict(np.array(x).reshape(-1, 1))
    
    idx_max = (y+y_err-y_reta).idxmax( )
    idx_min = (y-y_err-y_reta).idxmin( )

    b1 = ( (y+y_err)[idx_max]- a*x[idx_max] )
    b2 = ( (y-y_err)[idx_min]- a*x[idx_min] )
    
    a_max = ( (a*x.max() + b1)-(a*x.min() + b2) )/ (x.max()-x.min())
    a_min = ( (a*x.max() + b2)-(a*x.min() + b1) )/ (x.max()-x.min())
    
    a_medio = (a_max + a_min)/2
    erro_a = round_it((a_max - a_min)/2,1)

    b_max = (a*x.max()+b2) - a_min*x.max()
    b_min = (a*x.max()+b1) - a_max*x.max()
    
    b_medio = (b_max + b_min)/2
    erro_b = round_it((b_max - b_min)/2,1)
    
    return a_max, a_min, b_max, b_min, a_medio, erro_a, b_medio, erro_b, b1, b2


def plot(h, v, dados, xlabel, ylabel):
    # cálculo da escala
    (
        h,
        v,
        delta_x,
        delta_y,
        escala_nat_x,
        escala_nat_y,
        escala_boa_x,
        escala_boa_y,
        delta_bom_x,
        delta_bom_y,
        limite_x,
        limite_y,
        limite_bom_x,
        limite_bom_y,
        div_x,
        div_y,
        dados_mm,
    ) = escala(h, v, dados)

    try:
        arquivo = f"""{h} divisões na horizontal
{v} divisões na vertical
________________________________________________
Resultados eixo horizontal:

\u0394 = {round(delta_x,10)}
escala natural = {round(escala_nat_x,10)}
escala = {round(escala_boa_x,10)}
\u0394' = {round(delta_bom_x,10)}
limites = {limite_x}
limites corrigidos = {limite_bom_x}
escala de leitura 
{div_x}

________________________________________________
Resultados eixo vertical:

\u0394 = {round(delta_y,10)}
escala natural = {round(escala_nat_y,10)}
escala = {round(escala_boa_y,10)}
\u0394' = {round(delta_bom_y,10)}
limites = {limite_y}
limites corrigidos = {limite_bom_y}
escala de leitura 
{div_y}

________________________________________________
Dados em divisões 
{dados_mm}           

-----------------------------------------------------------------------------
MP-resuldados
Dos dados aos resultados. Um pouco de física, matemática, negócios e finanças.
mp.resuldados@gmail.com
"""
    except TypeError:
        arquivo = "Dados não informados."

    # proporcionalização
    ratio = (v / (np.array(div_y).max() - np.array(div_y).min())) / (
        h / (np.array(div_x).max() - np.array(div_x).min())
    )

    fig, ax = plt.subplots(figsize=(8.3, 11.7))  # tamanho A4

    ax.set_aspect(ratio)

    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xlim(div_x[0], div_x[-1])
    ax.set_ylim(div_y[0], div_y[-1])

    ax.set_axisbelow(True)

    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_yticks(div_y)
    div_y_minor = np.arange(div_y[0], div_y[-1], (div_y[1] - div_y[0]) / 10)
    ax.hlines(div_y_minor, div_x[0], div_x[-1], lw=0.1, color="lightgray")

    ax.set_xticks(div_x)
    div_x_minor = np.arange(div_x[0], div_x[-1], (div_x[1] - div_x[0]) / 10)
    ax.vlines(div_x_minor, div_y[0], div_y[-1], lw=0.1, color="lightgray")

    for label in ax.get_xticklabels(which="major"):
        label.set(rotation=rot)

    ax.hlines(div_y, div_x[0], div_x[-1], lw=1, color="darkgray")
    ax.vlines(div_x, div_y[0], div_y[-1], lw=1, color="darkgray")

    ax.text(
        0.5,
        0.5,
        "MP-resuldados",
        transform=ax.transAxes,
        fontsize=40,
        color="gray",
        alpha=0.1,
        ha="center",
        va="center",
        rotation=45,
    )

    # plot
    if not dados.empty:
        ax.errorbar(
            dados["x"],
            dados["y"],
            yerr=dados["erro"],
            marker=".",
            linestyle="none",
            color="green",
            label="dados experimentais",
        )

    return fig, arquivo

def plotar_paralelogramo(ax, a_max, a_min, b_max, b_min, a_medio, b_medio, b1, b2):
    x = dados['x']
    ax.plot(
        x,
        a_medio*x+b_medio,
        linestyle='-',
        color='red',
        label='ajuste linear',
    )
    ax.plot(
        x,
        a_medio*x+b1,
        linestyle='--',
        color='orange',
        label='método do paralelogramo',
    )
    ax.plot(
        x,
        a_medio*x+b2,
        linestyle='--',
        color='orange',

    )
    ax.plot(
        x,
        a_max*x+b_min,
        linestyle=':',
        color='orange',
    )
    ax.plot(
        x,
        a_min*x+b_max,
        linestyle=':',
        color='orange',
    )

    ax.legend(loc='upper left', ncols=1)
    
    try:
        arquivo = f'''_______________________________________________________
Resultados dos coeficientes do método do paralelogramo:

reta com maior coeficiente angular:
a_max = {a_max}
b_min = {b_min}

reta com menor coeficiente angular:
a_min = {a_min}
b_max = {b_max}
_______________________________________________________
Resultados dos coeficientes da reta y = a x + b:

a = {a_medio}
erro de a = {erro_a}

b = {b_medio}
erro de b = {erro_b}          

-----------------------------------------------------------------------------
MP-resuldados
Dos dados aos resultados. Um pouco de física, matemática, negócios e finanças.
mp.resuldados@gmail.com
'''
    
    except TypeError:
        arquivo = "Dados não informados."
    
    return arquivo
    
def round_it(x, sig):
    return round(x, sig - int(floor(log10(abs(x)))) - 1)



fig, arquivo = plot(h, v, dados, xlabel, ylabel)

df = pd.DataFrame(columns=["x", "y", "erro"], dtype="float")


col1, col2 = st.columns([2, 2])
with col1:
    with st.form("entrada de dados"):
        dados = st.data_editor(df, num_rows="dynamic")
        plotar = st.form_submit_button("plotar")
        if plotar:
            gravar_dados(dados)
            st.rerun()
            
    paralelo = st.button("método do paralelogramo")
    
    if paralelo:
        # função de plotar paralelogramo
        a_max, a_min, b_max, b_min, a_medio, erro_a, b_medio, erro_b, b1, b2 = paralelogramo(dados)
        ax = plt.gca() # verificar se precisa da fig
        arquivo2 = plotar_paralelogramo(ax, a_max, a_min, b_max, b_min, a_medio, b_medio, b1, b2)
    else:
        arquivo2 = ''
            
    subcols = st.columns(2)
    with subcols[0]:
        st.download_button(
            label="resultados da escala",
            data=arquivo,
            file_name="arquivo.txt",
        )

        st.download_button(
            label="resultados do paralelogramo",
            data=arquivo2,
            file_name="arquivo2.txt",
        )

    with subcols[1]:
        file = BytesIO()
        fig.savefig(file, format="pdf")

        st.download_button(
            label="baixar figura",
            data=file,
            file_name="figura.pdf",
        )

        
        
with col2:
    st.pyplot(fig)
