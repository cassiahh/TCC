from graficos import *
from processa_dados import *
from predicao import *

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.seasonal import seasonal_decompose
from sqlalchemy import create_engine, MetaData, Table
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import seaborn as sns
from pandas_datareader import data as web
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, LeakyReLU
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import os
import pandas.util.testing as tm
from datetime import date
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 20, 10
plt.style.use('ggplot')
import warnings

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")


st.title('Predição de ETF com aprendizado de máquina')

st.header('Disclaimer')
with st.beta_expander("Leia Me"):
    st.write("""
    As informações e as predições aqui contidas foram consolidadas ou elaboradas com base em informações obtidas de fontes, em princípio, fidedignas e de boa-fé. 
    Entretanto, a desenvolvedora não declara, nem garante, expressa ou tacitamente, que estas informações e opiniões sejam imparciais, precisas, completas ou corretas. 
    Todas as recomendações e estimativas apresentadas podem ser alteradas a qualquer momento sem aviso prévio, em função de mudanças que possam afetar as projeções da empresa.\n  
    Este material tem por finalidade apenas informar e servir como instrumento que auxilie a tomada de decisão de investimento. 
    Não é, e não deve ser interpretado como uma oferta ou solicitação de oferta para comprar ou vender quaisquer títulos e valores mobiliários ou outros instrumentos financeiros.
    """)


st.subheader('Código de Exchange-Traded Fund')
st.markdown('''
    <a href="http://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/etf/renda-variavel/etfs-listados/"  target=_blank>
        Detalhes dos ETFs Listados no site da B3 
    </a>''', unsafe_allow_html=True)
with st.beta_expander("Lista de ETFs disponíveis no B3"):
    st.write("""
    BBOV11.SA - BB ETF IBOVESPA FUNDO DE ÍNDICE\n 
    BBSD11.SA - BB ETF S&P DIVIDENDOS BRASIL FUNDO DE ÍNDICE \n
    ESGB11.SA - BTG PACTUAL ESG FUNDO DE ÍNDICE S&P/B3 BRAZIL ES \n
    XBOV11.SA - CAIXA ETF IBOVESPA FUNDO DE INDICE  \n
    BOVB11.SA - ETF BRADESCO IBOVESPA FDO DE INDICE  \n
    HASH11.SA - HASHDEX NASDAQ CRYPTO INDEX FUNDO DE ÍNDICE  \n
    SMAL11.SA - ISHARES BMFBOVESPA SMALL CAP FUNDO DE ÍNDICE  \n
    BOVA11.SA - ISHARES IBOVESPA FUNDO DE ÍNDICE  \n
    BRAX11.SA - ISHARES IBRX - ÍNDICE BRASIL (IBRX-100) FDO ÍNDICE \n
    ECOO11.SA - ISHARES ÍNDICE CARBONO EFIC. (ICO2) BRASIL-FDO ÍND  \n
    IVVB11.SA - ISHARES S&P 500 FDO INV COTAS FDO INDICE  \n
    BOVV11.SA - IT NOW IBOVESPA FUNDO DE ÍNDICE  \n
    DIVO11.SA - IT NOW IDIV FUNDO DE ÍNDICE  \n
    FIND11.SA - IT NOW IFNC FUNDO DE ÍNDICE  \n
    GOVE11.SA - IT NOW IGCT FUNDO DE ÍNDICE  \n
    MATB11.SA - IT NOW IMAT FUNDO DE ÍNDICE  \n
    ISUS11.SA - IT NOW ISE FUNDO DE ÍNDICE  \n
    TECK11.SA - IT NOW NYSE FANG+TM FUNDO DE ÍNDICE  \n
    PIBB11.SA - IT NOW PIBB IBRX-50 - FUNDO DE ÍNDICE  \n
    SPXI11.SA - IT NOW S&P500 TRN FUNDO DE INDICE  \n
    SMAC11.SA - IT NOW SMALL FDO ÍNDICE  \n
    XFIX11.SA - TREND ETF IFIX FUNDO DE ÍNDICE  \n
    GOLD11.SA - TREND ETF LBMA OURO FDO. INV. ÍNDICE - INVEST. EXT  \n
    ACWI11.SA - TREND ETF MSCI ACWI FDO. INV. ÍNDICE - INVEST. EXT  \n
    XINA11.SA - TREND ETF MSCI CHINA FDO. INV. ÍNDICE - INV. EXT  \n
    EURP11.SA - TREND ETF MSCI EUROPA FDO. INV. ÍNDICE - INV. EXT \n
    """)


df = pd.DataFrame()

with st.beta_container():
    form = st.form(key='my-form1')
    acao = form.text_input('Insira um código de ETF (exemplo: BOVA11.SA)')
    submit = form.form_submit_button('Inserir')
    if not acao:
        #    st.warning('Please input a ticker.')
        st.stop()
    # fazer validação de input
    if submit:
        st.write(f'Código inserido: {acao}')
    # Web Scraping
    df = web.DataReader(acao, data_source='yahoo', start='01-01-2017')
    # tratar erro
    df.drop('Adj Close', inplace=True, axis=1)


# Exibir dados brutos
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
with st.beta_container():
    st.subheader('Escolha um modo de exibição dos dados brutos')
    # transformar em função e usar @st.cache
    #st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    chosen = st.radio(
        'raw data',
        ("Não exibir", "Tail", "Head", "Dataframe"))
    if chosen == "Não exibir":
        st.write(f"Opção escolhida é: {chosen}")
    if chosen == "Tail":
        st.subheader('Tail')
        st.dataframe(df.tail())
    if chosen == "Head":
        st.subheader('Head')
        st.dataframe(df.head())
    if chosen == "Dataframe":
        st.subheader('Tabela')
        st.dataframe(df)


# plotar o gráfico de candlestick
Date = pd.DataFrame(df.index)
df['Date'] = df.index
if st.checkbox('gráfico Candlestick'):
    grafico_candlestick(df, acao)
    # verificar como retirar mensagem "None"


st.subheader('Estatística Descritiva')
if st.checkbox('abrir a síntese'):
    st.write(df.describe().round(5))


st.subheader('Preço fechamento x Volume transacionado')
if st.checkbox('gráfico Scatterplot'):
    grafico_scatterplot(df)


st.subheader('Variação dos preços de um dia para o outro')
if st.checkbox('Visualizar a Decomposição'):
    decomposicao(df)

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx



st.header('Máquina Preditiva com LSTM')
if st.checkbox('Clique para construir modelo 1'):

    #lstm_predict(tratamento_dados(df))

#    if st.checkbox('gráfico model1'):
    grafico_dos_valores_preditos(lstm_predict(tratamento_dados(df)))

st.header('Máquina Preditiva com Stacked LSTM networks')
if st.checkbox('gráfico model2'):
    pass

    #    grafico_dos_valores_preditos(prediction, y_test, scaler)

#Comparando os modelos da Máquina Preditiva
st.subheader('Comparação os modelos das Máquinas Preditivas')
if st.checkbox('model1 x model2'):
    #   grafico_compara_models(prediction_inverse, prediction2_inverse, y_test, scaler)
    pass
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


st.header('Termos e Política de Privacidade')
with st.beta_expander("Politica de privacidade"):
    st.write("""
    (...)
    \n
    """)
with st.beta_expander("Termos de Tratamento de Dados"):
    st.write("""
    (...)
    \n
    """)
with st.beta_expander("Termos de Serviço"):
    st.write("""
    Esta página foi produzida com finalidade acadêmica - TCC Cássia Chin de curso ADS do IFSP. 
    """)
with st.beta_expander("Cookies"):
    st.write("""
    (...)
    \n
    """)
