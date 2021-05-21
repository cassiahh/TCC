#from graficos import *
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
import pandas.util.testing as tm
from datetime import datetime
import os
from io import StringIO
import sys
import time
from time import sleep
import os
import streamlit.components.v1 as components
from tqdm import tqdm
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 10
plt.style.use('ggplot')
import warnings
warnings.filterwarnings("ignore")


#st.image("https://edumoreira.b-cdn.net/wp-content/uploads/2019/09/Qual-a-diferen%C3%A7a-entre-a%C3%A7%C3%B5es-e-ETFs.jpg")
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
#    st.image("https://static.streamlit.io/examples/dice.jpg")


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

# Web Scraping
df = pd.DataFrame()

form = st.form(key='my-form1')
acao = form.text_input('Insira um código de ETF (exemplo: BOVA11.SA)')
submit = form.form_submit_button('Inserir')
if not acao:
    #    st.warning('Please input a ticker.')
    st.stop()
# fazer validação de input
if submit:
    acao = acao.strip().upper()
    st.write(f'Código do ticker inserido: {acao}')

df = web.DataReader(acao, data_source='yahoo', start='01-01-2017')
# tratar erros: raise RemoteDataError, ConnectionError
# tratar dados ausentes
df.drop('Adj Close', inplace=True, axis=1)


# Exibir dados brutos
with st.beta_container():
    st.subheader('Escolha um modo de exibição dos dados brutos')
    # transformar em função e usar @st.cache
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
    candlestick = {
        'x': df.index,
        'open': df.Open,
        'close': df.Close,
        'high': df.High,
        'low': df.Low,
        'type': 'candlestick',
        'name': acao,
        'showlegend': False
    }
    data = [candlestick]
    layout = go.Layout()

    fig = go.Figure(data=data, layout=layout)
    c = fig.show()

    # st.write(c)
    st.write('Gráfico aberto em uma nova aba para melhor visualização', c)
    # st.write('Below is a DataFrame:', data_frame, 'Above is a dataframe.')
    # verificar como retirar mensagem "None"

st.subheader('Estatística Descritiva')
if st.checkbox('veja síntese'):
    st.write(df.describe().round(5))


st.subheader('Preço fechamento x Volume transacionado')
if st.checkbox('Clique para ver gráfico Scatterplot'):
    fig = plt.figure(figsize=(16, 10))
    sns.scatterplot(x="Close", y="Volume", data=df)
    plt.xlabel('Preço fechamento', fontsize=12)
    plt.ylabel('Volume transacionado', fontsize=12)
    plt.title("Preço fechamento X Volume transacionado")
    st.pyplot(fig)
    st.write('O Coeficiente de correlação de Pearson entre o Preço fechamento e o Volume transacionado é de: ')
    st.write(df['Close'].corr(df['Volume']))


def plot_comparacao(x, y1, y2, dataset1, dataset2, titulo):
    plt.figure(figsize=(16, 12))
    ax = plt.subplot(2, 1, 1)
    ax.set_title(titulo, fontsize=18, loc='left')
    sns.lineplot(x=x, y=y1, data=dataset1)
    plt.subplot(2, 1, 2)
    sns.lineplot(x=x, y=y2, data=dataset2)
 #   ax = ax


st.subheader('Variação dos preços de um dia para o outro')
if st.checkbox('Visualizar a decomposição'):
    df['variacao'] = df['Close'].diff()
    #st.table(df.tail())
    st.set_option('deprecation.showPyplotGlobalUse', False)
    #plot_comparacao('Date', 'Close', 'variacao', df, df, 'Analise comparativa da variação de preço ')
    st.pyplot(plot_comparacao('Date', 'Close', 'variacao', df, df, 'Analise comparativa da variação de preço '))


############################################################


#Pré-Processamento dos Dados: em formato de uma Série Temporal
df.index = df["Date"]
df = df["Close"]

#Normalização
dataset = df.values.reshape((-1, 1))
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

#Máquina Preditiva: Construção da LSTM
#Processo de modelagem
look_back = 5
future_target = 1

#predictions = model.predict(x_test[:5])
#st.write("\npredictions shape do model:", predictions.shape)
#predictions = model2.predict(x_test[:5])
#st.write("predictions shape do model 2:", predictions.shape)


#Divisão do dataset em treino e teste
tam = int(len(dataset) * 0.70)
dataset_teste = dataset[tam:]
dataset_treino = dataset[:tam]


#Padronização: Redes Neurais necessitam que os dados de input estejam na forma matricial,
# de preferência uma matriz tridimensional (sample, timestep, feature).
def process_data(data, look_back, forward_days, jump=1):
    X, Y = [], []
    for i in range(0, len(data) - look_back - forward_days + 1, jump):
        X.append(data[i:(i + look_back)])
        Y.append(data[(i + look_back):(i + look_back + forward_days)])
    return np.array(X), np.array(Y)


X, y = process_data(dataset_treino, look_back, future_target)
y = np.array([list(a.ravel()) for a in y])

x_test, y_test = process_data(dataset_teste, look_back, future_target)
y_test = np.array([list(a.ravel()) for a in y_test])

#Separação do conjunto de dados de teste e separar um porcentagem para validação
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.20, random_state=42)

#Definindo os números de neurônios por camada
n_first = 128
EPOCHS = 50
BATCH_SIZE = 2

############################################################
#adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
#opt = tf.keras.optimizers.RMSprop(learning_rate=0.001)
st.header('Máquina Preditiva com LSTM')
if st.checkbox('Clique para construir modelo 1'):
    # redirect where stdout goes, write to 'mystdout'
    # https://stackoverflow.com/a/1218951/2394542
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    #Construido o modelo
    model = Sequential()
    model.add(LSTM(n_first, input_shape=(look_back, 1)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.3))
    model.add(Dense(1))

    model.summary()
    sys.stdout = old_stdout
    st.text(mystdout.getvalue())

    model.compile(loss='mean_squared_error', optimizer='RMSprop')

    history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_validate, y_validate), shuffle=False, batch_size=BATCH_SIZE, verbose=2)

#Mostrar progresso


#for i in tqdm(range(EPOCHS)):

#epochs=50, data_size=None, batch_size=2, verbose=1, tqdm_class=tqdm_auto, **tqdm_kwargs


#Salvando os valores preditos
    prediction = model.predict(x_test)
    prediction_inverse = scaler.inverse_transform(prediction)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.figure(figsize=(15, 10))
    plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), c='g', label='Teste')
    plt.plot(scaler.inverse_transform(prediction.reshape(-1, 1)), c='b', label='Predito')
    plt.ylabel("Preço de Fechamento")
    plt.legend(loc='best')
    p1 = plt.show()

#st.pyplot(p1)

#model.save('modelo.h5')
#model_salvo = load_model('modelo.h5')
#model.get_config()

st.header('Máquina Preditiva com Stacked LSTM networks')
if st.checkbox('Clique para construir modelo 2'):
#Máquina Preditiva: Stacked Long-Short Term Memory networks

    # redirect where stdout goes, write to 'mystdout'
    # https://stackoverflow.com/a/1218951/2394542
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    model2 = Sequential()
    # input_shape : (X_train.shape[1] = timestep = 10, X_train.shape[2] = feature = 1)
    model2.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model2.add(Dropout(0.25))
    model2.add(LSTM(50, return_sequences=True))
    model2.add(Dropout(0.20))
    model2.add(LSTM(50))
    model2.add(Dropout(0.20))
    model2.add(Dense(1))

    model2.summary()
    sys.stdout = old_stdout
    st.text(mystdout.getvalue())

    model2.compile(optimizer='adam', loss='mean_squared_error')

    model2.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_validate, y_validate), shuffle=False, batch_size=BATCH_SIZE)

    prediction2 = model2.predict(x_test)
    prediction2_inverse = scaler.inverse_transform(prediction2)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.figure(figsize=(15, 10))
    plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), c='g', label='Teste')
    plt.plot(scaler.inverse_transform(prediction2.reshape(-1, 1)), c='r', label='Predito')
    plt.ylabel("Preço de Fechamento")
    plt.legend(loc='best')
    p2 = plt.show()
    #st.pyplot(p2)

#Salvando e carregando o modelo treinado
#model2.save('modelo2.h5')
#model2_salvo = load_model('modelo2.h5')

#Comparando os modelos da Máquina Preditiva
st.subheader('Comparação os modelos das Máquinas Preditivas')
if st.checkbox('model x model2'):
    st.set_option('deprecation.showPyplotGlobalUse', False)

    plt.figure(figsize=(15, 10))
    plt.plot(prediction_inverse, color='blue', label='Valor predicto: model')
    plt.plot(prediction2_inverse, color='red', label='Valor predicto: model 2')
    plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), c='g', label='Teste')
    plt.legend()
    pp = plt.show()
    st.pyplot(pp)
    st.write('Mean Squared Error model: ', mean_squared_error(y_test, prediction_inverse))
    st.write('Mean Squared Error model2: ', mean_squared_error(y_test, prediction2_inverse))

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#Avaliando o modelo da Máquina Preditiva
#Exponential Moving Average (EMA) análise de precificação de um ativo pela Média Móvel Exponencial
st.subheader('Avaliação dos modelos das Máquinas Preditivas')
if st.checkbox('Exponential Moving Average (EMA)'):
    window_size = 100
    N = dataset_treino.size

    run_avg_predictions = []
    run_avg_x = []

    mse_errors = []

    running_mean = 0.0
    run_avg_predictions.append(running_mean)

    decay = 0.5

    for pred_idx in range(1, N):

        running_mean = running_mean*decay + (1.0-decay)*dataset_treino[pred_idx-1]
        run_avg_predictions.append(running_mean)
        mse_errors.append((run_avg_predictions[-1]-dataset_treino[pred_idx])**2)
        run_avg_x.append(Date)

    st.write('MSE error para EMA averaging: %.5f' % (0.5*np.mean(mse_errors)))

if st.checkbox("Avaliação com 'evaluate'"):
    # Avaliação do model nos dados de teste usando 'evaluate'
    st.write("Model: usando evaluate nos dados de teste")
    results = model.evaluate(x_test, y_test, batch_size=128)
    st.write("test loss, test acc:", results)

    # Avaliação do model2 nos dados de teste usando 'evaluate'
    st.write("\nModel2: usando evaluate nos dados de teste")
    results2 = model2.evaluate(x_test, y_test, batch_size=128)
    st.write("test loss, test acc:", results2)

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
    (...)\n
    Esta página foi produzida com finalidade acadêmica - TCC Cássia Chin de curso ADS do IFSP. 
    """)
with st.beta_expander("Cookies"):
    st.write("""
    (...)
    \n
    """)


