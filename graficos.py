#from scratch import *

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
import plotly.graph_objs as go
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 16, 10
plt.style.use('ggplot')

import warnings
warnings.filterwarnings("ignore")


#class Graficos:

#  Gráfico Candlestick
@st.cache(suppress_st_warning=True)
def grafico_candlestick(df, acao):
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
#    st.write(c)
    st.write('Gráfico aberto em uma nova aba para melhor visualização', c)
    # st.write('Below is a DataFrame:', data_frame, 'Above is a dataframe.')
    # verificar como retirar mensagem "None"

# Gráfico Scatterplot
@st.cache(suppress_st_warning=True)
def grafico_scatterplot(df):
    fig = plt.figure(figsize=(16, 10))
    sns.scatterplot(x="Close", y="Volume", data=df)
    plt.xlabel('Preço fechamento', fontsize=12)
    plt.ylabel('Volume transacionado', fontsize=12)
    plt.title("Preço fechamento X Volume transacionado")
    st.pyplot(fig)
    st.write('O Coeficiente de correlação de Pearson entre o Preço fechamento e o Volume transacionado é de: ')
    st.write(df['Close'].corr(df['Volume']))


# Statsmodels
""" 
def grafico_statsmodels(df):
    resultado = seasonal_decompose(df['Close'], freq=35)
    ax = resultado.plot()
"""

# Função de plot
"""
def plotar (titulo, labelx, labely, x, y, dataset):
    sns.set_palette('Accent')
    sns.set_style('darkgrid')
    ax = sns.lineplot(x=x, y=y, data=dataset)
    ax.figure.set_size_inches(12,6)
    ax.set_title(titulo, loc='left', fontsize=18)
    ax.set_xlabel(labelx, fontsize=14)
    ax.set_ylabel(labely, fontsize=14)
    ax = ax
"""


# Função de plot de graficos para comparação
def plot_comparacao(x, y1, y2, dataset1, dataset2, titulo):
    plt.figure(figsize=(16, 10))
    ax = plt.subplot(2, 1, 1)
    ax.set_title(titulo, fontsize=18, loc='left')
    sns.lineplot(x=x, y=y1, data=dataset1)
    plt.subplot(2, 1, 2)
    sns.lineplot(x=x, y=y2, data=dataset2)
 #   ax = ax

@st.cache(suppress_st_warning=True)
def decomposicao(df):
    df['variacao'] = df['Close'].diff()
    #st.table(df.tail())
    st.set_option('deprecation.showPyplotGlobalUse', False)
    #plot_comparacao('Date', 'Close', 'variacao', df, df, 'Analise comparativa da variação de preço ')
    st.pyplot(plot_comparacao('Date', 'Close', 'variacao', df, df, 'Análise comparativa da variação de preço '))


# Gráfico do resultado da função perda por epochs
"""
def grafico_perda_por_epochs(history):
    plt.figure(figsize = (16,10))
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Perda por épocas')
    plt.xlabel('épocas')
    plt.ylabel('perda')
    plt.legend(['treino (loss)', 'validação (val_loss)'], loc='best',fontsize=15)
    plt.show()
"""

# Gráfico dos valores preditos
def grafico_dos_valores_preditos(prediction, y_test, scaler):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.figure(figsize = (16,10))
    plt.plot(scaler.inverse_transform(y_test.reshape(-1,1)),c='g', label='Teste')
    plt.plot(scaler.inverse_transform(prediction.reshape(-1,1)), c='b',label='Predito')
    plt.ylabel("Preço de Fechamento")
    plt.legend(loc='best')
    p = plt.show()
    st.pyplot(p)


# Gráfico comparativo: model x model2
def grafico_compara_models(prediction_inverse, prediction2_inverse, y_test, scaler):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.figure(figsize = (16,10))
    plt.plot(prediction_inverse, color='blue', label='Valor predicto: model')
    plt.plot(prediction2_inverse, color='red', label='Valor predicto: model 2')
    plt.plot(scaler.inverse_transform(y_test.reshape(-1,1)),c='g', label='Teste')
    plt.legend()
    mm = plt.show()
    st.pyplot(mm)
    print('Mean Squared Error model: ', mean_squared_error(y_test, prediction_inverse))
    print('Mean Squared Error model2: ', mean_squared_error(y_test, prediction2_inverse))
