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


#@st.cache(suppress_st_warning=True)
#@st.cache(hash_funcs={pd.DataFrame: lambda _: None})
#@st.cache(hash_funcs={FooType: hash})

def tratamento_dados(df):
    #Pré-Processamento dos Dados: em formato de uma Série Temporal
#def pre_processamento_dos_dados(df):
    df.index = df["Date"]
    df = df["Close"]

    #Normalização
#    def normalizacao(df):git
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

    #Divisão do dataset em treino e teste
#    def split_dataset(dataset):


    X, y = process_data(dataset_treino, look_back, future_target)
    y = np.array([list(a.ravel()) for a in y])

    x_test, y_test = process_data(dataset_teste, look_back, future_target)
    y_test = np.array([list(a.ravel()) for a in y_test])

    #Separação do conjunto de dados de teste e separar um porcentagem para validação
    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.20, random_state=42)
