from processa_dados import *
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


#Definindo os números de neurônios por camada
n_first = 128
EPOCHS = 50
BATCH_SIZE = 2

#adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
#opt = tf.keras.optimizers.RMSprop(learning_rate=0.001)

#Máquina Preditiva com LSTM'
def lstm_predict(X_train, y_train, X_validate, y_validate, x_test, scaler ):
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
#Salvando os valores preditos
    prediction = model.predict(x_test)
    prediction_inverse = scaler.inverse_transform(prediction)
#Mostrar progresso


#for i in tqdm(range(EPOCHS)):

#epochs=50, data_size=None, batch_size=2, verbose=1, tqdm_class=tqdm_auto, **tqdm_kwargs


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





