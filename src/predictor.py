# -*- coding: UTF-8 -*-
from src.graph import graph_predict, graph_compare_models

import streamlit as st
#from streamlit.caching import cache
import pandas as pd
from pandas_datareader import data as web
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from tensorflow.python.keras.optimizer_v2 import adam, rmsprop
from tensorflow.keras.optimizers import Adam, RMSprop
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, LeakyReLU
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from io import StringIO
#from datetime import date, tzinfo, timezone, datetime
import sys
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 16, 12
plt.style.use('ggplot')
import warnings
warnings.filterwarnings("ignore")


# Main Predicor class
class Predictor:
    data = pd.DataFrame()

    def ticker_selector(self):
        try:
            form = st.form(key='ticker_form')
            ticker = form.text_input('Insira um código de ETF')
            submit = form.form_submit_button('Inserir')
            global data
            if not ticker:
                st.warning('Favor inserir um código de ETF.')
                st.stop()
            if submit:
                ticker = ticker.strip().upper()
                #web scraping
                st.write(f'Código do ETF inserido: {ticker}')
                data = web.DataReader(ticker, data_source='yahoo', start='01-01-2017')
                #fill NaN based on the previous value
                data['Close'] = data['Close'].ffill()
                data = data.bfill(axis=1)
            return data
        except:
            #def file_selector(self):
            if st.checkbox('alternativa: se série temporal momentaneamente indisponível on-line'):
                file = st.file_uploader("Insira um arquivo CSV", type="csv")
                if file is not None:
                    data = pd.read_csv(file)
                    data['Date'] = pd.to_datetime(data['Date'])
                    data = data.sort_values('Date')
                    data.set_index('Date', inplace=True)
                    #fill NaN based on the previous value
                    data['Close'] = data['Close'].ffill()
                    data = data.bfill(axis=1)
                    return data
                else:
                    st.text("Fazer upload de um arquivo csv")
                    st.markdown('''
      <a href="https://finance.yahoo.com/etfs"  target=_blank>
          Pesquise um ticker e em 'Historical Data' faça download do dataset.
      </a>''', unsafe_allow_html=True)

    @st.cache
    def data(self):
        return self.data

    def show_raw_data(self, data):
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        with st.beta_container():
            st.subheader('Escolha um modo de exibição dos dados brutos')
            chosen = st.radio(
                'Show raw data',
                ("Não exibir", "Tail", "Head", "Dataframe"))
            if chosen == "Não exibir":
                st.write(f"Opção escolhida é: {chosen}")
            if chosen == "Tail":
                st.subheader('Tail')
                st.dataframe(self.data.tail())
            if chosen == "Head":
                st.subheader('Head')
                st.dataframe(self.data.head())
            if chosen == "Dataframe":
                st.subheader('Tabela')
                st.dataframe(self.data)

    # Data preparation
    #@st.cache(suppress_st_warning=True)
    def prepare_data(self, df, train_test=80, look_back=5, future_target=1):
        try:
            df.index = df['Date']
            df.drop('Date', inplace=True, axis=1)
            df = df['Close']
        except KeyError:
            df = df['Close']

        # Normalizing the data: Data scaling - preprocessing
        dataset = df.values.reshape((-1, 1))
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = self.scaler.fit_transform(dataset)

        # Divisão do dataset em treino e teste
        tam = int(len(dataset) * (train_test/100))
        dataset_teste = dataset[tam:]
        dataset_treino = dataset[:tam]

        # Processo de modelagem
        self.look_back = look_back
        self.future_target = future_target

        #  formatação dos dados de input na forma matricial
        @st.cache
        def process_data(data, look_back, forward_days, jump=1):
            X, Y = [], []
            for i in range(0, len(data) - look_back - forward_days + 1, jump):
                X.append(data[i:(i + look_back)])
                Y.append(data[(i + look_back):(i + look_back + forward_days)])
            return np.array(X), np.array(Y)

        X, y = process_data(dataset_treino, look_back, future_target)
        y = np.array([list(a.ravel()) for a in y])
        self.x_test, self.y_test = process_data(dataset_teste, look_back, future_target)
        self.y_test = np.array([list(a.ravel()) for a in self.y_test])
        #Separação do conjunto de dados de teste e separar um porcentagem para validação
        self.X_train, self.X_validate, self.y_train, self.y_validate = train_test_split(X, y, test_size=(1-train_test/100), random_state=42)

        return self.X_train, self.X_validate, self.y_train, self.y_validate, self.x_test, self.y_test, self.scaler


    def set_parameters(self):
        if self.data is not None:
            with st.form(key='predict_form'):
                self.neurons = int(st.selectbox('Unidades na camada LSTM: quantidade de neurônios na camada oculta', (1, 32, 64, 128, 256), 3))
                self.chosen_optimizer = st.selectbox('Otimizador', ('Adam', 'RMSprop'))
                #self.learning_rate = float(st.selectbox('Learning Rate: Taxa de Aprendizagem', (0.0001, 0.001, 0.01, 0.1), 1))
                self.epochs = int(st.slider('Épocas: quantidade de passagens completas do conjunto de dados', 1, 200, 30))
                self.batch_size = int(st.selectbox('Batch Size: quantidade de amostras em cada lote durante o treinamento e teste', (1, 2, 4, 8, 16, 32, 64, 128, 256), 1))

                if self.chosen_optimizer == 'Adam':
                    self.opt = 'adam'
                elif self.chosen_optimizer == 'RMSprop':
                    self.opt = 'rmsprop'
                predict_submit_button = st.form_submit_button(label='Submit')
                if predict_submit_button:
                    st.success('Configurado com sucesso!')
                #st.write('Hiperparâmetros do modelo: ', 'neurons: ',self.neurons, ', optimizer: ', self.opt, ', epochs: ', self.epochs, ', batch size: ', self.batch_size )
        return self.neurons, self.opt, self.epochs, self.batch_size

    def vanilla_lstm_predict(self, neurons=128, look_back=5, optimizer='Adam', epochs=30, batch_size=2):
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        model = Sequential()
        #model.add(LSTM(units, activation='tanh', input_shape=(n_steps, n_features)))
        model.add(LSTM(self.neurons, input_shape=(self.look_back, 1)))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.3))
        model.add(Dense(units=1))

        #exibir sumario do modelo
        model.summary()

        sys.stdout = old_stdout
        st.text(mystdout.getvalue())

        #@st.cache(hash_funcs={keras.utils.object_identity.ObjectIdentityDictionary: my_hash_func})
        def train_model1(self):
            model.compile(loss='mean_squared_error', optimizer=self.opt)  #, metrics=['accuracy']
            #history = model.fit(self.X_train, self.y_train, epochs=epochs, validation_data=(self.X_validate, self.y_validate), shuffle=False, batch_size=batch_size, verbose=2)
            self.model = model.fit(self.X_train, self.y_train, epochs=self.epochs, validation_data=(self.X_validate, self.y_validate), shuffle=False, batch_size=self.batch_size, verbose=2)
            results = model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size)
            #Salvando os valores preditos
            self.prediction = model.predict(self.x_test)
            self.prediction_inverse = self.scaler.inverse_transform(self.prediction)
            #return model_compile, history, self.prediction, self.prediction_inverse
            #self.prediction_inverse = prediction_inverse
            return self.model, results, self.prediction, self.prediction_inverse

        with st.spinner('Treinando modelo 1…'):
            train_model1(self)

        model, results, prediction, prediction_inverse = train_model1(self)

        st.success('Model Training Complete!')

        #if st.checkbox('gráfico: modelo 1'):
        graph_predict(scaler=self.scaler, y_test=self.y_test, prediction=self.prediction)

        with st.beta_expander("1. Hiperparâmetros"):
            st.write("\nÉpocas: ", self.epochs)
            st.write("\nBach-Size: ", self.batch_size)
            st.write("\nOtimizador: ", self.chosen_optimizer)
            st.write("\nLearning Rate: 0.001")
            st.write("\nActivation: 'tanh'")
            #AttributeError: 'History' object has no attribute 'predict'
            # predictions = model.predict(self.x_test[:5])
            # st.write("\nPredictions shape do modelo 1: (look_back, future_target)", predictions.shape)

        try:
            st.write("Avaliação com 'evaluate' (Scalar test loss):", results)

            porcentagem = (((self.prediction_inverse[-1]-self.prediction_inverse[-2])/self.prediction_inverse[-2])*100)
            #st.write ('pos D-1: ',prediction_inverse[-9], 'pos D: ', prediction_inverse[-10])
            if self.prediction_inverse[-1] > self.prediction_inverse[-2]:
                st.write('Segundo modelo 1, o valor vai subir em torno de %.6f %%' % (float(porcentagem)))
            else:
                st.write('Segundo modelo 1, o valor vai cair em torno de %.6f %%' % (float(-porcentagem)))
            #ValueError: y_true and y_pred have different number of output (2!=1)
            eqm1 = mean_squared_error(self.y_test, self.prediction_inverse)
            st.write('Erro Quadrático Médio (Mean squared error) modelo 1: ', eqm1)
        except ValueError:
            pass

        return self.prediction, self.prediction_inverse, model

        #st.header('Máquina Preditiva com Stacked LSTM networks')
        #if st.checkbox('Clique para construir modelo 2'):
        #Máquina Preditiva: Stacked Long-Short Term Memory networks

    def stacked_lstm_predict(self, look_back=5, optimizer='Adam', epochs=30, batch_size=2):
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        model2 = Sequential()
        # input_shape : (X_train.shape[1] = timestep = 10, X_train.shape[2] = feature = 1)
        #model.add(LSTM(units, activation='tanh' / activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
        model2.add(LSTM(100, return_sequences=True, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model2.add(Dropout(0.25))
        model2.add(LSTM(50, return_sequences=True))
        model2.add(Dropout(0.20))
        model2.add(LSTM(50))
        model2.add(Dropout(0.20))
        model2.add(Dense(1))

        model2.summary()

        sys.stdout = old_stdout
        st.text(mystdout.getvalue())

        #@st.cache(hash_funcs={keras.utils.object_identity.ObjectIdentityDictionary: my_hash_func})
        def train_model2(self):
            model2.compile(optimizer=self.opt, loss='mean_squared_error')
            self.model2 = model2.fit(self.X_train, self.y_train, epochs=epochs, validation_data=(self.X_validate, self.y_validate), shuffle=False, batch_size=batch_size)
            results2 = model2.evaluate(self.x_test, self.y_test, batch_size=self.batch_size)
            self.prediction2 = model2.predict(self.x_test)
            #Reverse variable data scaling: Going back to the initial data scale
            self.prediction2_inverse = self.scaler.inverse_transform(self.prediction2)
            return self.model2, results2, self.prediction2, self.prediction2_inverse

        with st.spinner('Treinando modelo 2…'):
            train_model2(self)

        model2, results2, prediction2, prediction2_inverse = train_model2(self)

        st.success('Model Training Complete!')

    #if st.checkbox('gráfico: modelo 2'):
        graph_predict(scaler=self.scaler, y_test=self.y_test, prediction=self.prediction2)

        with st.beta_expander("2. Hiperparâmetros"):
            st.write("\nÉpocas: ", self.epochs)
            st.write("\nBach-Size: ", self.batch_size)
            st.write("\nOtimizador: ", self.chosen_optimizer)
            st.write("\nLearning Rate: 0.001")
            st.write("\nActivation: 'tanh'")
            st.write("\nUnidades de neurônios nas camadas de LSTM: 100, 50, 50")
            #AttributeError:
            # predictions = model2.predict(self.x_test[:5])
            # st.write("\nPredictions shape do modelo 1: (look_back, future_target)", predictions.shape)

        try:
            st.write("Avaliação com 'evaluate' (Scalar test loss):", results2)
            porcentagem = (((self.prediction2_inverse[-1]-self.prediction2_inverse[-2])/self.prediction2_inverse[-9])*100)
            if self.prediction2_inverse[-1] > self.prediction2_inverse[-2]:
                st.write('Segundo modelo 2, o valor vai subir em torno de %.6f %%' % (float(porcentagem)))
            else:
                st.write('Segundo modelo 2, o valor vai cair em torno de %.6f %%' % (float(-porcentagem)))
            eqm2 = mean_squared_error(self.y_test, self.prediction2_inverse)
            st.write('Erro Quadrático Médio (Mean squared error) modelo 2: ', eqm2)
        except ValueError:
            pass

    #Comparando os modelos da Máquina Preditiva
    #st.header('Comparação dos modelos das Máquinas construidas')
    #@st.cache(hash_funcs={keras.utils.object_identity.ObjectIdentityDictionary: my_hash_func})
    def compare_models(self):
        st.header('Gráfico: modelo 1 x modelo 2')
        graph_compare_models(self.prediction_inverse, self.prediction2_inverse, self.y_test, self.scaler)

        # with st.beta_container():
        #     col1, col2 = st.beta_columns(2)

        #     with col1:
        #         st.subheader("Modelo 1: LSTM")
        #         st.write("\nUnidades: ", neurons)
        #         st.write("\nÉpocas: ", epochs)
        #         st.write("\nBach-Size: ", batch_size)
        #         st.write("\nOtimizador: ", optimizer))
        #         predictions = model.predict(x_test[:5])
        #         st.write("\npredictions shape do modelo 1: (look_back, future_target)", predictions.shape)
        #         results = model.evaluate(x_test, y_test, batch_size=128)
        #         st.write("Avaliação com 'evaluate' (Scalar test loss):", results)
        #         st.write("Erro Quadrático Médio: ", eqm1)

        #     with col2:
        #         st.subheader("Modelo 2: Stacked LSTM")
        #         st.write("\nÉpocas: ", epochs)
        #         st.write("\nBach-Size: ", batch_size)
        #         st.write("\nOtimizador: ", optimizer)
        #         predictions2 = model2.predict(x_test[:5])
        #         st.write("predictions shape do modelo 2: (look_back, future_target)", predictions2.shape)
        #         results2 = model2.evaluate(x_test, y_test, batch_size=128)
        #         st.write("Avaliação com 'evaluate' (Scalar test loss):", results2)
        #         st.write("Erro Quadrático Médio: ", eqm2)

        #except (Exception) as e:
        #    st.markdown('<span style="color:red">Atenção: Crie primeiro as máquinas preditivas</span>', unsafe_allow_html=True)
