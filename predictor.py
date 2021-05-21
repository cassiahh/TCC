import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import sys
from pandas.errors import ParserError
import time
import altair as altpi
import matplotlib.cm as cm
import graphviz
import base64
from bokeh.io import output_file, show
from bokeh.layouts import column
from bokeh.layouts import layout
from bokeh.plotting import figure
from bokeh.models import Toggle, BoxAnnotation
from bokeh.models import Panel, Tabs
from bokeh.palettes import Set3

# Keras specific
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time

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



# Predictor class
class Predictor:
   # Data preparation part, it will automatically handle with your data
    def prepare_data(self):

        data = self.data["Close"]
        data.index = self.data["Date"].date()
        st.table(data)

#        # Impute fill NaN based on the previous value of another cell
        #data.dropna(axis=0,inplace=True)
        data['close'] = data['close'].ffill()
        data = data.bfill(axis=1)
        st.table(data)
# """
       # Set target column
# #        self.chosen_target = st.sidebar.selectbox("Please choose target column", ('Close', 'Adj Close'))
#         self.chosen_target = data["Close"]

        #Normalização
        dataset = data.values.reshape((-1, 1))
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)



    #Divisão do dataset em treino e teste
        tam = int(len(dataset) * train_test)
        dataset_teste = dataset[tam:]
        dataset_treino = dataset[:tam]

        #Processo de modelagem
        look_back = 5
        future_target = 1

       # Train test split
        def process_data(data, look_back, forward_days, jump=1):
            X, Y = [], []
            for i in range(0, len(data) - look_back - forward_days + 1, jump):
                X.append(data[i:(i + look_back)])
                Y.append(data[(i + look_back):(i + look_back + forward_days)])
            return np.array(X), np.array(Y)

        try:
            X, y = process_data(dataset_treino, look_back, future_target)
            y = np.array([list(a.ravel()) for a in y])

            self.x_test, self.y_test = process_data(dataset_teste, look_back, future_target)
            self.y_test = np.array([list(a.ravel()) for a in self.y_test])

            self.X_train, self.X_validate, self.y_train, self.y_validate = train_test_split(X, y, test_size=(1 - train_test/100), random_state=42)
#            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=(1 - train_test/100), random_state=42)
        except:
            st.markdown('<span style="color:red">With this amount of data and split size the train data will have no records, <br /> Please change reduce and split parameter <br /> </span>', unsafe_allow_html=True)


   # Classifier type and algorithm selection: neural_network
    def set_neural_network(self):
        self.chosen_neural_network = st.sidebar.selectbox("Please choose a Algorithm type", ('LSTM', 'Stacked LSTM'))
        # Defining Hyperparameters
        if self.chosen_neural_network == 'LSTM':
            self.epochs = st.sidebar.slider('number of epochs', 1, 100, 20)
            # Definindo os números de neurônios por camada
            self.n_first = 128
            self.learning_rate = float(st.sidebar.text_input('learning rate:', '0.001'))
            # Otimizador dos pesos no treinamento da rede neural: Adam ou RMSProp
            # batch_size = 500 # Number of samples in a batch
            # Defining Hyperparameters
        elif self.chosen_neural_network == 'Stacked LSTM':
            self.epochs = st.sidebar.slider('number of epochs', 1, 100, 20)
            self.learning_rate = float(st.sidebar.text_input('learning rate:', '0.001'))
            self.n_first = int(st.sidebar.text_input('quantidade de loops', '128'))
            self.batch_size = int(st.sidebar.text_input('Number of samples in a batch', '2'))
            self.optimizer = st.sidebar.selectbox("Otimizador", ("Adam", "RMSProp"))
            # Otimizador dos pesos no treinamento da rede neural: Adam ou RMSProp
            # batch_size = 500 # Number of samples in a batch

   # Model training and predicitons
    def predict(self, predict_btn):


        if self.chosen_neural_network=='LSTM':
            model = Sequential()
            model.add(LSTM(self.n_first,input_shape = (look_back,1)))
            model.add(LeakyReLU(alpha=0.3))
            model.add(Dropout(0.3))
            model.add(Dense(1))


            # optimizer = keras.optimizers.SGD(lr=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss= "mean_squared_error" , optimizer='adam', metrics=["mean_squared_error"])
            self.model = model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=2)
            self.prediction = model.predict(self.X_test)
            self.prediction_train = model.predict(self.X_train)
            self.prediction_inverse = self.scaler.inverse_transform(self.prediction)


        elif self.chosen_neural_network=='Stacked LSTM':
            model2 = Sequential()
            # input_shape : (X_train.shape[1] = timestep = 10, X_train.shape[2] = feature = 1)
            model2.add(LSTM(100, return_sequences=True, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
            model2.add(Dropout(0.25))
            model2.add(LSTM(50, return_sequences=True))
            model2.add(Dropout(0.20))
            model2.add(LSTM(50))
            model2.add(Dropout(0.20))
            model2.add(Dense(1))

#           optimizer = tf.keras.optimizers.SGD(lr=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
            model2.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
            self.model2 = model2.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size)

            self.prediction2 = model2.predict(self.X_test)
            self.prediction2_train = model2.predict2_classes(self.X_train)
            self.prediction2_inverse = self.scaler.inverse_transform(self.prediction2)



        result = pd.DataFrame(columns=['Actual', 'Actual_Train', 'Prediction', 'Prediction_Train'])
        result_train = pd.DataFrame(columns=['Actual_Train', 'Prediction_Train'])
        result['Actual'] = self.y_test
        result_train['Actual_Train'] = self.y_train
        result['Prediction'] = self.prediction
        result_train['Prediction_Train'] = self.prediction_train
        result.sort_index()
        self.result = result
        self.result_train = result_train

        return self.prediction, self.prediction_train, self.result, self.result_train

   # Get the result metrics of the model
    def get_metrics(self):
        self.error_metrics = {}
        self.error_metrics['MSE_test'] = mean_squared_error(self.y_test, self.prediction)
        self.error_metrics['MSE_train'] = mean_squared_error(self.y_train, self.prediction_train)
        return st.markdown('### MSE Train: ' + str(round(self.error_metrics['MSE_train'], 3)) +
        ' -- MSE Test: ' + str(round(self.error_metrics['MSE_test'], 3)))

        self.error_metrics['Accuracy_test'] = accuracy_score(self.y_test, self.prediction)
        self.error_metrics['Accuracy_train'] = accuracy_score(self.y_train, self.prediction_train)
        return st.markdown('### Accuracy Train: ' + str(round(self.error_metrics['Accuracy_train'], 3)) +
        ' -- Accuracy Test: ' + str(round(self.error_metrics['Accuracy_test'], 3)))

   # Plot the predicted values and real values
    def plot_result(self):

        output_file("slider.html")

        s1 = figure(plot_width=800, plot_height=500, background_fill_color="#fafafa")
        s1.circle(self.result_train.index, self.result_train.Actual_Train, size=12, color="Black", alpha=1, legend_label = "Actual")
        s1.triangle(self.result_train.index, self.result_train.Prediction_Train, size=12, color="Red", alpha=1, legend_label = "Prediction")
        tab1 = Panel(child=s1, title="Train Data")

        if self.result.Actual is not None:
            s2 = figure(plot_width=800, plot_height=500, background_fill_color="#fafafa")
            s2.circle(self.result.index, self.result.Actual, size=12, color=Set3[5][3], alpha=1, legend_label = "Actual")
            s2.triangle(self.result.index, self.result.Prediction, size=12, color=Set3[5][4], alpha=1, legend_label = "Prediction")
            tab2 = Panel(child=s2, title="Test Data")
            tabs = Tabs(tabs=[ tab1, tab2 ])
        else:

            tabs = Tabs(tabs=[ tab1])

        st.bokeh_chart(tabs)


    # selector module for web app
    def acao_selector(self):
        data = pd.DataFrame()

        form = st.form(key='acao_form')
        acao = form.text_input('Insira um código de ETF (exemplo: BOVA11.SA)')
        submit = form.form_submit_button('Inserir')
        if not acao:
            st.warning('Favor inserir um código de ETF.')
            st.stop()
       # scraping (Remote Data Access) fazer validação de input
        #https://pandas-datareader.readthedocs.io/en/latest/remote_data.html
        if submit:
            acao = acao.strip().upper()
            st.write(f'Código do ETF inserido: {acao}')

# date = st.sidebar.date_input('start date', datetime.date(2011,1,1))
# st.write(date)
# start = datetime.datetime(2010, 1, 1)
# end = datetime.datetime(2017, 1, 11)
# df = web.DataReader("AAPL", 'yahoo', start, end)
# df.tail()
#data = web.DataReader(acao, data_source='yahoo', start='01-01-2017')
            data = web.DataReader(acao, data_source='yahoo', start=self.start_date)
            st.dataframe(self.data.tail())
        #    data.drop('Adj Close', inplace=True, axis=1)
        # if st.checkbox('Show raw data'):
        #     st.subheader('Escolha um modo de exibição dos dados brutos')
        #         #st.write(controller.data)
        #         # transformar em função e usar @st.cache
        #     chosen = st.radio(
        #         'raw data',
        #         ("Não exibir", "Tail", "Head", "Dataframe"))
        #     if chosen == "Não exibir":
        #         st.write(f"Opção escolhida é: {chosen}")
        #     if chosen == "Tail":
        #         st.subheader('Tail')
        #         st.dataframe(self.data.tail())
        #     if chosen == "Head":
        #         st.subheader('Head')
        #         st.dataframe(self.data.head())
        #     if chosen == "Dataframe":
        #         st.subheader('Tabela')
        #         st.dataframe(self.data)
        return data


if __name__ == '__main__':
    controller = Predictor()
    try:
        controller.data = controller.acao_selector()

        if controller.data is not None:
        #   split_data = st.sidebar.slider('Randomly reduce data size %', 1, 100, 10 )
            form = st.form(key='predict_form')
# st.write(date)
# start = datetime.datetime(2010, 1, 1)
# end = datetime.datetime(2017, 1, 11)
# df = web.DataReader("AAPL", 'yahoo', start, end)
# df.tail()
#            split_data = st.sidebar.slider('Reduce data size %', 1, 100, 100)
            start_date = form.sidebar.date_input('start date', datetime.date('01-01-2017'))
            train_test = form.sidebar.slider('Train-test split %', 1, 99, 20)
            look_back = form.sidebar.slider('Look back split %', 1, 50, 5)
            future_target = form.sidebar.slider('Future target split %', 1, 10, 1)

            # #Divisão do dataset em treino e teste
            # tam = int(len(dataset) * train_test)
            # dataset_teste = dataset[tam:]
            # dataset_treino = dataset[:tam]

#            controller.set_feature()
#        if len(controller.feature) == 1:
            controller.prepare_data(train_test)
            controller.set_neural_network()
            predict_btn = form.sidebar.button('Predict')
    except (AttributeError, ParserError, KeyError) as e:
        st.markdown('<span style="color:blue">WRONG FILE TYPE</span>', unsafe_allow_html=True)


#    if controller.data is not None:
        if predict_btn:
            st.sidebar.text("Progress:")
            my_bar = st.sidebar.progress(0)
            prediction, prediction_train, result, result_train = controller.predict(predict_btn)
            for percent_complete in range(100):
                my_bar.progress(percent_complete + 1)

            controller.get_metrics()
            controller.plot_result()
            controller.print_table()


#    if controller.data is not None:
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        if st.checkbox('Show raw data'):
            st.subheader('Escolha um modo de exibição dos dados brutos')
                #st.write(controller.data)
                # transformar em função e usar @st.cache
            chosen = st.radio(
                'raw data',
                ("Não exibir", "Tail", "Head", "Dataframe"))
            if chosen == "Não exibir":
                st.write(f"Opção escolhida é: {chosen}")
            if chosen == "Tail":
                st.subheader('Tail')
                st.dataframe(controller.data.tail())
            if chosen == "Head":
                st.subheader('Head')
                st.dataframe(controller.data.head())
            if chosen == "Dataframe":
                st.subheader('Tabela')
                st.dataframe(controller.data)


    def print_table(self):
        if len(self.result) > 0:
           result = self.result[['Actual', 'Prediction']]
           st.dataframe(result.sort_values(by='Actual',ascending=False).style.highlight_max(axis=0))


    # def set_feature(self):
    #     self.feature = self.chosen_target
    #     st.write('Sugestão preço de fechamento: Close')
    #     st.write('You selected:', self.feature)

