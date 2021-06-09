# -*- coding: UTF-8 -*-
from scr.predictor import *
from scr.graph import graph_candlestick, graph_scatterplot, graph_compare_plot
from scr.layout import *
import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 16, 12
plt.style.use('ggplot')
import warnings
warnings.filterwarnings("ignore")
#import pyautogui

#StreamlitAPIException: set_page_config() can only be called once per app, and must be called as the first Streamlit command in your script.
try:
    st.set_page_config(
        page_title="LSTM WebApp for ETF",
        page_icon="chart_with_upwards_trend",
        #layout="wide",
        initial_sidebar_state="expanded",
        )
except:
    pass
    #pyautogui.hotkey('f5')


st.title('ANÁLISE DE ETF NO MERCADO COM ML')

disclaimer()
etf_ticker()
st_faq()

#footer()

hide_footer_style = """
<style>
.reportview-container .main footer {visibility: hidden;}    
"""
st.markdown(hide_footer_style, unsafe_allow_html=True)

#https://medium.com/analytics-vidhya/build-your-basic-machine-learning-web-app-with-streamlit-60e29e43f5f7

if __name__ == '__main__':

    controller = Predictor()
    data = pd.DataFrame()

    controller.data = controller.ticker_selector()
    #st.write(controller.data.columns)
    #st.write(controller.data.dtypes)

    if controller.data is not None:
        controller.show_raw_data(controller.data)

        st.subheader('Gráfico Candlestick')
        if st.button('abrir'):
            graph_candlestick(controller.data)

        st.subheader('Estatística Descritiva')
        if st.checkbox('veja Síntese'):
            st.write(controller.data.describe().round(5))

        st.subheader('Preço fechamento x Volume transacionado')
        if st.checkbox('gráfico Scatterplot'):
            try:
                graph_scatterplot(controller.data)
                st.write('O Coeficiente de correlação de Pearson entre o Preço fechamento e o Volume transacionado é de:  %f ' % (controller.data['Close'].corr(controller.data['Volume'])))
                #AttributeError: 'float' object has no attribute 'shape'
            except AttributeError:
                st.pyplot(graph_scatterplot(controller.data))
            finally:
                st.write('Temporariamente indisponível para o ticker selecionado')
                #pass

        st.subheader('Variação diária dos preços')
        if st.checkbox('visualizar a Decomposição'):
            st.set_option('deprecation.showPyplotGlobalUse', False)
            #graph_compare_plot('Date', 'Close', 'variacao', df, df, 'Analise comparativa da variação de preço ')
            try:
                controller.data['variacao'] = controller.data['Close'].diff()
                st.pyplot(graph_compare_plot('Date', 'Close', 'variacao', controller.data, controller.data, 'Análise comparativa da variação de preço '))
            #DataError: No numeric types to aggregate
            except:
                controller.data['Close'] = pd.to_numeric(controller.data['Close'], errors='coerce')
                controller.data['variacao'] = controller.data['Close'].diff()
                st.pyplot(graph_compare_plot('Date', 'Close', 'variacao', controller.data, controller.data, 'Análise comparativa da variação de preço '))


    #if controller.data is not None:
        st.subheader('Configure a divisão e modelagem dos dados')
        with st.form(key='split_data_form'):
            train_test = st.slider('Train-test split %', 1, 99, 80 )
            look_back = st.slider('Look back split', 1, 20, 5)
            future_target = st.slider('Future target split', 1, 5, 1)
            split_submit_button = st.form_submit_button(label='Submit')
            if split_submit_button:
                st.success('Configurado com sucesso!')

            #st.write('Parâmetros de prepare data: ', 'train_test: ', train_test, ', look_back: ',  look_back, ', future_target: ', future_target)

        controller.prepare_data(controller.data, train_test, look_back,future_target)
        #st.write('***** rodou prepare_data *****')
        X_train, y_train, X_validate, y_validate, x_test, y_test, scaler = controller.prepare_data (controller.data, train_test, look_back, future_target)
        #st.write(X_train)

        st.subheader('Configure os hiperparâmetros dos modelos:')
        controller.set_parameters()
        #st.write('***** rodou set_parameters *****')

        #try:
        st.header('Máquina Preditiva com LSTM networks')
        #predict_btn1 = st.button('Clique para construir modelo 1')
        #if predict_btn1:
        if st.checkbox('Clique para construir modelo 1'):
            controller.vanilla_lstm_predict(controller.neurons , look_back, controller.opt, controller.epochs, controller.batch_size)
            #st.write(controller.neurons , look_back, controller.opt, controller.epochs, controller.batch_size)
            #st.write('***** rodou vanilla_lstm_predict *****')
        #    graph_predict(controller.scaler, controller.y_test, controller.prediction)
            #st.write('***** rodou graph_predict model 1 *****')


        st.header('Máquina Preditiva com Stacked LSTM')
        # if st.button('Clique para construir modelo 2'):
        if st.checkbox('Clique para construir modelo 2'):
            controller.stacked_lstm_predict(look_back, controller.opt, controller.epochs, controller.batch_size)
            #st.write(controller.neurons , look_back, controller.opt, controller.epochs, controller.batch_size)
            #st.write('***** rodou stacked_lstm_predict *****')
            # graph_predict(controller.scaler, controller.y_test, controller.prediction)
            #st.write('***** rodou graph_predict model 2 *****')

            try:
                #st.dataframe(controller.prediction_inverse)
                #st.header('Comparação dos modelos das Máquinas Preditivas')
                #if st.checkbox('gráfico: modelo 1 x modelo 2'):
                controller.compare_models()
                #graph_compare_models(controller.prediction_inverse, controller.prediction2_inverse, controller.y_test, controller.scaler)
            #AttributeError: 'Predictor' object has no attribute 'prediction_inverse'
            except: # AttributeError as e:
                st.warning('Construa e treine os 2 modelos')


    #faq()
    #st_faq()
    #footer()

