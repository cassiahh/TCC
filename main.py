# -*- coding: UTF-8 -*-
from src.predictor import *
from src.graph import graph_candlestick, graph_scatterplot, graph_compare_plot
from src.layout import *

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 16, 12
plt.style.use('ggplot')
import warnings
warnings.filterwarnings("ignore")
#import pyautogui

#Page config
try:
    st.set_page_config(
        page_title="LSTM WebApp for ETF",
        page_icon="chart_with_upwards_trend",
        #layout="wide",
        initial_sidebar_state="expanded",
        )
except:
    #pyautogui.hotkey('f5')
    pass

st.title('LSTM WebApp for ETF')
st.subtitle('Análise de ETF no mercado com ML')

disclaimer()
etf_ticker()
st_faq()


# Hide default footer
hide_footer_style = """
<style>
.reportview-container .main footer {visibility: hidden;}    
"""
st.markdown(hide_footer_style, unsafe_allow_html=True)


if __name__ == '__main__':

    controller = Predictor()
    data = pd.DataFrame()

    controller.data = controller.ticker_selector()

    if controller.data is not None:
        controller.show_raw_data(controller.data)

        st.subheader('Análise gráfica de mercado')
        if st.checkbox('gráfico Candlestick'):
            candlestick = st.empty()
            candlestick.plotly_chart(graph_candlestick(controller.data))

        st.subheader('Estatística Descritiva')
        if st.checkbox('veja Síntese'):
            st.write(controller.data.describe().round(5))

        st.subheader('Preço fechamento x Volume transacionado')
        if st.checkbox('gráfico Scatterplot'):
            try:
                graph_scatterplot(controller.data)
                st.write('O Coeficiente de correlação de Pearson entre o Preço fechamento e o Volume transacionado é de:  %f ' % (controller.data['Close'].corr(controller.data['Volume'])))
            except AttributeError:
                st.pyplot(graph_scatterplot(controller.data))
            except:
                st.write('Temporariamente indisponível: reinsira o ticker')
                #pass

        st.subheader('Variação diária dos preços')
        if st.checkbox('visualizar a Decomposição'):
            st.set_option('deprecation.showPyplotGlobalUse', False)
            try:
                controller.data['variacao'] = controller.data['Close'].diff()
                st.pyplot(graph_compare_plot('Date', 'Close', 'variacao', controller.data, controller.data, 'Análise comparativa da variação de preço '))
            except:
                controller.data['Close'] = pd.to_numeric(controller.data['Close'], errors='coerce')
                controller.data['variacao'] = controller.data['Close'].diff()
                st.pyplot(graph_compare_plot('Date', 'Close', 'variacao', controller.data, controller.data, 'Análise comparativa da variação de preço '))

        #if controller.data is not None:
        st.subheader('Configure a divisão e modelagem dos dados')
        with st.form(key='split_data_form'):
            train_test = st.slider('Train-test split %', 1, 99, 80)
            look_back = st.slider('Look back split', 1, 20, 5)
            future_target = st.slider('Future target split', 1, 5, 1)
            split_submit_button = st.form_submit_button(label='Submit')
            if split_submit_button:
                st.success('Configurado com sucesso!')

        controller.prepare_data(controller.data, train_test, look_back, future_target)
        X_train, y_train, X_validate, y_validate, x_test, y_test, scaler = controller.prepare_data(controller.data, train_test, look_back, future_target)

        st.subheader('Configure os hiperparâmetros dos modelos:')
        controller.set_parameters()

        st.header('Máquina Preditiva com LSTM networks')
        predict_btn1 = st.button('Clique para construir modelo 1')
        if predict_btn1:
        #if st.checkbox('Clique para construir modelo 1'):
            controller.vanilla_lstm_predict(controller.neurons, look_back, controller.opt, controller.epochs, controller.batch_size)
            empty_model2 = st.empty()

            st.header('Máquina Preditiva com Stacked LSTM')
            empty_model2.checkbox('Clique para construir modelo 2')
            empty_model2.checkbox('Clique para construir modelo 2', True)
        #if st.checkbox('Clique para construir modelo 2'):
            controller.stacked_lstm_predict(look_back, controller.opt, controller.epochs, controller.batch_size)
            try:
                #if st.checkbox('gráfico: modelo 1 x modelo 2'):
                controller.compare_models()
            except:  # AttributeError:
                st.warning('Construa e treine os 2 modelos')


            #footer()
