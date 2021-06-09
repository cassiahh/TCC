# -*- coding: UTF-8 -*-

import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import seaborn as sns
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 16, 12
plt.style.use('ggplot')
import warnings
warnings.filterwarnings("ignore")


#  Gráfico Candlestick
@st.cache(suppress_st_warning=True)
def graph_candlestick(data):
    candlestick = {
        'x': data.index,
        'open': data.Open,
        'close': data.Close,
        'high': data.High,
        'low': data.Low,
        'type': 'candlestick',
        'showlegend': False
    }
    data = [candlestick]
    layout = go.Layout()

    fig = go.Figure(data=data, layout=layout)
    result = fig.show()
    st.write('Gráfico aberto em uma nova aba para melhor visualização')


# Gráfico Scatterplot
@st.cache(suppress_st_warning=True)
def graph_scatterplot(data):
    fig = plt.figure(figsize=(16, 12))
    sns.scatterplot(x="Close", y="Volume", data=data)
    plt.xlabel('Preço fechamento', fontsize=12)
    plt.ylabel('Volume transacionado', fontsize=12)
    plt.title("Preço fechamento X Volume transacionado")
    st.pyplot(fig)
    #AttributeError: 'float' object has no attribute 'shape'
    #st.write('O Coeficiente de correlação de Pearson entre o Preço fechamento e o Volume transacionado é de:  %f ' % (data['Close'].corr(data['Volume'])))
    #return fig

@st.cache
def graph_plot(titulo, labelx, labely, x, y, dataset):
    ax = sns.lineplot(x=x, y=y, data=dataset)
    ax.figure.set_size_inches(12,6)
    ax.set_title(titulo, loc='left', fontsize=18)
    ax.set_xlabel(labelx, fontsize=12)
    ax.set_ylabel(labely, fontsize=12)

# Função de plot de graficos para comparação
@st.cache
def graph_compare_plot(x, y1, y2, dataset1, dataset2, titulo):
    plt.figure(figsize=(16, 12))
    ax = plt.subplot(2, 1, 1)
    ax.set_title(titulo, fontsize=18, loc='left')
    sns.lineplot(x=x, y=y1, data=dataset1)
    plt.subplot(2, 1, 2)
    sns.lineplot(x=x, y=y2, data=dataset2)


@st.cache(suppress_st_warning=True)
def decomposicao(data):
    data['variacao'] = data['Close'].diff()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    #plot_comparacao('Date', 'Close', 'variacao', df, df, 'Analise comparativa da variação de preço ')
    st.pyplot(graph_compare_plot('Date', 'Close', 'variacao', data, data, 'Análise comparativa da variação de preço '))


# Gráfico dos valores preditos
@st.cache(suppress_st_warning=True)
def graph_predict(scaler, y_test, prediction):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.figure(figsize=(16, 12))
    plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), c='g', label='Teste')
    plt.plot(scaler.inverse_transform(prediction.reshape(-1, 1)), c='b', label='Predito')
    plt.ylabel("Preço de Fechamento")
    plt.legend(loc='best')
    m = plt.show()
    st.pyplot(m)


# Gráfico comparativo: model x model2
@st.cache(suppress_st_warning=True)
def graph_compare_models(prediction_inverse, prediction2_inverse, y_test, scaler):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.figure(figsize=(16, 12))
    plt.plot(prediction_inverse, color='blue', label='Valor predicto: model')
    plt.plot(prediction2_inverse, color='red', label='Valor predicto: model 2')
    plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), c='g', label='Teste')
    plt.legend()
    mm = plt.show()
    st.pyplot(mm)
