import streamlit as st
import pandas as pd
from pandas_datareader import data as web
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

class Predictor:



    # def __init__(self, data):
    #     self.__data = data
    #
    # @property
    # def data(self):
    #     return self.__data

    # scraping
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

            data = web.DataReader(acao, data_source='yahoo', start='01-01-2017')
            data.drop('Adj Close', inplace=True, axis=1)
            st.dataframe(data.tail())

        return data



#    @classmethod
    def show_raw_data(self, data):
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        if st.checkbox('Show raw data'):
            st.subheader('Escolha um modo de exibição dos dados brutos')
                #st.write(controller.data)
                # transformar em função e usar @st.cache
            chosen = st.radio(
                '',
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

    def prepare_data(self, train_test):

        data = self.data["Close"]
        data.index = self.data["Date"]
        st.table(data)
        st.write("prepare")

#        # Impute fill NaN based on the previous value of another cell
        data['close'] = data['close'].ffill()
        data = data.bfill(axis=1)
        st.table(data)
        st.write("nan")

       # Set target column
#        self.chosen_target = st.sidebar.selectbox("Please choose target column", ('Close', 'Adj Close'))
#        self.chosen_target = data["Close"]



        #Normalização
        dataset = data.values.reshape((-1, 1))
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        st.write("normalização")

    #Divisão do dataset em treino e teste
        tam = int(len(dataset) * train_test)
        dataset_teste = dataset[tam:]
        dataset_treino = dataset[:tam]
        st.write("treino-teste")






if __name__ == '__main__':
    controller = Predictor()

#    age = st.slider('How old are you?', 0, 130, 25, format="%d years old")
#    st.write("I'm ", age, 'years old')
#    try:
    data = pd.DataFrame()

    controller.data = controller.acao_selector()
    controller.show_raw_data()

    # st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    # if st.checkbox('Show raw data'):
    #     st.subheader('Escolha um modo de exibição dos dados brutos')
    #         #st.write(controller.data)
    #         # transformar em função e usar @st.cache
    #     chosen = st.radio(
    #         '',
    #         ("Não exibir", "Tail", "Head", "Dataframe"))
    #     if chosen == "Não exibir":
    #         st.write(f"Opção escolhida é: {chosen}")
    #     if chosen == "Tail":
    #         st.subheader('Tail')
    #         st.dataframe(controller.data.tail())
    #     if chosen == "Head":
    #         st.subheader('Head')
    #         st.dataframe(controller.data.head())
    #     if chosen == "Dataframe":
    #         st.subheader('Tabela')
    #         st.dataframe(controller.data)



    if controller.data is not None:
        with st.form(key='split_form'):
#            text_input = st.text_input(label='Enter your name')
#            submit_button = st.form_submit_button(label='Submit')


#            start_date = st.sidebar.date_input('Start date', datetime.date('01-01-2017'))
            train_test = st.sidebar.slider('Train-test split %', 1, 99, 70)




            look_back = st.sidebar.slider('Look back split %', 1, 50, 5)
            future_target = st.sidebar.slider('Future target split %', 1, 10, 1)
            split_btn = st.sidebar.button('Split')
            controller.prepare_data(train_test)

        if split_btn:
            with st.form(key='predict_form'):

                predict_btn = st.sidebar.button('Predict')
#            controller.set_neural_network()

#            submit_button = st.form_submit_button(label='Submit')
#    except (AttributeError, KeyError) as e:
# st.markdown('<span style="color:blue">WRONG FILE TYPE</span>', unsafe_allow_html=True)


#    if controller.data is not None:
#         if predict_btn:
#             pass
            # st.sidebar.text("Progress:")
            # my_bar = st.sidebar.progress(0)
            # prediction, prediction_train, result, result_train = controller.predict(predict_btn)
            # for percent_complete in range(100):
            #     my_bar.progress(percent_complete + 1)


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

# """
       # #        # Standardize the feature data
       # #         X = data.loc[:, data.columns != self.chosen_target]
       # #         scaler = MinMaxScaler(feature_range=(0,1))
       # #         scaler.fit(X)
       # #         X = pd.DataFrame(scaler.transform(X))
       # #         X.columns = data.loc[:, data.columns != self.chosen_target].columns
       # #         y = data[self.chosen_target]
       # # """


# st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
# with st.beta_container():
#     st.subheader('Escolha um modo de exibição dos dados brutos')
#     # transformar em função e usar @st.cache
#     #st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
#     chosen = st.radio(
#         'raw data',
#         ("Não exibir", "Tail", "Head", "Dataframe"))
#     if chosen == "Não exibir":
#         st.write(f"Opção escolhida é: {chosen}")
#     if chosen == "Tail":
#         st.subheader('Tail')
#
#     if chosen == "Head":
#         st.subheader('Head')
#
#     if chosen == "Dataframe":
#         st.subheader('Tabela')
#
#
# st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

#
# """
# #import yfinance as yf
#
# def add_stream_url(track_ids):
#     return [f'https://open.spotify.com/track/{t}' for t in track_ids]
#
# def make_clickable(url, text):
#     return f'<a target="_blank" href="{url}">{text}</a>'
#
# """
#         data = self.data["Close"]
#         data['Date'] = datetime.date.fromisoformat(self.data['Date']).date()
# #        data = self.data.sort_values('Date')
#         data.index = self.data["Date"]
#
#         xxxxxxxxxxxxx
#
#         data = self.data["Close"]
#         data['Date'] = pd.to_datetime(self.data['Date']).date()
#         data = self.data.sort_values('Date')
#         data.index = self.data["Date"]
#
#         xxxxxxxxxxxxxxxxxxxxxxxx
#
# for key, val in data.items():
#         if isinstance(val, str):
#             if "T00:00:00-" in val:
#                 try:
#                     data[key] = datetime.fromisoformat(val)
#                 except ValueError:
#                     pass
#             try:
#                 data[key] = datetime.strptime(val, r"%Y-%m-%d")
#             except ValueError:
#                 pass
#     return data
#
#
# xxxxxxxxxxxxxxxxxxx
#
# datetime.replace(year=self.year, month=self.month, day=self.day)
#
#
# # """
# #         data = self.data["Close"]
# #         data['Date'] = pd.to_datetime(self.data['Date']).date()
# #         data = self.data.sort_values('Date')
# #         data.index = self.data["Date"]
# #         st.table(data)
# # #        data = data.sample(frac = round(split_data/100,2))
# #        # data de inicio do dataset selecionável
# # """
# # """
