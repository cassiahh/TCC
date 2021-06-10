import streamlit as st
import streamlit.components.v1 as components
from htbuilder import HtmlElement, div, br, hr, a, p, img, styles
from htbuilder.units import percent, px

#https://playground.tensorflow.org/


def disclaimer():
    st.sidebar.subheader('Disclaimer')
    with st.sidebar.beta_expander("Leia Me"):
        st.write("""
            As informações e as predições aqui contidas foram consolidadas ou elaboradas com base em informações obtidas de fontes, em princípio, fidedignas e de boa-fé. 
            Entretanto, a desenvolvedora não declara, nem garante, expressa ou tacitamente, que estas informações e opiniões sejam imparciais, precisas, completas ou corretas. 
            Todas as recomendações e estimativas apresentadas podem ser alteradas a qualquer momento sem aviso prévio, em função de mudanças que possam afetar as projeções da empresa.\n  
            Este material tem por finalidade apenas informar e servir como instrumento que auxilie a tomada de decisão de investimento. 
            Não é, e não deve ser interpretado como uma oferta ou solicitação de oferta para comprar ou vender quaisquer títulos e valores mobiliários ou outros instrumentos financeiros.
        """)


def etf_ticker():
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


def st_faq():
    st.sidebar.subheader('Termos e Política de Privacidade')
    with st.sidebar.beta_expander("Politica de privacidade"):
        st.write("""
            Os dados tratados para fornecer os Serviços são extraidos do "Yahoo Finance".\n
            Se o Serviço incluir um produto de terceiro, o usuário entende e concorda que o uso do Serviço também estarão sujeitos aos termos de serviço 
            e à política de privacidade deste terceiro, os quais concordará integralmente antes de utilizar a aplicação.
        """)
    with st.sidebar.beta_expander("Termos de Tratamento de Dados"):
        st.write("""
            Para os fins destes Termos de Tratamento de Dados e adequação legal, a presente aplicação não envolve coleta, uso ou envio de Dados Pessoais.\n
        """)
    with st.sidebar.beta_expander("Termos de Serviço"):
        st.write("""
            Esta página foi produzida com finalidade exclusivamente acadêmica - TCC Cássia Chin de curso ADS do IFSP. \n
            O usuário declara que tem a idade mínima necessária para celebrar um contrato válido.\n
            Como não há necessidade de efetuar login, ao acessar a aplicação, o usuário está de acordo com os Termos e as Políticas descritas.\n
            É proibido usar indevidamente ou interferir nos Serviços.\n
            Não se imputa responsabilidade a desenvolvedora por interrupções ou falhas dos serviços, sejam por atos ou omissões.\n
            O uso dos Serviços não concede a propriedade de qualquer direito intelectual, nem qualquer outra titularidade ou participação nos Serviços ou no conteúdo que acessa.\n
            Na máxima extensão permitida pela legislação aplicável, o usuário concorda e entende que a desenvolvedora não é responsável por: 
            quaisquer danos indiretos, especiais, incidentais, consequentes, agudos ou outros danos múltiplos, exemplares ou punitivos decorrentes ou relacionados a estes termos ou ao seu uso dos serviços. 
            Assim como, por eventuais litígios decorrentes ou relacionados a estes termos ou serviços
        """)


def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):
    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 10px; }
    </style>
    """
    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1
    )
    style_hr = styles(
        display="block",
        margin=px(0, 0, "auto", "auto"),
        border_style="inset",
        border_width=px(1)
    )
    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )
    st.markdown(style, unsafe_allow_html=True)
    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)
    st.markdown(str(foot), unsafe_allow_html=True)


def footer():
    myargs = [
        #"Made in ",
        "<b>Made with</b>: Python 3.9 ",
        link("https://www.python.org/", image('https://i.imgur.com/ml09ccU.png',
            width=px(18), height=px(18), margin="0em")),
        ", Streamlit ",
        link("https://streamlit.io/", image('https://docs.streamlit.io/en/stable/_static/favicon.png',
            width=px(24), height=px(25), margin="0em")),
        # ", Docker ",
        # link("https://www.docker.com/", image('https://www.docker.com/sites/default/files/d8/styles/role_icon/public/2019-07/Moby-logo.png?itok=sYH_JEaJ',
        #       width=px(20), height=px(18), margin= "0em")),
        # " and Google APP Engine ",
        # link("https://cloud.google.com/appengine", image('https://lh3.ggpht.com/_uP6bUdDOWGS6ICpMH7dBAy5LllYc_bBjjXI730L3FQ64uS1q4WltHnse7rgpKiInog2LYM1',
        #       width=px(19), height=px(19), margin= "0em", align="top")),
        br(),
        " Developed by  ",
        link("https://www.linkedin.com/in/cassia-chin-20269315/", "cassiahh@hotmail.com"),
        # br(),
        # link("https://buymeacoffee.com/cassiahh", image('https://i.imgur.com/thJhzOO.png')),
    ]
    layout(*myargs)


def faq():
    components.html(
        """
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    <div id="accordion">
      <div class="card">
        <div class="card-header" id="headingOne">
          <h5 class="mb-0">
            <button class="btn btn-link" data-toggle="collapse" data-target="#collapseOne" aria-expanded="true" aria-controls="collapseOne">
            Politica de privacidade
            </button>
          </h5>
        </div>
        <div id="collapseOne" class="collapse show" aria-labelledby="headingOne" data-parent="#accordion">
          <div class="card-body">
            Collapsible Group Item #1 content
          </div>
        </div>
      </div>
      <div class="card">
        <div class="card-header" id="headingTwo">
          <h5 class="mb-0">
            <button class="btn btn-link collapsed" data-toggle="collapse" data-target="#collapseTwo" aria-expanded="false" aria-controls="collapseTwo">
            Termos de Tratamento de Dados
            </button>
          </h5>
        </div>
        <div id="collapseTwo" class="collapse" aria-labelledby="headingTwo" data-parent="#accordion">
          <div class="card-body">
            Collapsible Group Item #2 content
          </div>
        </div>
      </div>
      <div class="card">
        <div class="card-header" id="headingThree">
          <h5 class="mb-0">
            <button class="btn btn-link" data-toggle="collapse" data-target="#collapseThree" aria-expanded="true" aria-controls="collapseThree">
            Termos de Serviço
            </button>
          </h5>
        </div>
        <div id="collapseThree" class="collapse show" aria-labelledby="headingThree" data-parent="#accordion">
          <div class="card-body">
            Esta página foi produzida com finalidade somente acadêmica - TCC Cássia Chin de curso ADS do IFSP. 
          </div>
        </div>
      </div>
      <div class="card">
        <div class="card-header" id="headingFour">
          <h5 class="mb-0">
            <button class="btn btn-link collapsed" data-toggle="collapse" data-target="#collapseFour" aria-expanded="false" aria-controls="collapseFour">
            Cookies
            </button>
          </h5>
        </div>
        <div id="collapseFour" class="collapse" aria-labelledby="headingFour" data-parent="#accordion">
          <div class="card-body">
            Collapsible Group Item #4 content
          </div>
        </div>
      </div>
    </div>
    """,
        height=600,
    )

#Change backgroud
# st.markdown(
#   """
#   <style>
#   .reportview-container {
#     background: url("url_goes_here")
#   }
#   .sidebar .sidebar-content {
#     background: url("url_goes_here")
#     }
#   </style>
#   """,
#   unsafe_allow_html=True
# )
#
