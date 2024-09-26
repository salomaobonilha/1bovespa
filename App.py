import streamlit as st
import json
from prophet.serialize import model_from_json
import pandas as pd
from prophet.plot import plot_plotly

def load_model():
    with open('model_bovespa_prophet.json','r') as file_in:
        modelo = model_from_json(json.load(file_in))
        return modelo

modelo = load_model()

st.title('Previsão de fechamento do índice Ibovespa')

st.caption('''Este projeto utiliza a biblioteca Prophet para prever o fechamento do índice da Bovespa. O modelo
           criado foi treinado com dados até o dia 01/01/2023 e possui um erro de previsão (RMSE - Erro Quadrático Médio) igual a 10.99 nos dados de teste.
           O usuário pode inserir o número de dias para os quais deseja a previsão, e o modelo gerará um gráfico
           interativo contendo as estimativas baseadas em dados históricos de fechamento da bolsa.
           Além disso, uma tabela será exibida com os valores estimados para cada dia.''')

st.subheader('Insira o número de dias para previsão:')

dias = st.number_input('', min_value=1, value=1, step=1)

if 'previsao_feita' not in st.session_state:
    st.session_state['previsao_feita']=False
    st.session_state['dados_previsao']=None

if st.button('Prever'):
    st.session_state.previsao_feita = True
    futuro = modelo.make_future_dataframe(periods=dias,freq='D')
    previsao = modelo.predict(futuro)
    st.session_state['dados_previsao']=previsao

if st.session_state.previsao_feita:
    fig = plot_plotly(modelo,st.session_state['dados_previsao'])
    fig.update_layout({
        'plot_bgcolor': 'rgba(255, 255, 255, 1)',  # Define o fundo da área do gráfico como branco
        'paper_bgcolor': 'rgba(255, 255, 255, 1)', # Define o fundo externo ao gráfico como branco
        'title': {'text': "Previsão de fechamento da Ibovespa", 'font': {'color': 'black'}},
        'xaxis': {'title': 'Data', 'title_font': {'color': 'black'}, 'tickfont': {'color': 'black'}},
        'yaxis': {'title': 'Fechamento Ibovespa', 'title_font': {'color': 'black'}, 'tickfont': {'color': 'black'}}
    })
    st.plotly_chart(fig)

    previsao = st.session_state['dados_previsao']
    tabela_previsao = previsao[['ds','yhat']].tail(dias)
    tabela_previsao.columns = ['Data (Dia/Mês/Ano)','Fechamento']
    tabela_previsao['Data (Dia/Mês/Ano)'] = tabela_previsao['Data (Dia/Mês/Ano)'].dt.strftime('%d-%m-%y')
    tabela_previsao['Fechamento'] = tabela_previsao['Fechamento'].round(2)
    tabela_previsao.reset_index(drop=True,inplace=True)
    st.write("Tabela contendo as previsões de fechamento do índice da Bovespa para os próximos {} dias:".format(dias))
    st.dataframe(tabela_previsao, height=300)

    csv = tabela_previsao.to_csv(index=False)
    st.download_button(label='Baixar tabela como csv',data=csv,file_name='previsao_ibovespa.csv',mime='text/csv')