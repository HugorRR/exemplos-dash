# Dashboard de Tipos de Gráficos no Streamlit

Este projeto demonstra todos os tipos de gráficos disponíveis através do Streamlit, incluindo visualizações nativas e integrações com bibliotecas populares como Matplotlib, Plotly, Altair, Bokeh, PyDeck e Graphviz.

## Tipos de Gráficos Incluídos

1. **Gráficos Nativos do Streamlit**
   - st.line_chart() - Gráfico de linha simples
   - st.area_chart() - Gráfico de área
   - st.bar_chart() - Gráfico de barras
   - st.map() - Visualização de dados geográficos em um mapa

2. **Matplotlib/Seaborn (st.pyplot())**
   - Gráficos de linha personalizados
   - Gráficos de barra usando Seaborn
   - Gráficos de dispersão
   - Mapas de calor

3. **Altair/Vega-Lite**
   - st.altair_chart() - Gráficos interativos Altair
   - st.vega_lite_chart() - Gráficos Vega-Lite

4. **Plotly**
   - st.plotly_chart() - Gráficos interativos de linha, barra, dispersão 3D e pizza

5. **Outros**
   - st.bokeh_chart() - Gráficos Bokeh
   - st.pydeck_chart() - Visualizações geoespaciais 3D
   - st.graphviz_chart() - Diagramas Graphviz

## Instalação

1. Clone este repositório:
```
git clone <url-do-repositorio>
cd <pasta-do-repositorio>
```

2. Instale as dependências:
```
pip install -r requirements.txt
```

## Execução

Para executar o aplicativo Streamlit, use o seguinte comando:
```
streamlit run app.py
```

O aplicativo será aberto automaticamente em seu navegador. Se não abrir, você pode acessar através do endereço indicado no terminal (normalmente http://localhost:8501).

## Requisitos

Este projeto requer Python 3.7+ e as bibliotecas listadas em `requirements.txt`.

> **Nota Importante**: O Streamlit é compatível apenas com a versão 2.4.3 do Bokeh. Se você encontrar erros relacionados à versão do Bokeh, execute o seguinte comando:
> ```
> pip install --force-reinstall --no-deps bokeh==2.4.3
> ```

## Estrutura do Projeto

- `app.py` - Código principal do aplicativo Streamlit
- `requirements.txt` - Lista de dependências
- `README.md` - Este arquivo de documentação

## Personalização

Você pode facilmente personalizar este dashboard:

- Modifique a função `generate_data()` para usar seus próprios dados
- Adicione novos tipos de gráficos ou personalize os existentes
- Altere o layout, cores e estilos dos gráficos conforme necessário

## Licença

Este projeto está disponível como código aberto sob os termos da licença MIT. 