import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
import pydeck as pdk
import graphviz
import seaborn as sns
from sklearn.datasets import make_blobs

# Set page configuration
st.set_page_config(
    page_title="Dashboard de Tipos de Gráficos",
    page_icon="📊",
    layout="wide"
)

# Header
st.title("📊 Dashboard de Tipos de Gráficos")
st.subheader("Demonstração de todos os tipos de gráficos disponíveis no Streamlit")

# Introdução com explicação
st.markdown("""
## Guia Completo de Visualização de Dados para Tomada de Decisões

Este dashboard apresenta um catálogo completo de visualizações que podem transformar dados em insights valiosos para sua empresa.

Aqui você encontrará exemplos de todos os tipos de gráficos e visualizações disponíveis, organizados por categoria e com explicações sobre quando usar cada um para obter o máximo valor dos seus dados.

Cada visualização inclui uma explicação dos casos de uso mais adequados e os benefícios que proporciona para análise de negócios.
""")

# Generate sample data
@st.cache_data
def generate_data():
    # Time series data for line, area, and bar charts
    dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    np.random.seed(42)
    
    df_timeseries = pd.DataFrame({
        'date': dates,
        'sales': np.random.randint(100, 200, size=30) + np.sin(np.arange(30)/2) * 40,
        'customers': np.random.randint(80, 150, size=30) + np.cos(np.arange(30)/2) * 30,
        'conversion': np.random.uniform(1, 5, size=30) + np.sin(np.arange(30)/3) * 1.5
    })
    
    # Categorical data
    categories = ['A', 'B', 'C', 'D', 'E']
    df_categorical = pd.DataFrame({
        'category': categories,
        'value1': np.random.randint(10, 100, size=5),
        'value2': np.random.randint(10, 100, size=5)
    })
    
    # Map data (Brazilian cities)
    df_map = pd.DataFrame({
        'lat': [-23.5505, -22.9068, -30.0346, -3.7319, -15.7801, -12.9714, -27.5954, -19.9167],
        'lon': [-46.6333, -43.1729, -51.2177, -38.5267, -47.9292, -38.5014, -48.5480, -43.9345],
        'size': np.random.randint(10, 100, size=8),
        'name': ['São Paulo', 'Rio de Janeiro', 'Porto Alegre', 'Fortaleza', 'Brasília', 'Salvador', 'Florianópolis', 'Belo Horizonte']
    })
    
    # Multivariate data for scatter plots
    X, y = make_blobs(n_samples=200, centers=4, random_state=42, cluster_std=1.5)
    df_scatter = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
    df_scatter['Cluster'] = y
    
    # Network data for graphviz
    network_data = """
    digraph G {
        Vendas -> Marketing;
        Marketing -> Estratégia;
        Vendas -> Financeiro;
        Financeiro -> Relatórios;
        Estratégia -> Relatórios;
        Operações -> Logística;
        Logística -> Entrega;
    }
    """
    
    return df_timeseries, df_categorical, df_map, df_scatter, network_data

# Get data
df_timeseries, df_categorical, df_map, df_scatter, network_data = generate_data()

# Create tabs for different chart types
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Gráficos Nativos", "Matplotlib/Seaborn", "Altair/Vega-Lite", "Plotly", "Outros Gráficos", "Visualizações de Dados"])

with tab1:
    st.header("Gráficos Nativos do Streamlit")
    st.markdown("""
    Os gráficos nativos do Streamlit são simples de implementar e não requerem bibliotecas adicionais.
    São ideais para visualizações rápidas e protótipos.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gráfico de Linha")
        st.markdown("""
        **Melhor para:** Visualizar tendências ao longo do tempo.
        """)
        st.line_chart(df_timeseries.set_index('date')[['sales', 'customers']])
        
        st.subheader("Gráfico de Barras")
        st.markdown("""
        **Melhor para:** Comparar valores entre diferentes categorias.
        """)
        st.bar_chart(df_timeseries.set_index('date')[['sales', 'customers']].iloc[-10:])
    
    with col2:
        st.subheader("Gráfico de Área")
        st.markdown("""
        **Melhor para:** Visualizar volumes e comparações acumulativas.
        """)
        st.area_chart(df_timeseries.set_index('date')[['sales', 'customers']])
        
        st.subheader("Mapa")
        st.markdown("""
        **Melhor para:** Visualizar dados geográficos de forma simples.
        """)
        st.map(df_map)

with tab2:
    st.header("Integração com Matplotlib e Seaborn")
    st.markdown("""
    Matplotlib e Seaborn oferecem maior controle sobre a personalização dos gráficos.
    Use quando precisar de visualizações estatísticas avançadas ou gráficos altamente personalizados.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gráficos de Linha Personalizados")
        st.markdown("""
        **Melhor para:** Criar visualizações altamente personalizadas.
        """)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df_timeseries['date'], df_timeseries['sales'], label='Vendas')
        ax.plot(df_timeseries['date'], df_timeseries['customers'], label='Clientes')
        ax.set_xlabel('Data')
        ax.set_ylabel('Valor')
        ax.set_title('Vendas e Clientes ao Longo do Tempo')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Gráficos de Barra com Seaborn")
        st.markdown("""
        **Melhor para:** Gráficos estatísticos elegantes com menos código.
        """)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='category', y='value1', data=df_categorical, ax=ax)
        ax.set_title('Valor por Categoria')
        ax.set_xlabel('Categoria')
        ax.set_ylabel('Valor')
        plt.tight_layout()
        st.pyplot(fig)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Gráfico de Dispersão")
        st.markdown("""
        **Melhor para:** Identificar relações e padrões entre variáveis.
        """)
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df_scatter['Feature 1'], df_scatter['Feature 2'], 
                            c=df_scatter['Cluster'], cmap='viridis', alpha=0.8)
        ax.set_title('Gráfico de Dispersão por Cluster')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col4:
        st.subheader("Gráfico de Calor")
        st.markdown("""
        **Melhor para:** Visualizar matrizes de correlação e dados em grade.
        """)
        # Create correlation matrix
        corr_matrix = df_timeseries[['sales', 'customers', 'conversion']].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Matriz de Correlação')
        plt.tight_layout()
        st.pyplot(fig)

with tab3:
    st.header("Integração com Altair e Vega-Lite")
    st.markdown("""
    Altair e Vega-Lite permitem criar gráficos interativos usando uma abordagem declarativa.
    Ideal para explorações de dados interativas e elegantes.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gráfico Interativo Altair")
        st.markdown("""
        **Melhor para:** Visualizações de dados interativas com sintaxe declarativa.
        """)
        chart = alt.Chart(df_timeseries).mark_line().encode(
            x='date:T',
            y='sales:Q',
            tooltip=['date:T', 'sales:Q']
        ).properties(
            title='Vendas ao Longo do Tempo',
            width=600
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)
    
    with col2:
        st.subheader("Gráfico de Barras Vega-Lite")
        st.markdown("""
        **Melhor para:** Definir visualizações usando especificações JSON do Vega-Lite.
        """)
        vega_spec = {
            'mark': 'bar',
            'encoding': {
                'x': {'field': 'category', 'type': 'nominal'},
                'y': {'field': 'value1', 'type': 'quantitative'},
                'color': {'field': 'category', 'type': 'nominal'},
                'tooltip': [
                    {'field': 'category', 'type': 'nominal'},
                    {'field': 'value1', 'type': 'quantitative'}
                ]
            }
        }
        
        st.vega_lite_chart(df_categorical, vega_spec, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Gráfico de Dispersão Interativo")
        st.markdown("""
        **Melhor para:** Criar gráficos de dispersão interativos para análise exploratória.
        """)
        scatter_chart = alt.Chart(df_scatter).mark_circle(size=60).encode(
            x='Feature 1:Q',
            y='Feature 2:Q',
            color='Cluster:N',
            tooltip=['Feature 1:Q', 'Feature 2:Q', 'Cluster:N']
        ).properties(
            width=600,
            height=400,
            title='Gráfico de Dispersão por Cluster'
        ).interactive()
        
        st.altair_chart(scatter_chart, use_container_width=True)
    
    with col4:
        st.subheader("Mapa de Calor Vega-Lite")
        st.markdown("""
        **Melhor para:** Visualizar dados em grade ou matrizes de correlação.
        """)
        # Prepare data for heatmap
        heatmap_data = df_timeseries[['sales', 'customers', 'conversion']].corr().reset_index()
        heatmap_data = heatmap_data.melt(id_vars='index')
        
        vega_heatmap = {
            'mark': 'rect',
            'encoding': {
                'x': {'field': 'index', 'type': 'nominal'},
                'y': {'field': 'variable', 'type': 'nominal'},
                'color': {'field': 'value', 'type': 'quantitative', 'scale': {'scheme': 'blueorange'}},
                'tooltip': [
                    {'field': 'index', 'type': 'nominal'},
                    {'field': 'variable', 'type': 'nominal'},
                    {'field': 'value', 'type': 'quantitative'}
                ]
            }
        }
        
        st.vega_lite_chart(heatmap_data, vega_heatmap, use_container_width=True)

with tab4:
    st.header("Integração com Plotly")
    st.markdown("""
    Plotly oferece gráficos interativos de alta qualidade com recursos avançados.
    Perfeito para dashboards com visualizações interativas sofisticadas.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gráfico de Linha Plotly")
        st.markdown("""
        **Melhor para:** Criar gráficos de linha interativos com recursos avançados.
        """)
        fig = px.line(df_timeseries, x='date', y=['sales', 'customers'], 
                     title='Vendas e Clientes ao Longo do Tempo')
        fig.update_layout(xaxis_title='Data', yaxis_title='Valor',
                         legend_title='Métrica', height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Gráfico de Barras Plotly")
        st.markdown("""
        **Melhor para:** Gráficos de barras interativos com múltiplas séries e recursos avançados.
        """)
        fig = px.bar(df_categorical, x='category', y=['value1', 'value2'], 
                    barmode='group', title='Valores por Categoria')
        fig.update_layout(xaxis_title='Categoria', yaxis_title='Valor',
                         legend_title='Série', height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Gráfico de Dispersão 3D")
        st.markdown("""
        **Melhor para:** Visualizações tridimensionais interativas para análise multivariada.
        """)
        # Add a third dimension to our data
        df_scatter_3d = df_scatter.copy()
        df_scatter_3d['Feature 3'] = np.random.randn(len(df_scatter)) * 2 + df_scatter_3d['Cluster'] * 1.5
        
        fig = px.scatter_3d(df_scatter_3d, x='Feature 1', y='Feature 2', z='Feature 3',
                           color='Cluster', opacity=0.7, title='Gráfico de Dispersão 3D')
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        st.subheader("Gráfico de Pizza")
        st.markdown("""
        **Melhor para:** Visualizar proporções de um total ou composição percentual.
        """)
        fig = px.pie(df_categorical, values='value1', names='category', 
                   title='Distribuição de Valores por Categoria')
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.header("Outros Tipos de Gráficos")
    st.markdown("""
    O Streamlit suporta integração com outras bibliotecas populares como Bokeh para gráficos interativos na web,
    PyDeck para visualizações geoespaciais 3D, e Graphviz para diagramas estruturais.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gráfico Bokeh")
        st.markdown("""
        **Melhor para:** Visualizações interativas para aplicações web.
        """)
        categories = df_categorical['category'].tolist()
        
        # Criando o gráfico Bokeh
        p = figure(
            x_range=categories, 
            height=350, 
            title="Gráfico de Barras Bokeh",
            toolbar_location=None
        )
        
        # Deslocamento para posicionar as barras lado a lado
        width = 0.4
        offset = width/2
        
        # Primeira série de barras (deslocada para a esquerda)
        p.vbar(
            x=[i-offset for i in range(len(categories))], 
            top=df_categorical['value1'], 
            width=width, 
            legend_label="Valor 1",
            color="#c9d9d3"
        )
        
        # Segunda série de barras (deslocada para a direita)
        p.vbar(
            x=[i+offset for i in range(len(categories))], 
            top=df_categorical['value2'], 
            width=width, 
            legend_label="Valor 2",
            color="#718dbf", 
            alpha=0.7
        )
        
        # Configurando rótulos do eixo x
        p.xaxis.ticker = list(range(len(categories)))
        p.xaxis.major_label_overrides = {i: category for i, category in enumerate(categories)}
        
        p.xgrid.grid_line_color = None
        p.legend.location = "top_right"
        p.legend.orientation = "horizontal"
        
        st.bokeh_chart(p, use_container_width=True)
    
    with col2:
        st.subheader("Diagrama Graphviz")
        st.markdown("""
        **Melhor para:** Visualizar estruturas hierárquicas, fluxos e redes.
        """)
        st.graphviz_chart(network_data)
    
    st.subheader("Visualização PyDeck 3D")
    st.markdown("""
    **Melhor para:** Visualizações geoespaciais avançadas e mapas 3D.
    """)
    
    # Prepare data for PyDeck
    df_map['size_scaled'] = df_map['size'] * 1000  # Scale for visibility
    
    # Define layer
    layer = pdk.Layer(
        "ScatterplotLayer",
        df_map,
        get_position=['lon', 'lat'],
        get_radius='size_scaled',
        get_fill_color=[255, 140, 0, 140],
        pickable=True,
        opacity=0.8,
        radius_scale=1,
        radius_min_pixels=5,
        radius_max_pixels=50,
    )
    
    # Set the viewport location
    view_state = pdk.ViewState(
        longitude=-53.0,
        latitude=-14.0,
        zoom=3,
        pitch=50,
    )
    
    # Render
    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "{name}\nValor: {size}"},
        map_style="mapbox://styles/mapbox/light-v9",
    )
    
    st.pydeck_chart(r)

with tab6:
    st.header("Visualizações Adicionais de Dados")
    st.markdown("""
    Além dos gráficos, o Streamlit também oferece outras formas de visualizar dados,
    como métricas, tabelas interativas, e visualizações formatadas.
    """)
    
    # Métricas
    st.subheader("Indicadores com Tendência")
    st.markdown("""
    **Melhor para:** Destacar KPIs e métricas importantes com indicadores de tendência.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Vendas", value="R$12.456", delta="8%")
    with col2:
        st.metric(label="Clientes", value="1.245", delta="-2%")
    with col3:
        st.metric(label="Conversão", value="3.2%", delta="0.5%")
    
    # Tabelas e DataFrames
    st.subheader("Tabelas Interativas")
    st.markdown("""
    **Melhor para:** Exibir dados tabulares com recursos interativos (ordenação, filtragem).
    """)
    st.dataframe(df_timeseries.head(10))
    
    st.subheader("Tabelas Estáticas")
    st.markdown("""
    **Melhor para:** Exibir pequenos conjuntos de dados de forma estática e compacta.
    """)
    st.table(df_categorical)
    
    # JSON
    st.subheader("Visualização de Estruturas JSON")
    st.markdown("""
    **Melhor para:** Exibir estruturas de dados hierárquicas de forma formatada.
    """)
    st.json({
        "nome": "Dashboard de Visualização",
        "criado_por": "Equipe de BI",
        "metricas": {
            "cliques": 1245,
            "visualizacoes": 5632,
            "conversao": 0.032
        },
        "integrações": ["Matplotlib", "Plotly", "Altair", "Bokeh", "PyDeck", "Graphviz"]
    })

# Conclusion
st.markdown("---")
st.header("Conclusão e Recomendações")
st.write("""
## Análise Comparativa das Bibliotecas de Visualização

Este dashboard demonstra os diferentes tipos de gráficos disponíveis através do Streamlit.
Cada biblioteca tem seus pontos fortes e casos de uso ideais:

### Gráficos Nativos do Streamlit
✅ **Vantagens:** Implementação rápida e fácil, sem dependências adicionais
❌ **Limitações:** Menos opções de personalização
🔍 **Melhor para:** Prototipagem rápida e análises simples

### Matplotlib/Seaborn
✅ **Vantagens:** Controle detalhado, grande comunidade, estabilidade
❌ **Limitações:** Menos interativos, sintaxe mais verbosa
🔍 **Melhor para:** Publicações científicas, visualizações estatísticas complexas

### Altair/Vega-Lite
✅ **Vantagens:** Sintaxe declarativa, interatividade elegante
❌ **Limitações:** Conjuntos de dados menores, curva de aprendizado para gramática
🔍 **Melhor para:** Visualizações interativas com menos código

### Plotly
✅ **Vantagens:** Alta interatividade, gráficos 3D, dashboard-ready
❌ **Limitações:** Arquivos maiores, pode ser mais lento
🔍 **Melhor para:** Dashboards interativos, visualizações complexas e 3D

### Bokeh
✅ **Vantagens:** Interatividade para web, design para aplicações
❌ **Limitações:** API mais complexa, cuidado com versões compatíveis
🔍 **Melhor para:** Aplicações web interativas integradas

### PyDeck/GraphViz
✅ **Vantagens:** Especializados em seus domínios (mapas 3D, diagramas)
❌ **Limitações:** Casos de uso mais específicos
🔍 **Melhor para:** Visualizações geoespaciais (PyDeck) e diagramas de fluxo (GraphViz)

## Recomendações para Implementação

1. Para protótipos rápidos: use os gráficos nativos do Streamlit
2. Para análises exploratórias detalhadas: use Plotly ou Altair
3. Para publicações científicas: use Matplotlib/Seaborn
4. Para mapas e visualizações geoespaciais: use PyDeck
5. Para diagramas e gráficos de rede: use Graphviz

Esta versatilidade de opções permite selecionar a ferramenta ideal para cada caso de uso específico.
""")

# Sugestões finais
st.info("""
💡 **Próximos Passos Sugeridos:**
1. Escolher a biblioteca mais adequada para os requisitos específicos do seu projeto
2. Implementar visualizações consistentes utilizando a biblioteca escolhida
3. Considerar a combinação de diferentes bibliotecas para diferentes tipos de visualização
""") 