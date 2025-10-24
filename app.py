# ============================================
# DASHBOARD - CENSO ESCOLAR 2024 E CURSOS TÉCNICOS + 📖 STORYTELLING (Aba 0)
# ============================================

import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pathlib import Path
from io import BytesIO

# --------------------------------------------
# 🔧 Configuração da Página
# --------------------------------------------
st.set_page_config(page_title="Censo Escolar 2024", page_icon="🎓", layout="wide")
st.title("📊 Dashboard - Censo da Educação Básica e Cursos Técnicos 2024")

# --------------------------------------------
# 📂 Caminhos dos Arquivos
# --------------------------------------------
# base = Path(r"C:\Users\gracabezerra\OneDrive - SENAC Pernambuco\Documentos\Python_work\microdados_censo_escolar_2024\microdados_censo_escolar_2024\dados")
# base = Path(r"C:\Projetos\MBA\microdados_censo_escolar_2024\dados")
arq_micro = "https://drive.usercontent.google.com/download?id=1eZjq5k-j50jkwVh7if5vHq9mb70Ryf00&export=download&authuser=0&confirm=t&uuid=55fd7bbc-eb15-4007-93e9-b38cd898abfc&at=AKSUxGP_eFbtXfhFGqwVIAmwBFOm:1761344808523"
arq_suple = "https://drive.usercontent.google.com/download?id=1eec97hETZ9gQTtdWMVbqUsMkbOu4XU2h&export=download&authuser=0&confirm=t&uuid=4a77ba39-b2ad-4dd6-9551-69e22f75b4fc&at=AKSUxGOlE5BZ5-iEDvmsPJXrPAe9:1761344839937"

# CSV específico da análise de storytelling (ficar ao lado do app, ou ajuste o path conforme necessário)
arq_story = "https://drive.usercontent.google.com/download?id=1P-dME1zfxQb72YyttTB9rJSdDMweJ_d9&export=download&authuser=0&confirm=t&uuid=13f51844-833f-480d-a664-b62ac77b1e7e&at=AKSUxGMWzF8N9ZD_pGbBm25ER2RE:1761344669692"

# =====================================================
# 🚀 CARREGAMENTO E PREPARAÇÃO DOS DADOS (Dashboard)
# =====================================================
@st.cache_data(persist="disk", show_spinner=True)
def carregar_dados():
    micro = pd.read_csv(arq_micro, sep=";", encoding="latin1", low_memory=False)
    suple = pd.read_csv(arq_suple, sep=";", encoding="latin1", low_memory=False)

    micro.columns = micro.columns.str.strip().str.upper()
    suple.columns = suple.columns.str.strip().str.upper()

    micro["CO_ENTIDADE"] = micro["CO_ENTIDADE"].astype(str)
    suple["CO_ENTIDADE"] = suple["CO_ENTIDADE"].astype(str)

    # Merge completo (uma escola pode ter vários cursos técnicos)
    df_local = pd.merge(
        micro, suple, on="CO_ENTIDADE", how="outer",
        validate="one_to_many", suffixes=("_BASICA", "_TECNICO")
    )

    # Campo TIPO_ESCOLA robusto
    df_local["NO_AREA_CURSO_PROFISSIONAL"] = df_local.get("NO_AREA_CURSO_PROFISSIONAL", pd.Series(dtype=str))
    df_local["TIPO_ESCOLA"] = df_local["NO_AREA_CURSO_PROFISSIONAL"].apply(
        lambda x: "Educação Técnica"
        if pd.notna(x) and str(x).strip().lower() not in ["", "não informado", "nan", "none"]
        else "Educação Básica"
    )

    return df_local

df = carregar_dados()

# =====================================================
# 🔧 NORMALIZAÇÃO DE CAMPOS
# =====================================================
def normalizar(df_, col):
    for sufixo in ["_BASICA", "_TECNICO"]:
        if f"{col}{sufixo}" in df_.columns:
            df_[col] = df_[f"{col}{sufixo}"]
            break

for c in ["SG_UF", "TP_DEPENDENCIA", "NO_ENTIDADE", "NO_MUNICIPIO"]:
    normalizar(df, c)

for col in ["NO_MUNICIPIO", "NO_ENTIDADE", "NO_AREA_CURSO_PROFISSIONAL"]:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.normalize('NFKD')
            .str.encode('ascii', errors='ignore')
            .str.decode('utf-8')
            .str.upper()
        )

# =====================================================
# 🌎 REGIÃO
# =====================================================
mapa_uf_regiao = {
    "AC": "NORTE", "AM": "NORTE", "AP": "NORTE", "PA": "NORTE", "RO": "NORTE", "RR": "NORTE", "TO": "NORTE",
    "AL": "NORDESTE", "BA": "NORDESTE", "CE": "NORDESTE", "MA": "NORDESTE",
    "PB": "NORDESTE", "PE": "NORDESTE", "PI": "NORDESTE", "RN": "NORDESTE", "SE": "NORDESTE",
    "DF": "CENTRO-OESTE", "GO": "CENTRO-OESTE", "MT": "CENTRO-OESTE", "MS": "CENTRO-OESTE",
    "ES": "SUDESTE", "MG": "SUDESTE", "RJ": "SUDESTE", "SP": "SUDESTE",
    "PR": "SUL", "RS": "SUL", "SC": "SUL"
}
df["NO_REGIAO"] = df["SG_UF"].map(mapa_uf_regiao).fillna("NÃO INFORMADA")

# =====================================================
# 🧩 FILTROS GERAIS (Sidebar)
# =====================================================
regioes = ["Selecione"] + sorted(df["NO_REGIAO"].dropna().unique())
regiao = st.sidebar.selectbox("🌎 Região:", regioes, index=0)
if regiao == "Selecione":
    st.warning("🌎 Escolha uma **Região** para continuar.")
    st.stop()
df = df[df["NO_REGIAO"] == regiao].copy()

st.sidebar.header("Filtros")

# Dependência
mapa_dependencia = {1: "FEDERAL", 2: "ESTADUAL", 3: "MUNICIPAL", 4: "PRIVADA"}
df["DEPENDENCIA_NOME"] = df["TP_DEPENDENCIA"].map(mapa_dependencia).fillna("DESCONHECIDA")

# UF
ufs = sorted(df["SG_UF"].dropna().unique().tolist())
ufs.insert(0, "Selecione")
uf = st.sidebar.selectbox("🗺️ Estado (UF):", ufs, index=0)
if uf == "Selecione":
    st.warning("🗺️ Escolha um Estado (UF) para visualizar os dados.")
    st.stop()

# Dependência administrativa
dependencias = sorted(df["DEPENDENCIA_NOME"].dropna().unique().tolist())
dependencias.insert(0, "Todas")
rede = st.sidebar.selectbox("🏫 Dependência Administrativa:", dependencias)

# Município
municipios = sorted(df.loc[df["SG_UF"] == uf, "NO_MUNICIPIO"].dropna().unique())
municipio = st.sidebar.selectbox("📍 Município:", ["Todos"] + municipios)

# Área do curso
areas = sorted(df["NO_AREA_CURSO_PROFISSIONAL"].dropna().unique())
area = st.sidebar.selectbox("💼 Área do Curso Profissional:", ["Todas"] + areas)

# Situação de funcionamento
mapa_situacao = {
    1: "EM ATIVIDADE",
    2: "PARALISADA",
    3: "EXTINTA (ANO DO CENSO)",
    4: "EXTINTA EM ANOS ANTERIORES"
}
if "TP_SITUACAO_FUNCIONAMENTO" in df.columns:
    df["SITUACAO_NOME"] = df["TP_SITUACAO_FUNCIONAMENTO"].map(mapa_situacao).fillna("NÃO INFORMADA")
else:
    df["SITUACAO_NOME"] = "NÃO INFORMADA"
situacoes = ["Todas"] + sorted(df["SITUACAO_NOME"].dropna().unique())
situacao = st.sidebar.selectbox("🏫 Situação de Funcionamento:", situacoes)

# Localização
col_loc = next((c for c in ["TP_LOCALIZACAO", "TP_LOCALIZACAO_BASICA", "TP_LOCALIZACAO_TECNICO"] if c in df.columns), None)
mapa_localizacao = {"1": "URBANA", "2": "RURAL", 1: "URBANA", 2: "RURAL"}
if col_loc:
    df["LOCALIZACAO_NOME"] = (
        df[col_loc]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .replace(mapa_localizacao)
        .fillna("NÃO INFORMADA")
    )
else:
    df["LOCALIZACAO_NOME"] = "NÃO INFORMADA"
localizacoes = ["Todas"] + sorted(df["LOCALIZACAO_NOME"].dropna().unique())
localizacao = st.sidebar.selectbox("🏞️ Localização da Escola:", localizacoes)
st.sidebar.markdown(f"**📍 Localização selecionada:** {localizacao}")

# =====================================================
# ⚙️ APLICAÇÃO DOS FILTROS
# =====================================================
df_parcial = df[df["SG_UF"] == uf].copy()
if rede != "Todas":
    df_parcial = df_parcial[df_parcial["DEPENDENCIA_NOME"] == rede]
if municipio != "Todos":
    df_parcial = df_parcial[df_parcial["NO_MUNICIPIO"] == municipio]
if area != "Todas":
    df_parcial = df_parcial[df_parcial["NO_AREA_CURSO_PROFISSIONAL"] == area]
if situacao != "Todas":
    df_parcial = df_parcial[df_parcial["SITUACAO_NOME"] == situacao]
if localizacao != "Todas":
    df_parcial = df_parcial[df_parcial["LOCALIZACAO_NOME"] == localizacao]

# =====================================================
# 🎓 TOTAL DE ESCOLAS
# =====================================================
total_escolas_unicas = df_parcial["CO_ENTIDADE"].nunique()
st.success(f"🎓 Total de Escolas Filtradas: {total_escolas_unicas:,}")

# =====================================================
# 🧭 ABAS
# =====================================================
aba0, aba1, aba2, aba3, aba4 = st.tabs([
    "📖 Abismo Digital",
    "🎓 Cursos Técnicos",
    "🏫 Dependência Administrativa",
    "📍 Municípios",
    "🏗️ Estrutura",
    # "📈 Indicadores Nacionais",
    # "📈 Estatística Descritiva"
])

# =====================================================
# 📖 ABA 0 - STORYTELLING (Desigualdade Digital)
# =====================================================
with aba0:

    # --- 0. Carregamento e Preparação dos Dados (Storytelling) ---
    @st.cache_data
    def load_data_story(path_csv: Path):
        df_st_local = pd.read_csv(path_csv)
        return df_st_local

    try:
        df_st = load_data_story(arq_story)
    except Exception as e:
        st.error(f"Não foi possível carregar '{arq_story}'. Detalhes: {e}")
        st.stop()

    # --- 1. Título e Introdução (Storytelling) ---
    # st.subheader("O Abismo Digital na Educação Básica Brasileira")
    # st.subheader("Uma Análise da Relação entre Infraestrutura Tecnológica e Matrículas (Censo Escolar 2024)")

    total_escolas = df_st.shape[0]
    total_matriculas = df_st['MATRICULAS'].sum()
    # st.markdown(f"""
    # O **acesso à tecnologia** nas escolas é um indicador crucial de equidade educacional. 
    # Nesta análise, atuamos como Cientistas de Dados para quantificar a disparidade de recursos digitais entre a rede pública e privada de ensino no Brasil.

    # - **Total de Escolas Analisadas:** {total_escolas:,}
    # - **Total de Matrículas:** {total_matriculas:,.0f}
    # """)

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Total de Escolas Analisadas", value=f"{total_escolas:,}".replace(",", "."))

    with col2:
        st.metric(
        label="Total de Matrículas",
        value=f"{total_matriculas:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    )

    

    st.markdown("---")

    # --- 2. O Abismo Público vs. Privado (Análise Comparativa) ---
    # st.header("2. O Abismo Público vs. Privado")
    st.markdown("""
    A métrica fundamental para esta análise é a **média de Equipamentos por Aluno (EQP/Aluno)**, que considera computadores de mesa, portáteis e tablets.
    """)

    # Cálculo Comparativo
    df_comparativo = df_st.groupby('TIPO_ESCOLA').agg(
        Media_EQP_Aluno=('EQP_POR_ALUNO', 'mean'),
        Total_Escolas=('DEPENDENCIA_ADM', 'count')
    ).reset_index()

    # Salvaguarda caso falte uma das categorias
    media_eqp_publica = df_comparativo.loc[df_comparativo['TIPO_ESCOLA'].str.upper() == 'PÚBLICA', 'Media_EQP_Aluno'].mean()
    media_eqp_privada = df_comparativo.loc[df_comparativo['TIPO_ESCOLA'].str.upper() == 'PRIVADA', 'Media_EQP_Aluno'].mean()

    if pd.isna(media_eqp_publica) or pd.isna(media_eqp_privada) or media_eqp_publica == 0:
        st.warning("Não foi possível calcular a razão de disparidade (verifique se há dados para Pública e Privada).")
        razao = np.nan
    else:
        razao = media_eqp_privada / media_eqp_publica

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Média EQP/Aluno (Rede Pública)", value=f"{(media_eqp_publica or 0):.4f}".replace(".", ","))

    with col2:
        st.metric(label="Média EQP/Aluno (Rede Privada)", value=f"{(media_eqp_privada or 0):.4f}".replace(".", ","))

    with col3:
        delta_txt = "Maior na Privada" if not pd.isna(razao) else "Indisponível"
        st.metric(
            label="Disparidade (Razão Privada/Pública)",
            value=f"{(razao or 0):.1f}".replace(".", ",") + "x",
            delta=delta_txt
        )

    # Gráfico de Distribuição (Boxplot)
    st.markdown("#### Distribuição de Equipamentos por Aluno")
    fig, ax = plt.subplots(figsize=(10, 6))
    # Filtrar outliers para melhor visualização
    df_plot = df_st[df_st['EQP_POR_ALUNO'] < 1]
    paleta = {'Pública': '#1f77b4', 'Privada': '#ff7f0e', 'PUBLICA': '#1f77b4', 'PRIVADA': '#ff7f0e'}
    sns.boxplot(x='TIPO_ESCOLA', y='EQP_POR_ALUNO', data=df_plot, ax=ax, palette=paleta)
    ax.set_title('Distribuição de Equipamentos por Aluno (Excluindo Outliers Extremos)')
    ax.set_xlabel('Tipo de Escola')
    ax.set_ylabel('Equipamentos por Aluno (EQP/Aluno)')
    st.pyplot(fig)

    st.markdown("---")

    # --- 3. Profundidade Regional ---
    # st.header("Profundidade Regional: Onde a Desigualdade é Mais Evidente?")
    st.markdown("""
    Disparidade tecnológica EQP/Aluno por Região
    """)

    df_regional = df_st.groupby(['REGIAO', 'TIPO_ESCOLA'])['EQP_POR_ALUNO'].mean().unstack()
    # harmoniza possíveis chaves
    cols_ok = {c: c.capitalize() for c in df_regional.columns}
    df_regional.rename(columns=cols_ok, inplace=True)
    if all(c in df_regional.columns for c in ['Privada', 'Pública']):
        df_regional['DIFERENCA'] = df_regional['Privada'] - df_regional['Pública']
    elif all(c in df_regional.columns for c in ['Privada', 'Publica']):
        df_regional['DIFERENCA'] = df_regional['Privada'] - df_regional['Publica']
        df_regional.rename(columns={'Publica': 'Pública'}, inplace=True)
    else:
        df_regional['DIFERENCA'] = np.nan
    df_regional = df_regional.sort_values(by='DIFERENCA', ascending=False)

    st.dataframe(df_regional[['Pública', 'Privada', 'DIFERENCA']].style.format("{:.4f}"))

    fig_reg, ax_reg = plt.subplots(figsize=(10, 6))
    sns.barplot(x=df_regional.index, y='DIFERENCA', data=df_regional.reset_index(), ax=ax_reg, palette='viridis')
    ax_reg.set_title('Diferença Média de EQP/Aluno (Privada - Pública) por Região')
    ax_reg.set_xlabel('Região')
    ax_reg.set_ylabel('Diferença de EQP/Aluno')
    plt.xticks(rotation=45)
    st.pyplot(fig_reg)

    st.markdown("---")

    # --- 4. Infraestrutura e Engajamento (Modelagem Estatística) ---
    # st.header("4. Infraestrutura e Engajamento: Correlação e Modelagem")
    # st.markdown("""
    # Para testar nossa hipótese de que a infraestrutura tecnológica impacta o engajamento (medido pelo número de matrículas), aplicamos uma **Regressão Linear Simples**.
    # """)

    # Preparação dos dados para Modelagem
    df_model = df_st[(df_st['MATRICULAS'] > 0) & (df_st['TOTAL_EQUIPAMENTOS'] > 0)].copy()
    if df_model.empty:
        st.warning("Sem dados suficientes para modelagem (verifique 'MATRICULAS' e 'TOTAL_EQUIPAMENTOS').")
    else:
        df_model['LOG_EQP'] = np.log(df_model['TOTAL_EQUIPAMENTOS'])
        df_model['LOG_MATRICULAS'] = np.log(df_model['MATRICULAS'])

        # Correlação de Pearson
        correlation = df_model['LOG_EQP'].corr(df_model['LOG_MATRICULAS'])
        # st.info(f"**Correlação de Pearson** (Log Equipamentos vs. Log Matrículas): **{correlation:.3f}**")
        # st.caption("Uma correlação positiva e forte indica que, em geral, escolas com mais equipamentos tendem a ter mais matrículas.")

        # Regressão Linear Simples
        X = df_model[['LOG_EQP']]
        y = df_model['LOG_MATRICULAS']
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # Visualização da Regressão
        fig_regress, ax_regress = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='LOG_EQP', y='LOG_MATRICULAS', data=df_model, hue='TIPO_ESCOLA', ax=ax_regress, alpha=0.6)
        ax_regress.plot(X, y_pred, color='red', linewidth=2, label=f'Regressão (R²: {model.score(X, y):.3f})')
        ax_regress.set_title('Regressão Linear: Log Matrículas vs. Log Total de Equipamentos')
        ax_regress.set_xlabel('Log do Total de Equipamentos')
        ax_regress.set_ylabel('Log do Número de Matrículas')
        ax_regress.legend(title='Tipo de Escola')
        st.pyplot(fig_regress)

        st.markdown(f"""
        - **Coeficiente de Determinação (R²):** **{model.score(X, y):.3f}**. Isso significa que {model.score(X, y)*100:.1f}% da variação no número de matrículas pode ser explicada pela variação no número de equipamentos.
        - **Coeficiente da Regressão (Inclinação):** **{model.coef_[0]:.3f}**. Para cada aumento de 1 unidade no log dos equipamentos, o log das matrículas aumenta em {model.coef_[0]:.3f} unidades.
        """)

    st.markdown("---")

    # --- 5. Conclusão e Insights ---
    # st.header("5. Conclusão e Caminhos para a Inclusão Digital")
    # st.markdown("""
    # A análise quantitativa confirma a **desigualdade digital** e a **importância da infraestrutura** para o engajamento escolar.
    # """)

    # st.markdown("#### 5.1 Sugestão de Hipóteses e Conclusões (Requisito 3 e 4.1)")
    # st.markdown("""
    # 1.  **Hipótese Confirmada:** A infraestrutura tecnológica é um fator preditivo do número de matrículas, com a rede privada apresentando uma vantagem desproporcional.
    # 2.  **Causa-Raiz:** A diferença de **7.1x** na média de EQP/Aluno entre redes é a principal evidência da disparidade de investimento.
    # 3.  **Insight:** A Região **Sul** (conforme a análise regional) é a que apresenta a maior diferença absoluta na média de EQP/Aluno, indicando que o problema não está restrito às regiões mais carentes, mas é uma questão de política de investimento por dependência administrativa.
    # """)

    # st.markdown("#### 5.2 Avaliação de Suposições e Ferramentas (Requisito 4.2 e 4.3)")
    # st.markdown("""
    # - **Suposições:** Assumimos que as variáveis de contagem de equipamentos são representativas da qualidade da infraestrutura. A análise estatística (Regressão Linear) validou a relação entre as variáveis.
    # - **Ferramentas:** O projeto utilizou **Pandas** para limpeza e engenharia de *features*, **Matplotlib** e **Seaborn** para visualização e **Scikit-learn** para a modelagem estatística (Regressão Linear), conforme solicitado pelo professor.
    # """)

    # st.markdown("#### 5.3 Coleta de Dados Adicionais (Requisito 4.4)")
    # st.markdown("""
    # Para um estudo mais aprofundado, sugerimos a coleta de dados adicionais:
    # - **Dados Qualitativos:** Nível de treinamento dos professores no uso dessas tecnologias.
    # - **Dados de Uso:** Frequência e forma como os equipamentos são integrados ao currículo pedagógico.
    # - **Dados de Desempenho:** Cruzamento com notas do IDEB para verificar se a infraestrutura se traduz em melhoria de desempenho.
    # """)

# =====================================================
# 🎓 ABA 1 - CURSOS TÉCNICOS
# =====================================================
with aba1:
    st.subheader("🎓 Distribuição de Cursos Técnicos por Área")
    df_tecnicos = df_parcial[df_parcial["TIPO_ESCOLA"] == "Educação Técnica"].copy()

    if df_tecnicos.empty:
        st.info("Nenhum curso técnico encontrado nos filtros aplicados.")
    else:
        df_area = (
            df_tecnicos.drop_duplicates(subset=["CO_ENTIDADE", "NO_AREA_CURSO_PROFISSIONAL"])
            .groupby("NO_AREA_CURSO_PROFISSIONAL")["CO_ENTIDADE"]
            .nunique()
            .reset_index()
            .rename(columns={"NO_AREA_CURSO_PROFISSIONAL": "Área do Curso", "CO_ENTIDADE": "Quantidade"})
            .sort_values("Quantidade", ascending=False)
        )
        grafico_area = (
            alt.Chart(df_area)
            .mark_bar(size=50)
            .encode(
                x="Quantidade:Q",
                y=alt.Y("Área do Curso:N", sort="-x"),
                tooltip=["Área do Curso", "Quantidade"],
                color="Área do Curso:N"
            )
            .properties(title="Quantidade de Escolas por Área de Curso Técnico")
        )
        st.altair_chart(grafico_area, use_container_width=True)
        st.dataframe(df_area, use_container_width=True, hide_index=True)

# =====================================================
# 🏫 ABA 2 - DEPENDÊNCIA ADMINISTRATIVA
# =====================================================
with aba2:
    st.subheader("🏫 Quantidade de Escolas por Dependência Administrativa")
    df_dep = df_parcial["DEPENDENCIA_NOME"].value_counts().reset_index()
    df_dep.columns = ["Dependência", "Quantidade"]
    grafico_dep = (
        alt.Chart(df_dep)
        .mark_bar(size=60)
        .encode(x="Dependência:N", y="Quantidade:Q", tooltip=["Dependência", "Quantidade"], color="Dependência:N")
        .properties(title="Escolas por Tipo de Dependência Administrativa")
    )
    st.altair_chart(grafico_dep, use_container_width=True)
    st.dataframe(df_dep, use_container_width=True, hide_index=True)

# =====================================================
# 📍 ABA 3 - MUNICÍPIOS
# =====================================================
with aba3:
    st.subheader("📍 Distribuição de Escolas por Município")

    if "NO_MUNICIPIO" in df_parcial.columns:
        df_mun = (
            df_parcial.drop_duplicates(subset=["CO_ENTIDADE"])["NO_MUNICIPIO"]
            .value_counts()
            .reset_index()
        )
        df_mun.columns = ["Município", "Quantidade"]
        df_mun["Município"] = df_mun["Município"].astype(str)
        df_mun["Quantidade"] = df_mun["Quantidade"].astype(int)

        if not df_mun.empty:
            grafico_mun = (
                alt.Chart(df_mun)
                .mark_bar(size=60)
                .encode(
                    x=alt.X("Município:N", sort="-y", title="Município"),
                    y=alt.Y("Quantidade:Q", title="Quantidade de Escolas"),
                    tooltip=["Município", "Quantidade"],
                    color=alt.Color("Município:N", scale=alt.Scale(scheme="tableau10"))
                )
                .properties(title="Quantidade de Escolas por Município")
            )
            st.altair_chart(grafico_mun, use_container_width=True)
            st.dataframe(df_mun, use_container_width=True, hide_index=True)
        else:
            st.info("Nenhum município encontrado com os filtros aplicados.")
    else:
        st.warning("⚠️ Coluna 'NO_MUNICIPIO' não encontrada nos dados.")

# =====================================================
# 🏗️ ABA 4 - ESTRUTURA
# =====================================================
with aba4:
    st.subheader("🏗️ Estrutura das Escolas Filtradas")

    df_tecnicas = df_parcial[df_parcial["TIPO_ESCOLA"] == "Educação Técnica"].drop_duplicates(subset=["CO_ENTIDADE"])
    df_basicas = df_parcial[df_parcial["TIPO_ESCOLA"] == "Educação Básica"].drop_duplicates(subset=["CO_ENTIDADE"])

    # Gráfico pizza
    contagem_tipo = df_parcial["TIPO_ESCOLA"].value_counts().reset_index()
    contagem_tipo.columns = ["Tipo de Escola", "Quantidade"]
    grafico_tipo = (
        alt.Chart(contagem_tipo)
        .mark_arc(innerRadius=60)
        .encode(
            theta="Quantidade:Q",
            color="Tipo de Escola:N",
            tooltip=["Tipo de Escola", "Quantidade"]
        )
        .properties(title="Distribuição de Escolas: Educação Básica x Técnica")
    )
    st.altair_chart(grafico_tipo, use_container_width=True)

    # Escolas técnicas
    st.markdown("### 🎓 Escolas de Educação Técnica")
    st.dataframe(
        df_tecnicas[["CO_ENTIDADE", "NO_ENTIDADE", "NO_MUNICIPIO", "DEPENDENCIA_NOME"]],
        use_container_width=True, hide_index=True
    )
    st.caption(f"Total de escolas técnicas: {len(df_tecnicas)}")

    # Escolas básicas
    st.markdown("### 📚 Escolas de Educação Básica")
    st.dataframe(
        df_basicas[["CO_ENTIDADE", "NO_ENTIDADE", "NO_MUNICIPIO", "DEPENDENCIA_NOME"]],
        use_container_width=True, hide_index=True
    )
    st.caption(f"Total de escolas básicas: {len(df_basicas)}")

    # Exportar
    buffer = BytesIO()
    df_export = pd.concat([df_basicas, df_tecnicas])
    df_export.to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)
    st.download_button(
        label="📥 Exportar lista completa (básicas e técnicas)",
        data=buffer,
        file_name=f"escolas_filtradas_{uf}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# =====================================================
# 📈 ABA 5 - INDICADORES NACIONAIS
# =====================================================
# with aba5:
#     st.subheader("📈 Indicadores Nacionais por Região")

#     # 1️⃣ Distribuição de Matrículas por Região
#     dados = pd.DataFrame({
#         "Região": ["Sudeste", "Nordeste", "Sul", "Centro-Oeste", "Norte"],
#         "Percentual": [42.5, 25.0, 15.0, 9.0, 8.5]
#     })

#     grafico_regiao = (
#         alt.Chart(dados)
#         .mark_bar(size=60, cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
#         .encode(
#             x=alt.X("Região:N", sort="-y", title="Região"),
#             y=alt.Y("Percentual:Q", title="Percentual de Matrículas (%)"),
#             color=alt.Color("Região:N", scale=alt.Scale(scheme="tableau20")),
#             tooltip=["Região", "Percentual"]
#         )
#         .properties(title="Distribuição de Matrículas por Região (Censo Escolar 2024)")
#     )

#     st.altair_chart(grafico_regiao, use_container_width=True, theme="streamlit")

#     # 2️⃣ Comparativo entre São Paulo e o conjunto Norte + Centro-Oeste
#     dados_sp = pd.DataFrame({
#         "Categoria": ["São Paulo", "Norte + Centro-Oeste"],
#         "Percentual de Matrículas (%)": [22.0, 17.5]
#     })

#     grafico_sp = (
#         alt.Chart(dados_sp)
#         .mark_bar(size=80, cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
#         .encode(
#             x=alt.X("Categoria:N", sort="-y"),
#             y=alt.Y("Percentual de Matrículas (%):Q"),
#             color=alt.Color("Categoria:N", scale=alt.Scale(scheme="set2")),
#             tooltip=["Categoria", "Percentual de Matrículas (%)"]
#         )
#         .properties(title="Comparativo: São Paulo vs Norte + Centro-Oeste")
#     )

#     st.altair_chart(grafico_sp, use_container_width=True, theme="streamlit")
#     st.markdown("<br><br>", unsafe_allow_html=True)

#     # 3️⃣ Distribuição Urbana x Rural (Gráfico de Pizza)
#     dados_local = pd.DataFrame({
#         "Localização": ["Urbana", "Rural"],
#         "Percentual": [95.1, 4.9]
#     })

#     grafico_local = (
#         alt.Chart(dados_local)
#         .mark_arc(innerRadius=50, outerRadius=120)
#         .encode(
#             theta=alt.Theta("Percentual:Q", title="Percentual de Matrículas"),
#             color=alt.Color("Localização:N", scale=alt.Scale(scheme="tableau10")),
#             tooltip=["Localização", "Percentual"]
#         )
#         .properties(title="Distribuição das Matrículas: Urbana vs Rural")
#     )

#     st.altair_chart(grafico_local, use_container_width=True, theme="streamlit")

#     # Texto explicativo
#     st.markdown("""
#     ### 📊 Análise dos Indicadores Nacionais
#     - O **Sudeste** concentra **42,5%** das matrículas, reforçando o peso econômico e populacional da região.  
#     - O **Nordeste** é a **segunda maior região** em número de matrículas (25%), destacando avanços na cobertura educacional.  
#     - Apenas **4,9% das matrículas** ocorrem em áreas **rurais**, revelando desafios de acesso fora dos centros urbanos.  
#     - **São Paulo**, isoladamente, concentra **22% das matrículas do país**, superando a soma das regiões **Norte e Centro-Oeste**.  
#     """)

# =====================================================
# 📊 ABA 6 - ESTATÍSTICA DESCRITIVA
# =====================================================
# with aba6:
    st.header("📊 Análise Estatística Descritiva")

    # 🔹 Seleciona colunas numéricas e categóricas
    colunas_numericas = [col for col in df_parcial.columns if df_parcial[col].dtype in ['int64', 'float64']]
    colunas_categoricas = [col for col in df_parcial.columns if df_parcial[col].dtype == 'object']

    # -----------------------------
    # 🧮 Resumo estatístico numérico
    # -----------------------------
    st.subheader("📈 Estatísticas Descritivas das Variáveis Numéricas")
    if colunas_numericas:
        resumo = df_parcial[colunas_numericas].describe().T
        resumo["Coef. Variação (%)"] = (df_parcial[colunas_numericas].std() / df_parcial[colunas_numericas].mean()) * 100
        st.dataframe(resumo.style.format("{:.2f}"), use_container_width=True)

        st.markdown("**📖 Interpretação rápida:**")
        st.markdown("""
        - A **média** mostra o valor típico de cada variável.
        - O **desvio padrão** mede a dispersão — quanto maior, mais heterogêneos são os valores.
        - O **coeficiente de variação (CV)** permite comparar a variação relativa entre variáveis (CV > 30% indica alta variabilidade).
        """)

        # Permite o usuário escolher uma variável para visualizar
        variavel_num = st.selectbox("📊 Escolha uma variável numérica para visualizar:", colunas_numericas)
        if variavel_num:
            col1, col2 = st.columns(2)
            with col1:
                grafico_box = (
                    alt.Chart(df_parcial)
                    .mark_boxplot(color="#0F9ED5")
                    .encode(y=alt.Y(f"{variavel_num}:Q", title=variavel_num))
                    .properties(title=f"Distribuição (Boxplot) - {variavel_num}", height=300)
                )
                st.altair_chart(grafico_box, use_container_width=True)
            with col2:
                grafico_hist = (
                    alt.Chart(df_parcial)
                    .mark_bar()
                    .encode(
                        x=alt.X(f"{variavel_num}:Q", bin=alt.Bin(maxbins=30), title=variavel_num),
                        y="count()",
                        color=alt.value("#F57C00")
                    )
                    .properties(title=f"Histograma - {variavel_num}", height=300)
                )
                st.altair_chart(grafico_hist, use_container_width=True)

    else:
        st.warning("Nenhuma variável numérica disponível para análise.")

    # -----------------------------
    # 🔹 Frequência das variáveis categóricas
    # -----------------------------
    st.subheader("🧩 Distribuição das Variáveis Categóricas")
    if colunas_categoricas:
        var_categ = st.selectbox("📋 Escolha uma variável categórica:", colunas_categoricas)
        if var_categ:
            freq = df_parcial[var_categ].value_counts(dropna=False).reset_index()
            freq.columns = [var_categ, "Frequência"]
            freq["%"] = (freq["Frequência"] / freq["Frequência"].sum() * 100).round(2)
            st.dataframe(freq, use_container_width=True, hide_index=True)

            grafico_cat = (
                alt.Chart(freq)
                .mark_bar(size=40, cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
                .encode(
                    x=alt.X(var_categ + ":N", sort="-y"),
                    y=alt.Y("Frequência:Q"),
                    color=alt.Color(var_categ + ":N", scale=alt.Scale(scheme="tableau10")),
                    tooltip=[var_categ, "Frequência", "%"]
                )
                .properties(title=f"Distribuição da variável: {var_categ}", height=350)
            )
            st.altair_chart(grafico_cat, use_container_width=True)

            st.markdown("**📖 Interpretação rápida:**")
            st.markdown(f"""
            - A variável **{var_categ}** apresenta **{len(freq)} categorias distintas**.
            - A categoria mais frequente é **{freq.iloc[0,0]}** com **{freq.iloc[0,1]} ocorrências**, 
              representando **{freq.iloc[0,2]}%** do total.
            """)
    else:
        st.info("Sem variáveis categóricas disponíveis para exibição.")

    st.markdown("---")
    st.info("💡 Dica: Combine esta aba com filtros no painel lateral para refinar a análise estatística por região, município ou tipo de escola.")
