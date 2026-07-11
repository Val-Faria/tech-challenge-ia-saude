# 🧠 Tech Challenge - Fase 2
## Otimização de Modelos de Diagnóstico e Interpretação com IA Generativa

### FIAP - Pós-Graduação em IA para Desenvolvedores

---

## 📌 Objetivo

Este projeto tem como objetivo otimizar um modelo de Machine Learning para classificação de pacientes com suspeita de hipotireoidismo utilizando um Algoritmo Genético para seleção automática de hiperparâmetros de um Random Forest.

Além da otimização do modelo, foi integrada uma Large Language Model (LLM) da OpenAI para transformar as previsões do modelo em explicações compreensíveis, auxiliando profissionais de saúde na interpretação dos resultados.

> **Importante:** o sistema possui finalidade acadêmica e experimental. As previsões produzidas não substituem avaliação clínica, diagnóstico médico ou exames laboratoriais.

---

# 📂 Dataset

Foi utilizado o conjunto de dados **Hypothyroid Disease Dataset**, contendo registros clínicos de pacientes utilizados para classificação entre:

- Negative
- Hypothyroid

O conjunto passou pelas etapas de:

- limpeza
- tratamento de valores ausentes
- codificação de variáveis categóricas
- divisão treino/teste
- balanceamento das classes quando necessário

---

# 🏗 Arquitetura da Solução

Fluxo geral do projeto:

```text
Dataset
      │
      ▼
Pré-processamento
      │
      ▼
Random Forest (Baseline)
      │
      ▼
Algoritmo Genético
      │
      ▼
Melhores Hiperparâmetros
      │
      ▼
Modelo Otimizado
      │
      ▼
Avaliação
      │
      ▼
LLM (OpenAI)
      │
      ▼
Explicação em Linguagem Natural
```

---

# 🤖 Modelo de Machine Learning

Foi utilizado o algoritmo:

- Random Forest Classifier

O modelo baseline foi comparado com um modelo otimizado através de Algoritmo Genético.

As métricas utilizadas foram:

- Recall (métrica principal)
- Precision
- Accuracy
- F1-Score
- ROC AUC

---

# 🧬 Algoritmo Genético

O Algoritmo Genético foi desenvolvido para otimizar automaticamente os hiperparâmetros do Random Forest.

Os hiperparâmetros avaliados foram:

- n_estimators
- max_depth
- min_samples_split
- min_samples_leaf
- max_features

Características implementadas:

- população inicial aleatória
- seleção por torneio
- crossover uniforme
- mutação
- elitismo
- cache de fitness
- histórico completo das gerações

---

# 🧪 Experimentos

Foram realizados três experimentos variando:

- tamanho da população
- número de gerações
- probabilidade de crossover
- taxa de mutação

Ao final, foi selecionado automaticamente o melhor indivíduo considerando o maior Recall.

---

# 📊 Resultados

O projeto gera automaticamente:

## Figuras

- Evolução do melhor Recall
- Evolução do Recall médio
- Comparação Baseline × Modelo Otimizado
- Matrizes de Confusão
- Importância das Variáveis

## Arquivos CSV

- comparação dos experimentos
- histórico das gerações
- métricas do baseline
- ganhos do modelo otimizado
- importância das variáveis
- interpretação da LLM
- avaliação da resposta da LLM

---

# 💬 IA Generativa

Após a classificação, o projeto envia para uma LLM:

- classe prevista
- probabilidades
- principais variáveis
- métricas do modelo

A IA produz uma interpretação em linguagem natural contendo:

- explicação da previsão
- interpretação das probabilidades
- limitações do modelo
- recomendação de avaliação clínica

---

# 🛠 Tecnologias Utilizadas

- Python
- Google Colab
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- OpenAI API
- Joblib

---

# 📁 Estrutura do Projeto

```text
TechChallenge_Tireoide
│
├── dataset/
├── models/
├── reports/
│   ├── figures/
│   └── results/
├── notebooks/
└── README.md
```

---

# ▶ Como Executar

### 1. Clonar o repositório

```bash
git clone https://github.com/Val-Faria/tech-challenge-ia-saude.git
```

### 2. Acessar a pasta do projeto

```bash
cd tech-challenge-ia-saude
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4. Configurar a chave da OpenAI

Defina a variável de ambiente `OPENAI_API_KEY` ou informe a chave quando o notebook solicitar.

### 5. Abrir o notebook

Abra o arquivo:

```
02_otimizacao_hiperparametros_ag.ipynb
```

preferencialmente utilizando o **Google Colab** ou o **Jupyter Notebook**.



# 👥 Integrantes

- Marcelo Viana de Araujo
- Valéria Faria
- Benicio
- Nirton Afonso

---

# 📚 Instituição

FIAP

Pós-Graduação em Inteligência Artificial para Desenvolvedores

Tech Challenge – Fase 2

2026
