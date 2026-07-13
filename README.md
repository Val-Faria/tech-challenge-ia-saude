# 🧠 Tech Challenge - Fase 2
## Otimização de Modelos de Diagnóstico e Interpretação com IA Generativa

### FIAP - Pós-Graduação em IA para Desenvolvedores

[![Abrir no Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Val-Faria/tech-challenge-ia-saude/blob/Nirton/02_otimizacao_hiperparametros_ag.ipynb)

## Entregáveis

- [Relatório técnico final em PDF](./Relatorio%20Fase%202.pdf)
- [Notebook executado no Google Colab](./02_otimizacao_hiperparametros_ag.ipynb)
- [Modelo Random Forest otimizado](./models/random_forest_otimizado.joblib)
- Vídeo de demonstração: link será adicionado após a gravação e validação final.

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
- divisão estratificada em treino, validação e teste
- ponderação das classes no Random Forest

---

# 🏗 Arquitetura da Solução

Fluxo geral do projeto:

```mermaid
flowchart TD
    A[Dataset clínico] --> B[Limpeza e divisão estratificada]
    B --> C[Random Forest baseline]
    B --> D[Algoritmo Genético]
    D --> E[Validação cruzada estratificada - 3 folds]
    E --> F[Melhores hiperparâmetros]
    F --> G[Modelo otimizado]
    G --> H[Avaliação final no conjunto de teste]
    G --> I[Interpretação com LLM]
    G --> J[Artefato Joblib]
    J --> K[API FastAPI]
    K --> L[Logs JSON]
    K --> M[Métricas Prometheus]
    K --> N[Google Cloud Run]
    N --> O[Escalabilidade automática]
```

O conjunto de teste permanece isolado durante a busca de hiperparâmetros. A API
carrega apenas o artefato final e não participa do processo de treinamento.

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
- crossover de dois pontos
- mutação uniforme inteira com limites por gene
- elitismo real com preservação do melhor indivíduo
- cache de fitness
- histórico completo das gerações
- logging em arquivo e no console

---

# 🧪 Experimentos

Foram realizados três experimentos variando:

- tamanho da população
- número de gerações
- probabilidade de crossover
- taxa de mutação

As configurações usam seeds distintas. Cada indivíduo é avaliado por validação
cruzada estratificada com três folds sobre o conjunto de treino. A seleção é
lexicográfica: maximiza primeiro o Recall médio e utiliza F1-Score e AUC para
resolver empates. O desvio-padrão do Recall é registrado como medida de
estabilidade.

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
tech-challenge-ia-saude/
├── api/
│   └── main.py
├── cloudrun/
│   └── service.yaml
├── models/
│   └── random_forest_otimizado.joblib
├── tests/
├── 02_otimizacao_hiperparametros_ag.ipynb
├── Dockerfile
├── Relatorio Fase 2.docx
├── Relatorio Fase 2.pdf
├── README.md
└── requirements.txt
```

As pastas `dataset/` e `reports/` são criadas automaticamente pelo notebook.

---

# ▶ Como Executar

### 1. Clonar o repositório

```bash
git clone --branch Nirton --single-branch https://github.com/Val-Faria/tech-challenge-ia-saude.git
```

### 2. Acessar a pasta do projeto

```bash
cd tech-challenge-ia-saude
```

### 3. Instalar as dependências

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

No Linux/macOS, ative o ambiente com `source .venv/bin/activate`.

### 4. Configurar a chave da OpenAI

Defina a variável de ambiente `OPENAI_API_KEY` ou informe a chave quando o notebook solicitar.

### 5. Abrir o notebook

Abra o arquivo:

```
02_otimizacao_hiperparametros_ag.ipynb
```

preferencialmente utilizando o **Google Colab** ou o **Jupyter Notebook**.

No Colab, use **Ambiente de execução > Executar tudo**. O notebook instala as
dependências essenciais, clona o repositório e baixa o dataset automaticamente.

### 6. Executar os testes

```bash
pytest -q
```

### 7. Executar a API local

O modelo treinado já está versionado em `models/random_forest_otimizado.joblib`.
O notebook pode ser executado novamente para reproduzir o treinamento e substituir
o artefato, caso necessário.

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8080
```

- documentação interativa: `http://localhost:8080/docs`
- saúde: `http://localhost:8080/health`
- métricas: `http://localhost:8080/metrics`

O arquivo `cloudrun/service.yaml` configura escalabilidade automática entre zero e
cinco instâncias. A implantação em nuvem permanece opcional.



# 👥 Integrantes

- Marcelo Viana de Araujo
- Rodrigo de Moraes Filomeno
- Nirton Afonso de Oliveira Filho
- Valkiria Nonato de Faria
- Benicio Antonio Cardoso

---

# 📚 Instituição

FIAP

Pós-Graduação em Inteligência Artificial para Desenvolvedores

Tech Challenge – Fase 2

2026
