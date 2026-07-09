# Tech Challenge — IA aplicada à Saúde

Este projeto tem como objetivo desenvolver uma solução experimental de Machine Learning para apoio à classificação de registros clínicos relacionados à tireoide, com foco na identificação de casos de hipotireoidismo. A proposta combina modelos supervisionados, otimização de hiperparâmetros com Algoritmo Genético e interpretação dos resultados em linguagem natural, mantendo caráter acadêmico e não substitutivo à avaliação de profissionais da saúde.

## Objetivo do Projeto

O objetivo principal é construir e otimizar um modelo de classificação capaz de identificar registros positivos para hipotireoidismo, priorizando a métrica de **recall**. Essa escolha está alinhada ao contexto clínico do problema, em que a redução de falsos negativos é especialmente relevante, pois casos positivos não identificados podem representar maior risco em uma aplicação de triagem ou apoio diagnóstico.

## Dataset

O projeto utiliza o dataset `hypothyroid_final.csv`, contendo registros clínicos e laboratoriais relacionados à tireoide. O conjunto original possui informações numéricas e categóricas, incluindo variáveis como idade, sexo, indicadores clínicos, exames laboratoriais e a variável-alvo `target`.

A variável-alvo foi tratada como um problema de classificação binária:

```text
0 = negative
1 = hypothyroid
```

Após o carregamento e a limpeza inicial, o dataset utilizado na modelagem ficou com:

```text
3.086 registros
25 variáveis preditoras
1 variável-alvo
```

## Etapas Concluídas

### 1. Preparação do Ambiente

Foram configuradas as bibliotecas necessárias para análise de dados, visualização, modelagem, avaliação de métricas e implementação do Algoritmo Genético. A biblioteca `DEAP` foi utilizada para estruturar o processo evolutivo de otimização dos hiperparâmetros.

Também foi definida uma seed global para aumentar a reprodutibilidade dos experimentos:

```python
RANDOM_STATE = 42
```

### 2. Carregamento dos Dados

O dataset foi carregado automaticamente a partir de uma fonte no GitHub, permitindo a execução do projeto tanto no Google Colab quanto em ambiente local.

O carregamento inicial retornou:

```text
3.163 linhas
26 colunas
```

### 3. Limpeza dos Dados

A limpeza adotou uma abordagem conservadora, adequada ao contexto médico. Foram removidos registros duplicados para reduzir viés amostral, enquanto os valores ausentes foram preservados para tratamento posterior dentro do pipeline de modelagem.

Após a remoção de duplicados e a filtragem das classes de interesse, o dataset final passou a conter:

```text
3.086 registros
```

### 4. Separação entre Features e Target

As variáveis preditoras foram separadas da variável-alvo:

```python
X = dados_limpos.drop(columns=[TARGET_COLUMN])
y = dados_limpos[TARGET_COLUMN].map({
    NEGATIVE_LABEL: 0,
    POSITIVE_LABEL: 1
})
```

O mapeamento utilizado foi:

```text
negative     → 0
hypothyroid  → 1
```

### 5. Divisão dos Dados

O dataset foi dividido em três conjuntos independentes:

```text
Treino:     70%
Validação: 15%
Teste:     15%
```

A divisão foi feita com estratificação da variável-alvo, preservando a proporção entre classes nos conjuntos.

Resumo da divisão:

```text
Treino:     2.160 registros | 99 positivos
Validação:    463 registros | 21 positivos
Teste:        463 registros | 21 positivos
```

Essa separação permite que o modelo seja treinado no conjunto de treino, otimizado no conjunto de validação e posteriormente avaliado no conjunto de teste.

### 6. Pré-processamento

O pré-processamento foi construído com `Pipeline` e `ColumnTransformer`, garantindo que imputação e transformação de variáveis fossem aplicadas corretamente dentro do fluxo de treinamento.

As variáveis foram separadas em:

```text
Variáveis numéricas:
age, TSH, T3, TT4, T4U, FTI, TBG

Variáveis categóricas:
sex, on_thyroxine, query_on_thyroxine, on_antithyroid_medication,
thyroid_surgery, query_hypothyroid, query_hyperthyroid, pregnant,
sick, tumor, lithium, goitre, TSH_measured, T3_measured,
TT4_measured, T4U_measured, FTI_measured, TBG_measured
```

Para as variáveis numéricas, foi utilizada imputação pela mediana. Para as variáveis categóricas, foi utilizada imputação pelo valor mais frequente e codificação com One-Hot Encoding.

### 7. Modelo Base

O modelo escolhido para otimização foi o **Random Forest**, encapsulado em um pipeline com o pré-processador.

Configuração inicial do modelo:

```python
RandomForestClassifier(
    n_estimators=400,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
```

O uso de `class_weight="balanced_subsample"` foi adotado para lidar com o desbalanceamento entre as classes.

### 8. Avaliação Inicial do Modelo

Antes da otimização com Algoritmo Genético, o modelo Random Forest foi treinado e avaliado no conjunto de validação.

Resultado inicial em validação:

```text
Accuracy:  0.991
Precision: 0.870
Recall:    0.952
F1-score:  0.909
AUC:       0.986
```

Esse resultado mostrou que o modelo base já apresentava desempenho elevado, especialmente em recall, métrica principal do projeto.

### 9. Otimização com Algoritmo Genético

A etapa seguinte consistiu na otimização dos hiperparâmetros do Random Forest utilizando Algoritmo Genético.

Os genes definidos para cada indivíduo foram:

```text
n_estimators
max_depth
min_samples_split
min_samples_leaf
max_features
```

O espaço de busca foi definido como:

```python
PARAM_BOUNDS = {
    "n_estimators": (50, 350),
    "max_depth": (3, 20),
    "min_samples_split": (2, 12),
    "min_samples_leaf": (1, 10),
    "max_features": ["sqrt", "log2"]
}
```

A função fitness foi definida com foco em maximizar o **recall** no conjunto de validação. Além do recall, também foram calculadas as métricas de accuracy, precision, F1-score e AUC para análise complementar.

### 10. Experimentos com Algoritmo Genético

Foram executados três experimentos com diferentes configurações evolutivas.

#### Experimento 1 — Configuração Base

```text
População: 20
Gerações: 10
Crossover: 0.7
Mutação: 0.2
```

Resultado:

```text
Melhor Recall: 0.9524
Genes vencedores: [107, 3, 6, 4, 0]
```

#### Experimento 2 — Alta População

```text
População: 40
Gerações: 10
Crossover: 0.8
Mutação: 0.1
```

Resultado:

```text
Melhor Recall: 0.9524
Genes vencedores: [125, 10, 3, 3, 1]
```

#### Experimento 3 — Alta Mutação

```text
População: 20
Gerações: 10
Crossover: 0.6
Mutação: 0.4
```

Resultado:

```text
Melhor Recall: 0.9524
Genes vencedores: [260, 17, 2, 2, 1]
```

## Resultado Inicial da Otimização

Os três experimentos com Algoritmo Genético atingiram o mesmo recall de validação, `0.9524`, apesar de utilizarem configurações diferentes de população, mutação e hiperparâmetros do Random Forest. Esse resultado sugere que o modelo alcançou um possível platô para a métrica de recall no conjunto de validação, identificando aproximadamente 20 dos 21 casos positivos. No entanto, como a otimização considerou apenas o recall, ainda é necessário comparar os modelos por precision, F1-score e AUC, além de avaliar a melhor configuração no conjunto de teste para confirmar sua capacidade de generalização.

Como os três experimentos atingiram o mesmo recall de validação, a escolha da melhor configuração não deve se basear apenas nesse valor. A próxima análise deve verificar se os genes vencedores mantêm equilíbrio nas demais métricas, especialmente precision, F1-score e AUC, e confirmar o desempenho da configuração selecionada no conjunto de teste, que ainda não foi utilizado durante a otimização.

## Tecnologias Utilizadas

```text
Python
Pandas
NumPy
Scikit-learn
Matplotlib
DEAP
Joblib
Google Colab
```

## Principais Técnicas Aplicadas

```text
Limpeza de dados
Remoção de duplicados
Tratamento de dados ausentes via Pipeline
One-Hot Encoding
Random Forest
Class weighting
Treino, validação e teste estratificados
Algoritmo Genético
Otimização de hiperparâmetros
Avaliação por recall, precision, F1-score e AUC
```

## Status Atual

Até o momento, foram concluídas as seguintes etapas:

```text
Carregamento do dataset
Limpeza inicial dos dados
Separação entre X e y
Divisão estratificada em treino, validação e teste
Construção do pipeline de pré-processamento
Treinamento inicial com Random Forest
Avaliação baseline em validação
Definição do espaço de busca dos hiperparâmetros
Implementação da função fitness
Implementação do Algoritmo Genético
Execução de três experimentos evolutivos
Análise inicial dos resultados de recall
```

## Próximos Passos

As próximas etapas previstas são:

```text
Comparar os genes vencedores com métricas complementares
Selecionar a melhor configuração considerando recall e equilíbrio geral
Avaliar o modelo final no conjunto de teste
Gerar matriz de confusão
Analisar falsos negativos
Considerar F2-score ou ajuste de threshold em uma próxima iteração
```

## Observação Importante

Este projeto possui finalidade acadêmica e experimental. Os resultados obtidos devem ser interpretados como apoio à análise e estudo de técnicas de Machine Learning aplicadas à saúde, não substituindo avaliação médica, decisão clínica ou julgamento profissional.

