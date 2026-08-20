![capa](git_assets/teste.png)

# O impacto do gênero na inclusão nas áreas STEM: uma análise com Machine Learning sobre dados do Enem
Machine Learning aplicado a +23 milhões de registros do Enem (2018-2023) para identificar quais fatores influenciam na entrada de mulheres nas áreas de exatas: e o quanto o gênero, isoladamente, ainda pesa nessa equação.

## O que esse projeto responde

A presença feminina em cursos de Ciências, Tecnologia, Engenharia e Matemática (STEM) segue desproporcional à masculina, mesmo com a paridade de acesso ao ensino superior já alcançada globalmente. Este projeto usa os microdados públicos do Enem para investigar, com evidência estatística, **o quanto o gênero é um fator determinante nesse desequilíbrio — mesmo quando outras variáveis socioeconômicas são levadas em conta.**

## O principal achado

**O gênero feminino segue sendo o 2º fator mais relevante para reduzir a probabilidade de um perfil STEM — atrás apenas da renda familiar**. Esse efeito se intensificou visivelmente em 2020, no início da pandemia de Covid-19.

![Direção do impacto de gênero no Perfil STEM](git_assets/impactogenero.png)
*Impacto médio (SHAP) do gênero na probabilidade de perfil STEM, por ano. Barras à esquerda = mulheres reduzem a chance; à direita = homens aumentam a chance.*

> Esse tipo de achado tem aplicação direta para quem desenha políticas públicas de incentivo, programas de bolsas de estudo ou iniciativas de diversidade em tecnologia — o modelo não só confirma a disparidade, como aponta quais outras variáveis pesam junto (renda familiar, escolaridade dos pais), o que ajuda a priorizar onde intervir.

## Como cheguei lá
<ol>
<li> **Dados:** microdados oficiais do Enem (2018-2023), disponibilizados pelo INEP — datasets de até 2,47 GB por ano, totalizando milhões de registros de participantes.
<li> **Pipeline de processamento em escala:** para viabilizar o processamento em uma máquina pessoal, os dados foram tratados em chunks, convertidos para o formato Parquet e tiveram os tipos numéricos otimizados (downcast), reduzindo drasticamente o uso de memória sem perda de informação relevante. </li>
<li> **Modelagem preditiva:** comparação de 4 algoritmos ensemble (Random Forest, XGBoost, LightGBM, CatBoost) para classificar participantes com perfil STEM, usando notas de Matemática e Ciências da Natureza como proxy. </li>
<li> **Otimização e validação:** tuning de hiperparâmetros com Optuna e avaliação por ROC-AUC e F1-Score, testando 4 cenários diferentes de rigor na definição de "perfil STEM".</li>
<li> **Interpretabilidade:** uso do framework SHAP para abrir a "caixa-preta" do modelo e entender exatamente o peso de cada variável (gênero, renda, escolaridade dos pais, raça, região) na previsão — e como isso evoluiu ano a ano. </li>
</ol>

## Resultados principais
| Modelo escolhido | CatBoost |
|---|---|
| Cenário mais rigoroso testado (T3: nota acima da mediana em Matemática **e** Ciências da Natureza) | ROC-AUC de 0,7707 |
| Fatores mais relevantes (nessa ordem) | Renda familiar → Gênero → Escolaridade da mãe → Escolaridade do pai |
| Maior disparidade de gênero observada | Matemática, com gap crescente entre 2018 (43,2 pontos) e 2023 (58,1 pontos) |


## Stack técnica

`Python` `Pandas` `Scikit-learn` `XGBoost` `LightGBM` `CatBoost` `SHAP` `Optuna` `Apache Parquet` `Matplotlib` `Seaborn`

## Estrutura do projeto

```
├── data/
│   ├── raw/            # Microdados brutos do INEP (não versionados aqui por tamanho)
│   └── processed/       # Dados tratados em formato Parquet
├── notebooks/            # Notebooks de processamento, modelagem e análise
├── imgs/                 # Gráficos gerados pelo projeto
├── texto/                 # Documentação e artigo acadêmico completo
├── requirements.txt
└── README.md
```

## Como rodar

```bash
git clone https://github.com/ClaudiaSobral/enem-genero-stem-ml.git
cd enem-genero-stem-ml
pip install -r requirements.txt
```

Os microdados do Enem podem ser baixados diretamente no [portal do INEP](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem) e devem ser colocados em `data/raw/`. Os notebooks em `notebooks/` seguem a ordem: processamento → modelagem → interpretabilidade (SHAP).

## Limitações

Os modelos utilizados permitem identificar correlação, não causalidade — ou seja, apontam quais fatores se associam a um perfil STEM, mas não isolam definitivamente por que isso acontece. Para uma leitura causal mais robusta, seria necessário incorporar dados adicionais sobre a trajetória escolar e vivência das alunas ao longo do Ensino Médio.

## Aprofundamento

Este projeto foi desenvolvido como Trabalho de Conclusão de Curso do MBA em Data Science e Analytics (USP/Esalq) sob orientação do professor Manoel Flavio Leal. O artigo completo, com revisão de literatura, discussão metodológica detalhada e referências, está disponível aqui.

## Sobre mim

Sou a Claudia Sobral, ex-animadora de TV migrando para Dados — trago dessa trajetória a capacidade de transformar números em histórias claras. [claudiasobral.com](https://claudiasobral.com/) · [LinkedIn](https://www.linkedin.com/in/claudia-sobral/)
