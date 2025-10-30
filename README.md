# Desafio Projeto Integrador - Unifacisa
## Machine Learning, IA e IA Generativa

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-CPU-ff6f00)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU-ee4c2c)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-blue)](https://xgboost.readthedocs.io/)

---

## 👥 Equipe

- **Eduardo Nascimento de Oliveira**
- **Pedro Henrique de Araújo**
- **Isaque Brito Araújo**

---

## 📋 Sobre o Projeto

Este repositório contém as soluções completas para o Desafio Projeto Integrador da disciplina de IA, ML e IA Generativa da Unifacisa, ministrada pelo Prof. Me. Ricardo Roberto de Lima. O projeto abrange exercícios práticos de Machine Learning e Deep Learning, do nível básico ao avançado, cobrindo **Classificação**, **Clusterização**, **Regressão**, **Sistemas de Recomendação** e **Visão Computacional**.

---

## 🎯 Questões Implementadas

### **📊 Parte 1 - Modelos de Classificação** (Questões resolvidas em sala)

#### **Questão 1 (Fácil) - Classificação de Flores Íris**
- **Dataset:** Iris (Scikit-Learn)
- **Objetivo:** Classificar três espécies de flores (Setosa, Versicolor, Virginica)
- **Features:** Comprimento e largura das pétalas e sépalas
- **Modelos:** KNN, Random Forest ou SVM
- **Split:** 80% treino, 20% teste
- **Análise:** Avaliação de acurácia e possíveis melhorias

#### **Questão 2 (Intermediário) - Detecção de Fraudes em Cartões de Crédito**
- **Dataset:** Transações financeiras (Kaggle)
- **Objetivo:** Identificar transações fraudulentas
- **Features:** Valor, localização, histórico do usuário, horário
- **Pré-processamento:** Tratamento de valores nulos e normalização
- **Modelos:** Árvores de Decisão, Random Forest, XGBoost
- **Métricas:** Precisão, Recall, F1-Score
- **Análise:** Técnicas para melhorar a detecção

---

### **🔍 Parte 2 - Modelos de Clusterização** (Questões resolvidas em sala)

#### **Questão 3 (Fácil) - Agrupamento de Clientes de E-commerce**
- **Dataset:** Dados de compras de clientes
- **Objetivo:** Segmentar clientes por comportamento de compra
- **Features:** Quantidade de compras, valor gasto, frequência
- **Algoritmo:** K-Means com Elbow Method
- **Visualização:** Gráfico de dispersão dos clusters
- **Análise:** Identificação de padrões por grupo

#### **Questão 4 (Intermediário) - Análise de Clusterização de Dados de Saúde**
- **Dataset:** Informações de pacientes
- **Objetivo:** Identificar padrões de saúde
- **Features:** Idade, IMC, pressão arterial, glicose, colesterol
- **Técnicas:** PCA (redução de dimensionalidade)
- **Algoritmos:** DBSCAN ou K-Means
- **Análise:** Interpretação dos clusters de saúde

---

### **🚀 Parte 3 - Modelos Avançados** (Questões 5-10)

#### **Questão 5 (Intermediário) - Diagnóstico de Doenças Cardíacas**
- **Dataset:** Heart Failure Prediction Dataset (Kaggle)
- **Objetivo:** Prever presença de doença cardíaca
- **Features:** Idade, pressão arterial, colesterol, frequência cardíaca
- **Modelos:** Random Forest e SVM (comparação)
- **Métricas:** Precisão, Recall, Curva ROC-AUC
- **Análise:** Modelo com melhor desempenho e variáveis mais impactantes

#### **Questão 6 (Avançado) - Previsão do Valor de Imóveis**
- **Dataset:** King County House Data
- **Objetivo:** Prever preços de imóveis
- **Features:** Localização, número de quartos, tamanho do terreno
- **Técnicas:** Feature Engineering
- **Modelos:** Regressão Linear, XGBoost, Redes Neurais Artificiais (ANNs)
- **Métricas:** RMSE e R²
- **Análise:** Modelo com menor erro e otimizações

#### **Questão 7 (Intermediário) - Recomendação de Produtos em Supermercado**
- **Dataset:** Transações de supermercado
- **Objetivo:** Sistema de recomendação baseado em associações
- **Features:** Itens comprados em conjunto
- **Algoritmo:** Apriori
- **Métricas:** Suporte, Confiança e Lift
- **Análise:** Regras de associação relevantes para aumentar vendas

#### **Questão 8 (Avançado) - Recomendação de Filmes com Filtragem Colaborativa**
- **Dataset:** MovieLens
- **Objetivo:** Sugerir filmes baseado em avaliações de usuários
- **Abordagens:** 
  - Filtragem Colaborativa (usuários e itens)
  - Autoencoders (Deep Learning)
- **Métricas:** RMSE e MAE
- **Análise:** Comparação de eficiência e melhorias no sistema

#### **Questão 9 (Avançado) - Classificação de Imagens de Raio-X com CNNs**
- **Dataset:** Chest X-ray Dataset
- **Objetivo:** Classificar doenças pulmonares (pneumonia vs saudável)
- **Modelo:** Rede Neural Convolucional (CNN)
- **Técnicas:** Data Augmentation para generalização
- **Métricas:** Precisão e Matriz de Confusão
- **Análise:** Desafios de treinamento e melhorias de desempenho

#### **Questão 10 (Intermediário) - Previsão de Vendas Mensais em Rede de Varejo**
- **Dataset:** Rossmann Store Sales (Kaggle)
- **Objetivo:** Prever vendas mensais de lojas
- **Features:** Gastos com publicidade, número de funcionários, promoções, sazonalidade
- **Técnicas:** Feature Engineering (variáveis binárias, extração mês/trimestre)
- **Modelos:** Regressão Linear, Árvore de Decisão, XGBoost
- **Métricas:** RMSE, MAE e R²
- **Análise:** Modelo com menor margem de erro e variáveis mais impactantes

---

## 🛠️ Tecnologias e Bibliotecas

### Bibliotecas Principais
```python
# Machine Learning e Análise de Dados
scikit-learn
pandas
numpy
xgboost

# Deep Learning
tensorflow-cpu
torch
torchvision
torchaudio
torchmetrics

# Visualização
matplotlib
seaborn
plotly

# Sistemas de Recomendação
mlxtend          # Para algoritmo Apriori
scikit-surprise  # Para filtragem colaborativa

# Processamento de Imagens
opencv-python
pillow
```

---

## 📦 Instalação

### Configuração do Ambiente

```bash
# Clone o repositório
git clone https://github.com/eduardo-oliveira-dev/questoes-ciencia-de-dados-ads.git
cd questoes-ciencia-de-dados-ads

# Crie um ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

### Instalação Manual das Bibliotecas

```python
# TensorFlow (versão CPU)
!pip install tensorflow-cpu

# PyTorch (versão CPU)
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Métricas para PyTorch
!pip install torchmetrics

# XGBoost
!pip install xgboost

# Bibliotecas de análise e visualização
!pip install scikit-learn pandas matplotlib seaborn numpy plotly

# Sistemas de Recomendação
!pip install mlxtend scikit-surprise

# Processamento de imagens
!pip install opencv-python pillow
```

## 🚀 Execução

### Jupyter Notebook

```bash
# Inicie o Jupyter Notebook
jupyter notebook

# Ou Jupyter Lab
jupyter lab
```

Navegue até a pasta `notebooks/` e execute os notebooks na ordem desejada.

### Google Colab

Os notebooks também podem ser executados no Google Colab. Basta fazer o upload dos arquivos e executar as células de instalação de dependências.

**Nota:** Para a Questão 9 (CNNs), recomenda-se usar GPU. No Colab: Runtime > Change runtime type > GPU.

---

## 📊 Metodologia

### Pré-processamento
- Tratamento de valores ausentes e outliers
- Normalização e padronização de dados
- Feature engineering (criação de novas variáveis)
- Encoding de variáveis categóricas
- Data augmentation (para imagens)

### Modelagem
- Seleção de algoritmos apropriados para cada problema
- Split de dados (treino/validação/teste)
- Treinamento com validação cruzada
- Ajuste de hiperparâmetros
- Comparação de múltiplos modelos

### Avaliação
- Métricas específicas para cada tipo de problema:
  - **Classificação:** Acurácia, Precisão, Recall, F1-Score, ROC-AUC
  - **Regressão:** RMSE, MAE, R²
  - **Recomendação:** RMSE, MAE, Suporte, Confiança, Lift
- Análise de matrizes de confusão
- Validação e interpretação dos resultados

---

## 📈 Resumo dos Resultados

### Questões 1-4 (Resolvidas em Sala)
✅ Fundamentos de classificação e clusterização  
✅ Implementação de modelos básicos  
✅ Análise exploratória de dados  
✅ Visualização de resultados

### Questões 5-10 (Trabalho Extra)
✅ Modelos avançados de classificação e regressão  
✅ Sistemas de recomendação (Apriori e Filtragem Colaborativa)  
✅ Deep Learning com CNNs para imagens médicas  
✅ Feature engineering e otimização de modelos  
✅ Análise comparativa de diferentes abordagens

---

## 🔍 Datasets Utilizados

| Questão | Dataset | Tipo | Fonte |
|---------|---------|------|-------|
| 1 | Iris | Classificação | Scikit-Learn |
| 2 | Credit Card Fraud | Classificação | Kaggle |
| 3 | E-commerce Customers | Clusterização | Kaggle/Gerado |
| 4 | Patient Health Data | Clusterização | Kaggle/Gerado |
| 5 | Heart Failure Prediction Dataset | Classificação | Kaggle |
| 6 | King County House Data | Regressão | Scikit-Learn |
| 7 | Market Basket | Associação | Kaggle/Gerado |
| 8 | MovieLens | Recomendação | GroupLens |
| 9 | Chest X-ray Images | Classificação | Kaggle |
| 10 | Rossmann Store Sales | Regressão | Kaggle |

---

## 📝 Observações

- ✅ O uso de IA Generativa foi permitido para auxiliar na implementação dos códigos
- ✅ Todos os notebooks contêm comentários explicativos e análises detalhadas
- ✅ Os modelos foram avaliados com múltiplas métricas para garantir robustez
- ✅ Questões 1-4 foram resolvidas em sala de aula
- ✅ Questões 5-10 foram desenvolvidas como trabalho extra valendo 10 pontos

---

## 🎓 Competências Desenvolvidas

Este projeto proporcionou experiência prática em:

### Fundamentos (Questões 1-4)
- ✅ Classificação binária e multiclasse
- ✅ Algoritmos de clusterização (K-Means, DBSCAN)
- ✅ Análise exploratória de dados
- ✅ Visualização de resultados

### Avançado (Questões 5-10)
- ✅ Modelos de classificação médica
- ✅ Regressão para previsão de valores contínuos
- ✅ Sistemas de recomendação (regras de associação e filtragem colaborativa)
- ✅ Deep Learning com CNNs para visão computacional
- ✅ Feature engineering e otimização de modelos
- ✅ Avaliação crítica de diferentes abordagens
- ✅ Trabalho com dados reais de diferentes domínios

---

## 📄 Licença

Este projeto é de uso acadêmico para a disciplina de IA, ML e IA Generativa da Unifacisa.

---

**Unifacisa - 2025** 🚀  
