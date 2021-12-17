# -*- coding: utf-8 -*-

#Biblioteca para carga dos dados
import pandas as pd
#Definção dos títulos das colunas
headers = ['ESCT', 'NDEP', 'RENDA', 'TIPOR', 'VBEM', 'NPARC',
           'VPARC', 'TEL', 'IDADE', 'RESMS', 'ENTRADA', 'CLASSE']

#Carga do conjunto de treino

arquivo = 'https://raw.githubusercontent.com/MLRG-CEFET-RJ/ml-class/master/ml-t3/datasets/credtrain.txt'
data_train = pd.read_csv(arquivo, sep='\t', header=None, names=headers)

#Carga do conjunto de teste

arquivo = 'https://raw.githubusercontent.com/MLRG-CEFET-RJ/ml-class/master/ml-t3/datasets/credtest.txt'
data_test = pd.read_csv(arquivo, sep='\t', header=None, names=headers)
data_test.head()

#Biblioteca para transformação dos dados em matrizes
import numpy as np
#Transoformação dos atributos e da classe alvo em matrizes
#data_train.loc[:,'ESCT'] #Tira o label
X_train_ = np.array(data_train.iloc[:, 0:11]) #inclui o label
y_train_ = np.array(data_train['CLASSE'])

#Transformação dos atributos e da classe alvo em matrizes
X_test = np.array(data_test.iloc[:, 0:11])
y_test = np.array(data_test['CLASSE'])

#Função para fatiamento dos conjuntos de dados
from sklearn.model_selection import train_test_split
#Separação de treino e validação
X_train, X_val, y_train, y_val = train_test_split(X_train_, y_train_, #Conjuntos de dados
                                                  train_size=0.8,     #Tamanho da fatia de treinamento
                                                  random_state=31)

#Checagem rápida de parte dos dados carregados
print(data_train.head(),
      data_test.head(),
      X_train[0],
      y_train[0],
      sep='\n\n')

data_train.describe()


#2 Criando e treinando um modelo

#Importando o algoritmo que será usado como base
from sklearn.tree import DecisionTreeClassifier
#Criação do modelo
modelo = DecisionTreeClassifier(max_depth=3,
                                random_state=31)
#Conferência do moddelo
modelo
#Treinamento do modelo
modelo.fit(X_train,
           y_train)
DecisionTreeClassifier(max_depth=3, random_state=31)

#Visualização gráfica da árvore de decisão
import os
from graphviz import Source
from sklearn.tree import export_graphviz

export_graphviz(modelo,
                out_file='credit_tree.dot', #Arquivo para armazenamento do modelo gráfico
                feature_names=headers[0:11], #Nomes dos atributos
                rounded=True,
                filled=True
                )
#Uso do arquivo gerado para visualizar a árvore
Source.from_file('credit_tree.dot')


###### 3 Aplicando o modelo treinado #######
#Revendo o conjunto de dados de validação
X_val[:5]

#Predição no conjunto de validação
y_val_pred = modelo.predict(X_val)
#Visualização das primeiras 10 observações preditas
y_val_pred[:10]
y_val[:10]


###### 4 Avaliando o desempenho do modelo ######
from sklearn.metrics import confusion_matrix
#Matriz de confusão
confusion_matrix(y_val, y_val_pred)
#Organização dos dados para visualização da matriz
cm = confusion_matrix(y_val, y_val_pred)
tn, fp, fn, tp = cm.ravel()

#Reorganização da matriz confusão para 
cm_reorganizada = np.array([[tp, fn], [fp, tn]])
cm_reorganizada

from matplotlib import pyplot as plt
import seaborn as sns

#Visualização Gráfica da Matriz de Confusão
modelo_title = 'Decision Tree - Validação'
fig = plt.figure(figsize=(10,8))
fig.suptitle('Matriz de Confusão', fontsize=14, fontweight='bold')

sns.heatmap(cm_reorganizada, cmap='Blues', annot=True)

plt.title(modelo_title, fontsize=14)

plt.xticks([])
plt.yticks([])

plt.annotate('TP', (0.3,0.5), fontweight='bold')
plt.annotate('FN', (1.3,0.5), fontweight='bold')
plt.annotate('FP', (0.3,1.5), fontweight='bold')
plt.annotate('TN', (1.3,1.5), fontweight='bold')


from sklearn.metrics import classification_report
#Visualizaçaõ do relatório de classificação
print(classification_report(y_val, y_val_pred))

####### 5 Usando o modelo com dados novos #######
#Predição no conjunto de testes
y_pred = modelo.predict(X_test)
y_pred[:5]
#Organização dos dados para visualização da matriz
cm = confusion_matrix(y_test, y_pred)
#Reorganização da matriz confusão para 
tn, fp, fn, tp = cm.ravel()
cm_reorganizada = np.array([[tp, fn], [fp, tn]])
cm_reorganizada

#Dimensão dos conjuntos de dados
print(X_train.shape,
      X_val.shape,
      X_test.shape,
      sep='\n')

#Visualização Gráfica da Matriz de Confusão
modelo_title = 'Decision Tree - Teste (FINAL)'
fig = plt.figure(figsize=(10,8))
fig.suptitle('Matriz de Confusão', fontsize=14, fontweight='bold')

sns.heatmap(cm_reorganizada, cmap='Blues', annot=True)

plt.title(modelo_title, fontsize=14)

plt.xticks([])
plt.yticks([])

plt.annotate('TP', (0.3,0.5), fontweight='bold')
plt.annotate('FN', (1.3,0.5), fontweight='bold')
plt.annotate('FP', (0.3,1.5), fontweight='bold')
plt.annotate('TN', (1.3,1.5), fontweight='bold')

#Visualização do relatório de classificação
print(classification_report(y_test, y_pred))




