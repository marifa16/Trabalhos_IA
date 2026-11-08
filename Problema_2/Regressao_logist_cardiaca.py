import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

PASTA = Path(__file__).resolve().parent #Pasta do arquivo
ARQUIVO = Path(sys.argv[1])if len(sys.argv) > 1 else PASTA / "processo_dados.csv" #Arquivo de dados

COLUNAS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

VARIAVEIS_X = COLUNAS[:-1] #Todas colunas menos a última
VARIAVEL_Y = COLUNAS[-1] #Última coluna
SEED = 42 

#Título no terminal

def titulo(texto):
    print("\n" + "=" * 70)
    print(texto)
    print("=" * 70)
    
#Carregando dados

df = pd.read_csv(ARQUIVO, header=None, names=COLUNAS, na_values="?")

print(f"Arquivo: {ARQUIVO}")
print(f"Formato: {df.shape[0]}")
print("\nPrimeiras linhas")
print(df.head())
print("\nResumo estatístico")
print(df.describe().round(2))

#Verificando valores

print("\nValores ausentes")

ausentes = df.isna().sum()
print(ausentes[ausentes > 0] if ausentes.sum() > 0 else "Não tem valores ausentes")
print(f"\nTotal de valores ausentes: {ausentes.sum()}")
print("-> Serão preenchidos com o valor mais frequente (moda) calculado no treino.")
print("   (ca e thal são variáveis de categorias, então a moda é mais adequada que a média.)")

#Binarização da variável alvo

print("\nBinarizando variável alvo")

print("Distribuição original da variável alvo:")
print(df[VARIAVEL_Y].value_counts().sort_index())

y = df[VARIAVEL_Y].astype(int) #verifica se é inteiro
x = df[VARIAVEIS_X]

print("\nDistribuição binarizada da variável alvo:")
contagem = y.value_counts().sort_index()
for classe, quantidade in contagem.items():
    print(f"Classe {classe}:{quantidade} pacientes({100*quantidade/len(y):1.1f}%)")
    
#Divisão em treino e teste

