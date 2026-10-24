import sys 
from pathlib import Path 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import make_pipeline 


PASTA = Path(__file__).resolve().parent / "usina.csv"
Arquivo = Path(sys.argv[1])if len(sys.argv) > 1 else PASTA / "Folds5x2_pp.xlsx"

VARIAVEIS_X = ["AT", "V", "AP", "RH"] #variáveis independentes
VARIAVEIS_Y = "PE" #variável dependente 
SEMENTE = 42 #o valor do sorteio treino seja sempre o mesmo, para os resultados serem reproduzíveis

def titulo(texto):
    print("\n" + "=" * 80) 
    print(texto)
    print("=" * 80 + "\n")


titulo("Carregando os dados...")

if Arquivo.suffix.lower() in (".xls", ".xlsx"): #lê arquivos do Excel, que podem ter várias planilhas
    df = pd.read_excel(Arquivo, sheet_name=0) # lê a primeira planilha
else:
    df = pd.read_csv(Arquivo) # lê arquivos CSV


print(f"Arquivo carregado: {Arquivo}")
print(f"Formato do arquivo: {df.shape}")
print("\nPrimeiras linhas do arquivo:")
print(df.head())
print("\nResumo estatístico das variáveis:")
print(df.describe().round(2))

#Checado se há volores ausentes

ausentes = df.isnull().sum()
print(ausentes)
print(f"\nNúmero de valores ausentes: {ausentes.sum()}")
if ausentes.sum() == 0:
    print("Não há valores ausentes!")


#Correlação entre as variáveis 

correlacao = df[VARIAVEIS_X + [VARIAVEIS_Y]].corr()[VARIAVEIS_Y].drop(VARIAVEIS_Y) #correlação entre as variáveis independentes e a variável dependente
correlacao = correlacao.sort_values(key=abs, ascending=False) #ordena pela correlação absoluta
print(correlacao.round(4))

mais_influente = correlacao.index[0] #variável mais influente na variável dependente
print(f"\n -> A variável com maior correlação com a variável dependente ({VARIAVEIS_Y})" f"(r = {correlacao.iloc[0]:.3f})")


#Correlação entre as variáveis independentes 

correlacao_ind = df[VARIAVEIS_X].corr() #correlação entre as variáveis independentes
print(correlacao_ind.round(3))


print("\nPares com correlação maior que 0.7:") 
achou = False
for i,a in enumerate(VARIAVEIS_X): #compara cada variável com as demais, sem repetir pares
    for b in VARIAVEIS_X[i+1:]:
        if abs(correlacao_ind.loc[a,b]) > 0.7:
            print(f" -> {a} e {b}: {correlacao_ind.loc[a,b]:.3f}")
            achou = True
            if not achou:
                print("Nenhum par de variáveis independentes")

print("\nVIF para cada variável independente:")
for var in VARIAVEIS_X:
    outras_var = [v for v in VARIAVEIS_X if v != var] #variáveis independentes, exceto a variável atual
    modelo = LinearRegression()
    r2 = modelo.score(df[outras_var], df[var]) #R² da regressão da variável atual em função das demais
    vif = 1 / (1 - r2)
    print(f" -> {var}: VIF = {vif:.2f}")
    
#Mapa de calor da correlação completo
correlacao_total = df[VARIAVEIS_X + [VARIAVEIS_Y]].corr()
fig, ax = plt.subplots(figsize=(6, 5))
im = 