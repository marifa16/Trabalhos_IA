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
im = ax.imshow(correlacao_total, cmap="coolwarm", vmin=-1, vmax=1) 
nomes = correlacao_total.columns
ax.set_xticks(range(len(nomes)))
ax.set_yticks(range(len(nomes)))
ax.set_xticklabels(nomes)
ax.set_yticklabels(nomes)

#Adiciona os valores da correlação no mapa de calor

for i in range(len(nomes)):
    for j in range(len(nomes)):
        ax.text(j, i, f"{correlacao_total.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)

fig.colorbar(im,ax=ax)
ax.set_title("Mapa de calor da correlação")
fig.tight_layout()
fig.savefig(PASTA / "correlacao_total.png", dpi=300)

#Divisão dos dados em treino/teste (80%/20%)

X = df[VARIAVEIS_X]
Y = df[VARIAVEIS_Y]
       
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.2, random_state=SEMENTE)

print(f"\nTreino: {X_treino.shape[0]} amostras, Teste: {X_teste.shape[0]} amostras")

#Pré-pocessamento: valores ausentes e padronização das variáveis

imputer = SimpleImputer(strategy="median") #substitui valores ausentes pela média da variável
X_treino = imputer.fit_transform(X_treino) #aprende a média e substitui os valores ausentes no conjunto
X_teste = imputer.transform(X_teste) #aplica a média aprendida 

scaler = StandardScaler() #padroniza as variáveis para média 0 e desvio padrão 1
X_treino = scaler.fit_transform(X_treino) #aplica a padronização no conjunto
X_teste = scaler.transform(X_teste) #aplica a padronização aprendida

#Média das variáveis no treino antes da padronização

for nome, m, s in zip(VARIAVEIS_X, scaler.mean_,scaler.scale_):
    print(f" -> {nome}: média = {m:.2f} | desvio padrão -> {s:.2f}")
print("\nMédia das variáveis após a padronização -> 0 | desvio padrão -> 1")
print(f" -> média:", np.round(X_treino.mean(axis=0), 4))
print(f" -> desvio padrão: ", np.round(X_treino.std(axis=0), 4))

#Treinamento do modelo de regressão linear

modelo = LinearRegression()
modelo.fit(X_treino, Y_treino) #treina o modelo com os dados

#Coeficientes váriaveis padronizadas que comparam entre si

print(f" intercepto: {modelo.intercept_:.4f}")
for nome, coef in zip(VARIAVEIS_X, modelo.coef_):
    print(f" -> {nome}:{coef:+.4f}(MW por desvio padrão da variável)")
    
#Convertendo as unidaddes

coeficiente_originais = modelo.coef_ / scaler.scale_ 
intercepto_original = modelo.intercept_ - np.sum(modelo.coef_ * scaler.mean_ / scaler.scale_)

print(f"\nCoeficientes Originai: ")
print(f" intercepto: {intercepto_original:.4f}")
for nome, coef in zip(VARIAVEIS_X, coeficiente_originais):
    print(f" -> {nome}:{coef:+.4f} (MW por unidade da variável)")
    
eq = f"PE = {intercepto_original:.2f}"
for nome, coef in zip(VARIAVEIS_X, coeficiente_originais):
    eq += f" {'+' if coef >= 0 else '-'}{abs(coef):.4f}*{nome}"
print(f"\nEquação da regressão linear: {eq}")

#Avaliar o desempenho do modelo

def metricas(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae 

pred_treino = modelo.predict(X_treino)
pred_teste = modelo.predict(X_teste)

r2_treino, nrmse_treino, mae_treino = metricas(Y_treino, pred_treino)
r2_teste, nrmse_teste, mae_teste = metricas(Y_teste, pred_teste)

print(f"{'Conjunto':<10}{'R²':>10}{'NRMSE':>14}{'MAE':>12}")
print(f"{'Treino':<10}{r2_treino:>10.4f}{nrmse_treino:>14.4f}{mae_treino:>14.3f}")
print(f"{'Teste':<10}{r2_teste:>10.4f}{nrmse_teste:>14.4f}{mae_teste:>14.3f}")

amplitude = Y.max() - Y.min()
print(f"\nPE varia de {Y.min():.1f} a {Y.max():.1f}MW" f"(amplitude = {amplitude:.1f}MW; desvio padrão = {Y.std():.2f}MW)")
print(f" NRMSE no teste = {100*nrmse_teste/amplitude:.2f}% da amplitude")
print(f" MAE no teste = {100*mae_teste/amplitude:.2f}% da amplitude")
print(f" MAE no teste = {100*mae_teste/amplitude:.2f}% da media de PE")

from sklearn.pipeline import make_pipeline

pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression())
r2_cv = cross_val_score(pipe, X, Y, cv=5, scoring="r2")
print(f"\nValidação cruzada - R²: " f"{r2_cv.mean():.4f}+-{r2_cv.std():.4f}")

#Diagnostico de sobreajuste 

print(f"\nDiferença de R²: {r2_treino - r2_teste:+4f}" f" -> {'sem sinal de overfitting' if abs(r2_treino - r2_teste)<0.02 else 'possível overfitting'}")
