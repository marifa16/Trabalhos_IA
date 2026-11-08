import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PASTA_SCRIPT = Path(__file__).resolve().parent
PASTA_RAIZ = PASTA_SCRIPT.parent
ARQUIVO_PADRAO = PASTA_RAIZ / "heart+disease" / "processed.cleveland.data"
ARQUIVO_DADOS = Path(sys.argv[1]) if len(sys.argv) > 1 else ARQUIVO_PADRAO

COLUNAS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

SEED = 42


def titulo(texto: str) -> None:
    print("\n" + "=" * 72)
    print(texto)
    print("=" * 72)


def main() -> None:
    if not ARQUIVO_DADOS.exists():
        raise FileNotFoundError(
            "Arquivo de dados nao encontrado. "
            f"Passe o caminho no terminal ou coloque em: {ARQUIVO_PADRAO}"
        )

    titulo("1) Carregamento dos dados")
    df = pd.read_csv(ARQUIVO_DADOS, header=None, names=COLUNAS, na_values="?")
    for coluna in COLUNAS:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    print(f"Arquivo utilizado: {ARQUIVO_DADOS}")
    print(f"Formato do dataset: {df.shape[0]} linhas x {df.shape[1]} colunas")

    titulo("2) Tratamento de ausentes e binarizacao da variavel alvo")
    ausentes = df.isna().sum()
    print("Valores ausentes por coluna:")
    print(ausentes[ausentes > 0] if ausentes.sum() > 0 else "Nenhum valor ausente")

    y_original = df["num"].astype(int)
    y = (y_original > 0).astype(int)
    x = df.drop(columns=["num"])

    print("\nDistribuicao original do alvo (0 a 4):")
    print(y_original.value_counts().sort_index())
    print("\nDistribuicao binaria (0 = sem doenca, 1 = com doenca):")
    print(y.value_counts().sort_index())

    titulo("3) Divisao treino/teste")
    x_treino, x_teste, y_treino, y_teste = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )
    print(f"Treino: {x_treino.shape[0]} amostras")
    print(f"Teste: {x_teste.shape[0]} amostras")

    titulo("4) Pre-processamento (imputacao + padronizacao)")
    imputador = SimpleImputer(strategy="most_frequent")
    escalador = StandardScaler()

    x_treino_imp = imputador.fit_transform(x_treino)
    x_teste_imp = imputador.transform(x_teste)

    x_treino_pad = escalador.fit_transform(x_treino_imp)
    x_teste_pad = escalador.transform(x_teste_imp)

    print("Media das variaveis padronizadas no treino (aprox. 0):")
    print(np.round(x_treino_pad.mean(axis=0), 3))
    print("Desvio padrao das variaveis padronizadas no treino (aprox. 1):")
    print(np.round(x_treino_pad.std(axis=0), 3))

    titulo("5) Treinamento do modelo")
    modelo = LogisticRegression(solver="liblinear", random_state=42, max_iter=1000)
    modelo.fit(x_treino_pad, y_treino)

    n_iteracoes = int(np.max(modelo.n_iter_))
    print("Parametros: solver='liblinear', random_state=42, max_iter=1000")
    print(f"Iteracoes para convergencia: {n_iteracoes}")
    print(
        "Status de convergencia: "
        f"{'convergiu antes do limite' if n_iteracoes < 1000 else 'atingiu o limite de iteracoes'}"
    )

    titulo("6) Avaliacao no conjunto de teste")
    y_pred = modelo.predict(x_teste_pad)
    y_prob = modelo.predict_proba(x_teste_pad)[:, 1]

    acuracia = accuracy_score(y_teste, y_pred)
    precisao = precision_score(y_teste, y_pred)
    recall = recall_score(y_teste, y_pred)
    f1 = f1_score(y_teste, y_pred)
    auc = roc_auc_score(y_teste, y_prob)

    print(f"Acuracia:  {acuracia:.4f}")
    print(f"Precisao:  {precisao:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")

    titulo("7) Curva ROC e Matriz de Confusao")
    fpr, tpr, _ = roc_curve(y_teste, y_prob)

    plt.style.use("seaborn-v0_8")
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    ax_roc.plot(fpr, tpr, linewidth=2, label=f"Regressao Logistica (AUC = {auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", label="Aleatorio (AUC = 0.5)")
    ax_roc.set_xlabel("Taxa de Falsos Positivos")
    ax_roc.set_ylabel("Taxa de Verdadeiros Positivos")
    ax_roc.set_title("Curva ROC")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(alpha=0.3)
    fig_roc.tight_layout()
    fig_roc.savefig(PASTA_SCRIPT / "curva_roc.png", dpi=150)

    fig_cm, ax_cm = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_teste,
        y_pred,
        display_labels=["Sem doenca", "Com doenca"],
        cmap="Blues",
        ax=ax_cm,
    )
    ax_cm.set_title("Matriz de Confusao")
    fig_cm.tight_layout()
    fig_cm.savefig(PASTA_SCRIPT / "matriz_confusao.png", dpi=150)

    print("Graficos salvos em:")
    print(f"- {PASTA_SCRIPT / 'curva_roc.png'}")
    print(f"- {PASTA_SCRIPT / 'matriz_confusao.png'}")

    if "agg" in plt.get_backend().lower():
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
