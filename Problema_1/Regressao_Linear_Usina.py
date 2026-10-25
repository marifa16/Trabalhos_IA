from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.parent

VARIAVEIS_X = ["AT", "V", "AP", "RH"]
VARIAVEL_Y = "PE"
SEMENTE = 42


def titulo(texto: str) -> None:
    print("\n" + "=" * 80)
    print(texto)
    print("=" * 80 + "\n")


def localizar_arquivo_entrada() -> Path:
    if len(sys.argv) > 1:
        caminho = Path(sys.argv[1]).expanduser()
        if not caminho.is_absolute():
            caminho = (Path.cwd() / caminho).resolve()
        return caminho

    candidatos = [
        BASE_DIR / "Folds5x2_pp.xlsx",
        BASE_DIR / "usina.csv",
        WORKSPACE_DIR / "Folds5x2_pp.xlsx",
        WORKSPACE_DIR / "usina.csv",
    ]
    for candidato in candidatos:
        if candidato.exists():
            return candidato

    caminhos = "\n - ".join(str(c) for c in candidatos)
    raise FileNotFoundError(
        "Nenhum dataset encontrado automaticamente.\n"
        "Informe o arquivo como argumento, por exemplo:\n"
        '  ".venv/bin/python" Problema_1/Regressao_Linear_Usina.py Problema_1/Folds5x2_pp.xlsx\n'
        f"Locais verificados:\n - {caminhos}"
    )


def carregar_dados(arquivo: Path) -> pd.DataFrame:
    if not arquivo.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {arquivo}")

    if arquivo.suffix.lower() in (".xls", ".xlsx"):
        return pd.read_excel(arquivo, sheet_name=0)
    if arquivo.suffix.lower() == ".csv":
        return pd.read_csv(arquivo)
    raise ValueError("Formato inválido. Use .csv, .xls ou .xlsx")


def calcular_vif(df_x: pd.DataFrame) -> dict[str, float]:
    resultado = {}
    for var in df_x.columns:
        outras = [c for c in df_x.columns if c != var]
        modelo_aux = LinearRegression()
        modelo_aux.fit(df_x[outras], df_x[var])
        r2_aux = modelo_aux.score(df_x[outras], df_x[var])
        if r2_aux >= 0.999999:
            resultado[var] = float("inf")
        else:
            resultado[var] = 1.0 / (1.0 - r2_aux)
    return resultado


def metricas(y_true: pd.Series, y_pred: np.ndarray) -> tuple[float, float, float]:
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae


def main() -> None:
    titulo("Carregando os dados")
    arquivo = localizar_arquivo_entrada()
    df = carregar_dados(arquivo)

    colunas_necessarias = set(VARIAVEIS_X + [VARIAVEL_Y])
    faltantes = colunas_necessarias.difference(df.columns)
    if faltantes:
        raise ValueError(
            "Colunas obrigatórias ausentes no dataset: "
            + ", ".join(sorted(faltantes))
        )

    print(f"Arquivo carregado: {arquivo}")
    print(f"Formato do arquivo: {df.shape}")
    print("\nPrimeiras linhas:")
    print(df.head())
    print("\nResumo estatístico:")
    print(df[VARIAVEIS_X + [VARIAVEL_Y]].describe().round(2))

    titulo("Pré-processamento (ausentes)")
    ausentes = df[VARIAVEIS_X + [VARIAVEL_Y]].isnull().sum()
    print(ausentes)
    print(f"\nTotal de valores ausentes: {int(ausentes.sum())}")
    if int(ausentes.sum()) == 0:
        print("Não há valores ausentes.")

    titulo("1) Correlação entre variáveis de entrada e PE")
    corr_y = (
        df[VARIAVEIS_X + [VARIAVEL_Y]]
        .corr()[VARIAVEL_Y]
        .drop(VARIAVEL_Y)
        .sort_values(key=np.abs, ascending=False)
    )
    print(corr_y.round(4))
    mais_influente = corr_y.index[0]
    print(
        f"\nMaior influência linear: {mais_influente} "
        f"(r = {corr_y.iloc[0]:.3f})"
    )

    titulo("2) Correlação entre variáveis independentes")
    corr_x = df[VARIAVEIS_X].corr()
    print(corr_x.round(3))

    print("\nPares com |correlação| > 0.70:")
    achou = False
    for i, a in enumerate(VARIAVEIS_X):
        for b in VARIAVEIS_X[i + 1 :]:
            valor = corr_x.loc[a, b]
            if abs(valor) > 0.70:
                print(f" - {a} e {b}: {valor:.3f}")
                achou = True
    if not achou:
        print(" - Nenhum par com correlação forte.")

    print("\nVIF por variável:")
    vif = calcular_vif(df[VARIAVEIS_X])
    for var in VARIAVEIS_X:
        valor = vif[var]
        texto = "infinito" if np.isinf(valor) else f"{valor:.2f}"
        print(f" - {var}: VIF = {texto}")

    corr_total = df[VARIAVEIS_X + [VARIAVEL_Y]].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr_total, cmap="coolwarm", vmin=-1, vmax=1)
    nomes = corr_total.columns
    ax.set_xticks(range(len(nomes)))
    ax.set_yticks(range(len(nomes)))
    ax.set_xticklabels(nomes)
    ax.set_yticklabels(nomes)
    for i in range(len(nomes)):
        for j in range(len(nomes)):
            ax.text(
                j,
                i,
                f"{corr_total.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )
    fig.colorbar(im, ax=ax)
    ax.set_title("Mapa de calor da correlação")
    fig.tight_layout()
    arquivo_corr = BASE_DIR / "correlacao_total.png"
    fig.savefig(arquivo_corr, dpi=300)
    plt.close(fig)
    print(f"\nMapa de calor salvo em: {arquivo_corr}")

    titulo("3) Regressão linear múltipla (80/20)")
    x = df[VARIAVEIS_X]
    y = df[VARIAVEL_Y]
    x_treino, x_teste, y_treino, y_teste = train_test_split(
        x, y, test_size=0.2, random_state=SEMENTE
    )
    print(f"Treino: {x_treino.shape[0]} amostras | Teste: {x_teste.shape[0]} amostras")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    modelo = LinearRegression()

    x_treino_imp = imputer.fit_transform(x_treino)
    x_teste_imp = imputer.transform(x_teste)
    x_treino_std = scaler.fit_transform(x_treino_imp)
    x_teste_std = scaler.transform(x_teste_imp)

    print("\nEstatísticas de padronização (treino):")
    for nome, media, desvio in zip(VARIAVEIS_X, scaler.mean_, scaler.scale_):
        print(f" - {nome}: média={media:.2f} | desvio={desvio:.2f}")

    modelo.fit(x_treino_std, y_treino)

    print("\nCoeficientes padronizados:")
    print(f" intercepto: {modelo.intercept_:.4f}")
    for nome, coef in zip(VARIAVEIS_X, modelo.coef_):
        print(f" - {nome}: {coef:+.4f} MW por 1 desvio padrão")

    coef_orig = modelo.coef_ / scaler.scale_
    intercepto_orig = modelo.intercept_ - np.sum(modelo.coef_ * scaler.mean_ / scaler.scale_)
    print("\nCoeficientes na unidade original:")
    print(f" intercepto: {intercepto_orig:.4f}")
    for nome, coef in zip(VARIAVEIS_X, coef_orig):
        print(f" - {nome}: {coef:+.4f} MW por unidade")

    eq = f"PE = {intercepto_orig:.2f}"
    for nome, coef in zip(VARIAVEIS_X, coef_orig):
        sinal = "+" if coef >= 0 else "-"
        eq += f" {sinal} {abs(coef):.4f}*{nome}"
    print(f"\nEquação estimada: {eq}")

    titulo("4) Avaliação do modelo")
    pred_treino = modelo.predict(x_treino_std)
    pred_teste = modelo.predict(x_teste_std)
    r2_treino, rmse_treino, mae_treino = metricas(y_treino, pred_treino)
    r2_teste, rmse_teste, mae_teste = metricas(y_teste, pred_teste)

    print(f"{'Conjunto':<10}{'R²':>10}{'RMSE':>14}{'MAE':>12}")
    print(f"{'Treino':<10}{r2_treino:>10.4f}{rmse_treino:>14.4f}{mae_treino:>12.4f}")
    print(f"{'Teste':<10}{r2_teste:>10.4f}{rmse_teste:>14.4f}{mae_teste:>12.4f}")

    amplitude = y.max() - y.min()
    media_y = y.mean()
    print(
        f"\nPE varia de {y.min():.1f} a {y.max():.1f} MW "
        f"(amplitude = {amplitude:.1f} MW; média = {media_y:.2f} MW)"
    )
    print(f"RMSE no teste = {100 * rmse_teste / amplitude:.2f}% da amplitude")
    print(f"MAE no teste = {100 * mae_teste / amplitude:.2f}% da amplitude")
    print(f"MAE no teste = {100 * mae_teste / media_y:.2f}% da média de PE")

    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression())
    r2_cv = cross_val_score(pipe, x, y, cv=5, scoring="r2")
    print(f"\nValidação cruzada (5-fold) - R²: {r2_cv.mean():.4f} +/- {r2_cv.std():.4f}")

    diferenca_r2 = r2_treino - r2_teste
    status = "sem sinal forte de overfitting" if abs(diferenca_r2) < 0.02 else "possível overfitting"
    print(f"Diferença de R² treino-teste: {diferenca_r2:+.4f} -> {status}")

    residuos_teste = y_teste - pred_teste
    fig, eixo = plt.subplots(1, 3, figsize=(16, 4.5))

    eixo[0].scatter(y_teste, pred_teste, s=8, alpha=0.4)
    limite = [y.min(), y.max()]
    eixo[0].plot(limite, limite, "r--", label="previsão perfeita")
    eixo[0].set_xlabel("PE real (MW)")
    eixo[0].set_ylabel("PE previsto (MW)")
    eixo[0].set_title(f"Previsão vs Real (Teste) - R²={r2_teste:.3f}")
    eixo[0].legend()

    eixo[1].scatter(pred_teste, residuos_teste, s=8, alpha=0.4)
    eixo[1].axhline(0, color="r", linestyle="--")
    eixo[1].set_xlabel("PE previsto (MW)")
    eixo[1].set_ylabel("Resíduo (MW)")
    eixo[1].set_title("Resíduo vs Previsão (Teste)")

    eixo[2].hist(residuos_teste, bins=40, edgecolor="black")
    eixo[2].set_xlabel("Resíduo (MW)")
    eixo[2].set_ylabel("Frequência")
    eixo[2].set_title("Distribuição dos resíduos (Teste)")

    fig.tight_layout()
    arquivo_diag = BASE_DIR / "fig2_diagnostico.png"
    fig.savefig(arquivo_diag, dpi=150)
    plt.close(fig)
    print(f"Gráficos de diagnóstico salvos em: {arquivo_diag}")


if __name__ == "__main__":
    main()