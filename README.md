# Delt-IEt Battery Lab — MVP 1.0

Aplicação local para que pesquisadores e engenheiros carreguem dados de baterias,
calculem a entropia de cada curva de descarga e executem o motor Delt-IEt.

## O que esta versão faz

- Recebe arquivos NASA `.mat`, CSV e Excel.
- Para `.mat`, tenta identificar a estrutura NASA e extrai ciclos `discharge`,
  `Voltage_measured` e `Capacity`.
- Para CSV/Excel, aceita:
  - **Formato longo:** uma linha por amostra, com colunas de ciclo e tensão.
  - **Formato largo:** uma linha por ciclo e várias colunas com amostras de tensão.
- Calcula a entropia de Shannon da distribuição de amplitudes de tensão por ciclo.
- Aplica o motor legado do notebook B0005.
- Mostra índice, limiar, eventos, capacidade (se fornecida) e exporta CSV/relatório.

## Instalação — Windows

1. Instale Python 3.10 ou superior.
2. Extraia o ZIP.
3. Abra a pasta `Delt-IEt_Battery_Lab`.
4. Na barra de endereço do Explorador, digite `cmd` e pressione Enter.
5. Execute:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

O navegador abrirá a aplicação. Se não abrir automaticamente, copie o endereço
mostrado no terminal (normalmente `http://localhost:8501`).

## Instalação — Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## Formato de dados recomendado (CSV longo)

```csv
cycle,voltage,capacity_Ah
1,4.18,2.05
1,4.16,2.05
1,4.12,2.05
2,4.17,2.03
2,4.14,2.03
2,4.10,2.03
```

A coluna `capacity_Ah` é opcional. Cada ciclo precisa ter pelo menos três
amostras de tensão. Se houver coluna de tempo/amostra, o aplicativo permite
ordenar as amostras dentro de cada ciclo.

## Fidelidade ao notebook original

O núcleo `deltiet_engine.py` mantém deliberadamente as operações da função
`deltiet()` fornecida no notebook B0005:

1. `dS = np.diff(S, prepend=S[0])`
2. `d2S = np.diff(dS, prepend=dS[0])`
3. média e desvio-padrão móveis com janela `h`, `min_periods=1`, `ddof=0`
4. `Z = (S - mu) / sigma`, com sigma mínimo `1e-12`
5. `DeltIEt = Z * abs(d2S)`
6. limiar `Θ = percentile(abs(DeltIEt), q)` sobre a série inteira
7. evento quando `abs(DeltIEt) > Θ`

A entropia é calculada em cada curva completa de tensão de descarga, como no
notebook original. Não foi substituída por entropia em janela deslizante dentro
da curva.

## Limitação metodológica que fica explícita

O notebook legado calcula o percentil sobre **toda a série**, inclusive ciclos
posteriores ao evento. Isso é retrospectivo e não equivale a um limiar causal
disponível em tempo real. O texto metodológico mais amplo também menciona
período de aquecimento e limiar histórico adaptativo. Esta primeira versão não
finge implementar essa variante: preserva o cálculo efetivamente presente no
notebook para evitar alterar os resultados publicados.

Antes de oferecer a ferramenta como monitoramento online, será necessário
definir e validar separadamente a regra causal/histórica, sem misturá-la com o
modo de reprodução legado.

## Estrutura

- `app.py` — interface Streamlit.
- `deltiet_engine.py` — núcleo matemático isolado.
- `tests/test_engine.py` — testes de consistência.
- `requirements.txt` — dependências.
- `examples/exemplo_formato_longo.csv` — dados sintéticos ilustrativos (não são dados NASA).

## Aviso

Ferramenta experimental de pesquisa. Eventos são saídas matemáticas do método,
não diagnóstico de falha, identificação de mecanismo eletroquímico ou garantia
de vida útil. Validação independente e avaliação de segurança são necessárias
antes de qualquer uso operacional.
