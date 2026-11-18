# Rede Neural do Zero para Classificação de Dígitos (MNIST)

Trabalho Prático 2 de Introdução à Inteligência Artificial (DCC/UFMG). Uma rede neural de duas camadas (784 → H → 10), implementada do zero em NumPy e treinada por gradiente descendente para classificar os dígitos manuscritos do MNIST, com uma interface web para treinar a rede e testar dígitos desenhados à mão.

**Resultado principal:** 95,44% de acurácia no conjunto de teste, com taxa de aprendizado 0,5, 256 neurônios na camada escondida e 500 épocas.

📄 **Relatório:** [`reports/relatorio.pdf`](reports/relatorio.pdf) (fonte LaTeX em [`reports/relatorio.tex`](reports/relatorio.tex))

## Destaques

- **Tudo em NumPy:** inicialização, ReLU, softmax numericamente estável, entropia cruzada vetorizada e passo de gradiente estão em [`src/learn.py`](src/learn.py).
- **Três experimentos:** efeito da taxa de aprendizado, do número de neurônios e do número de épocas. Os dados de cada treino ficam em [`experiments/results/`](experiments/results/), e o resumo está em [`summary.csv`](experiments/results/summary.csv).
- **Limite prático:** a rede acerta 95% no teste, mas erra muito com dígitos desenhados na interface. Deslocar as imagens de teste em 2 pixels derruba a acurácia para 71%, porque uma rede totalmente conectada não tem invariância a translação. A análise está na seção "Testes na Interface" do relatório.

| Taxa de aprendizado (H = 64) | Neurônios (α = 0,5) | Épocas (α = 0,5, H = 64) |
|---|---|---|
| 0,001 → 14,91% | 4 → 78,83% | 50 → 78,84% |
| 0,01 → 45,19% | 16 → 93,72% | 100 → 90,24% |
| 0,1 → 90,21% | 64 → 94,96% | 200 → 92,43% |
| **0,5 → 94,96%** | **256 → 95,44%** | **500 → 94,96%** |

## Estrutura

```
.
├── src/                        # código da rede e da interface
│   ├── learn.py                # a rede neural (carga dos dados, forward, perda, backward, gradiente, acurácia)
│   ├── server.py               # servidor Flask + Socket.IO que treina a rede e conversa com a interface
│   ├── plot.py                 # gera as curvas de perda e acurácia a partir de src/model/
│   ├── templates/index.html    # página da interface
│   ├── static/                 # JavaScript da interface (canvas de desenho, gráficos, comunicação)
│   ├── model/                  # pesos salvos pela interface (atualmente, os da melhor rede, H = 256)
│   └── requirements.txt        # dependências Python
├── experiments/
│   ├── run_experiments.py      # roda os três experimentos fora da interface
│   ├── make_figures.py         # gera as figuras do relatório
│   ├── run.log                 # log da execução dos experimentos
│   └── results/                # pesos, curvas (CSV/PNG) e resumo de cada treino
├── reports/
│   ├── relatorio.tex           # relatório (template AAAI 2027)
│   ├── referencias.bib         # referências
│   └── figures/                # figuras usadas no relatório
├── uso-de-ia/                  # transcrição da conversa com o Claude Code
└── docs/tp2-rna.pdf            # enunciado do trabalho
```

## Como rodar

Requer **Python 3.10+**. O projeto foi testado com Python 3.14 e NumPy 2.3.5.

### 1. Instalar as dependências

```bash
git clone https://github.com/Vipigal/mnist-neural-network-training.git
cd mnist-neural-network-training

python3 -m venv tp2-env
source tp2-env/bin/activate          # no Windows: tp2-env\Scripts\activate
pip install -r src/requirements.txt
```

Na primeira execução, o MNIST (cerca de 50 MB) é baixado do OpenML e fica em cache em `~/scikit_learn_data`.

### 2. Subir a interface

O servidor precisa ser iniciado **de dentro de `src/`**, porque ele lê e grava os pesos no caminho relativo `model/`.

```bash
cd src
python server.py
```

Abra **http://localhost:5001** no navegador.

### 3. Usar a interface

- **Treinar:** defina as épocas, a taxa de aprendizado e o número de neurônios, marque *Inicializar pesos* e clique em **Iniciar**. A época, a perda de treino e a acurácia de teste são atualizadas a cada passo.
- **Salvar Pesos:** grava os pesos atuais e as curvas em `src/model/`. Durante o treino, as curvas também são salvas a cada 100 épocas.
- **Carregar Pesos:** carrega os pesos de `src/model/`. O repositório já vem com os pesos da melhor rede, então dá para testar sem treinar.
- **Testar:** desenhe um dígito no canvas e veja a previsão. Use **Limpar Dígito** para recomeçar. Dica: desenhe o dígito centralizado e ocupando boa parte da grade, como no MNIST.

Com os pesos salvos, `python plot.py` (também de dentro de `src/`) gera `losses.png` e `accuracies.png`.

### 4. Reproduzir os experimentos e as figuras

Da raiz do projeto:

```bash
python experiments/run_experiments.py   # cerca de 16 min num Ryzen 5 3600
python experiments/make_figures.py      # gera reports/figures/*.pdf
```

O primeiro script sobrescreve `experiments/results/`. A semente é fixa (42), então os números se repetem na mesma máquina e na mesma versão do NumPy.

### 5. Compilar o relatório

Com uma distribuição LaTeX que tenha `pdflatex` e `bibtex` (TeX Live, MiKTeX ou TinyTeX), ou no Overleaf:

```bash
cd reports
pdflatex relatorio && bibtex relatorio && pdflatex relatorio && pdflatex relatorio
```

O estilo AAAI exige **pdfTeX**; XeLaTeX e Tectonic não funcionam.

## Uso de IA

Este trabalho foi feito com apoio do Claude Code (modelo Claude Opus 5.5). O que foi feito com IA está descrito na seção "Declaração de Uso de IA" do relatório, e a transcrição completa está em [`uso-de-ia/conversa-claude-code.md`](uso-de-ia/conversa-claude-code.md).

## Referência

LeCun, Y.; Bottou, L.; Bengio, Y.; Haffner, P. *Gradient-based learning applied to document recognition*. Proceedings of the IEEE, 86(11): 2278–2324, 1998.
