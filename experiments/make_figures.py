"""Gera as figuras do relatório (reports/figures/) a partir de experiments/results/.

Uso: python experiments/make_figures.py   (a partir da raiz do projeto, depois de run_experiments.py)
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import learn  # noqa: E402

RES = os.path.join(ROOT, "experiments", "results")
FIG = os.path.join(ROOT, "reports", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
W = 3.3  # largura de uma coluna AAAI, em polegadas


def full(name, metric):
    return np.loadtxt(os.path.join(RES, name, f"{metric}_full.csv"), delimiter=",")


# Figura 1: perda por taxa de aprendizado
plt.figure(figsize=(W, 2.0))
for lr in [0.001, 0.01, 0.1, 0.5]:
    plt.plot(np.arange(1, 501), full(f"lr{lr}_h64_e500", "losses"), label=f"$\\alpha$ = {lr}")
plt.xlabel("Época"); plt.ylabel("Perda (treino)")
plt.grid(True, linestyle="--", alpha=.6); plt.legend()
plt.savefig(os.path.join(FIG, "fig_taxa.pdf")); plt.close()

# Figura 2: acurácia por número de neurônios
plt.figure(figsize=(W, 2.0))
for h in [4, 16, 64, 256]:
    plt.plot(np.arange(1, 501), full(f"lr0.5_h{h}_e500", "accuracies"), label=f"{h} neurônios")
plt.xlabel("Época"); plt.ylabel("Acurácia (teste)")
plt.grid(True, linestyle="--", alpha=.6); plt.legend(loc="lower right")
plt.savefig(os.path.join(FIG, "fig_neuronios.pdf")); plt.close()

# Figura 3: curvas da melhor configuração
fig, ax = plt.subplots(1, 2, figsize=(W, 1.6))
ax[0].plot(np.arange(1, 501), full("lr0.5_h256_e500", "losses"), color="red")
ax[0].set_xlabel("Época"); ax[0].set_ylabel("Perda (treino)")
ax[1].plot(np.arange(1, 501), full("lr0.5_h256_e500", "accuracies"), color="blue")
ax[1].set_xlabel("Época"); ax[1].set_ylabel("Acurácia (teste)")
for a in ax:
    a.grid(True, linestyle="--", alpha=.6)
fig.tight_layout()
plt.savefig(os.path.join(FIG, "fig_melhor.pdf")); plt.close()

# Figura 4: o "1" desenhado na interface, original e deslocado, e a média dos "1" do MNIST
P = tuple(np.load(os.path.join(RES, "lr0.5_h256_e500", f"{n}.npy")) for n in ["W1", "b1", "W2", "b2"])
draw = np.load(os.path.join(RES, "desenho_1_canvas.npy"))
_, X_test, _, y_test = learn.load_mnist()
mean1 = X_test[y_test.argmax(1) == 1].mean(0).reshape(28, 28)


def label(img):
    a2 = learn.forward(img.reshape(1, 784), P)[-1][0]
    return f"previsto: {a2.argmax()} ({a2.max():.0%})"


panels = [(draw, "(a) desenho", label(draw)),
          (np.roll(draw, 2, axis=1), "(b) +2 px à direita", label(np.roll(draw, 2, axis=1))),
          (mean1 / mean1.max(), "(c) média dos \"1\"", "MNIST (teste)")]
fig, ax = plt.subplots(1, 3, figsize=(W, 1.35))
for a, (img, title, sub) in zip(ax, panels):
    a.imshow(img, cmap="gray", vmin=0, vmax=1)
    a.axvline(13.5, color="tab:orange", lw=.6, ls="--")
    a.set_title(title); a.set_xlabel(sub)
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
plt.savefig(os.path.join(FIG, "fig_deslocamento.pdf")); plt.close()

print("figuras em", FIG, ":", sorted(os.listdir(FIG)))
