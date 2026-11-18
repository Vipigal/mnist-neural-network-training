"""Roda os experimentos da seção 4 do TP2 usando as funções de src/learn.py.

Reproduz o laço do server.py: a cada época, um gradient_descent_step no conjunto
de treino e a acurácia no conjunto de teste. Para cada treino salva, em
experiments/results/<nome>/:
  - W1.npy, b1.npy, W2.npy, b2.npy  (pesos finais, como o "Salvar Pesos")
  - losses.csv, accuracies.csv      (formato da interface: 1 ponto a cada 100 épocas)
  - losses_full.csv, accuracies_full.csv (1 ponto por época)
  - losses.png, accuracies.png      (mesmo estilo do plot.py, por época)
e acumula uma linha em experiments/results/summary.csv.

Uso: python experiments/run_experiments.py   (a partir da raiz do projeto)
"""
import csv
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import learn  # noqa: E402

RESULTS = os.path.join(ROOT, "experiments", "results")
SEED = 42
BASE_HIDDEN = 64
BASE_EPOCHS = 500

X_train, X_test, y_train, y_test = learn.load_mnist()
cache = {}


def plot(values, color, path, xlabel, ylabel):
    plt.figure()
    plt.plot(values, color=color, linestyle="solid")
    plt.grid(True, linestyle="--")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(path)
    plt.close()


def train(lr, hidden, epochs):
    key = (lr, hidden, epochs)
    if key in cache:
        print(f"   (reaproveitando lr={lr} h={hidden} e={epochs})", flush=True)
        return cache[key]

    name = f"lr{lr}_h{hidden}_e{epochs}"
    out = os.path.join(RESULTS, name)
    os.makedirs(out, exist_ok=True)

    np.random.seed(SEED)
    parameters = learn.initialize_parameters(784, hidden, 10)
    losses, accuracies = [], []          # formato da interface (server.py)
    losses_full, accuracies_full = [], []
    epoch = 0
    start = time.perf_counter()
    while epoch < epochs:
        parameters, loss = learn.gradient_descent_step(X_train, y_train, parameters, lr)
        acc = learn.accuracy(y_test, learn.forward(X_test, parameters)[-1])
        epoch += 1
        losses_full.append(loss)
        accuracies_full.append(acc)
        if epoch % 100 == 0:
            losses.append(loss)
            accuracies.append(acc)
        if epoch % 50 == 0 or epoch == epochs:
            print(f"   {name} época {epoch}: loss {loss:.4f} acc {acc:.4f} "
                  f"({time.perf_counter() - start:.0f}s)", flush=True)
    elapsed = time.perf_counter() - start

    for n, p in zip(["W1", "b1", "W2", "b2"], parameters):
        np.save(os.path.join(out, f"{n}.npy"), p)
    np.savetxt(os.path.join(out, "losses.csv"), losses, delimiter=",")
    np.savetxt(os.path.join(out, "accuracies.csv"), accuracies, delimiter=",")
    np.savetxt(os.path.join(out, "losses_full.csv"), losses_full, delimiter=",")
    np.savetxt(os.path.join(out, "accuracies_full.csv"), accuracies_full, delimiter=",")
    plot(losses_full, "red", os.path.join(out, "losses.png"), "Número de Épocas", "Perda")
    plot(accuracies_full, "blue", os.path.join(out, "accuracies.png"), "Número de Épocas", "Acurácia")

    result = {
        "name": name, "learning_rate": lr, "hidden_size": hidden, "epochs": epochs,
        "final_loss": float(losses_full[-1]), "final_accuracy": float(accuracies_full[-1]),
        "best_accuracy": float(max(accuracies_full)), "seconds": round(elapsed, 1),
        "nan": bool(np.isnan(losses_full).any()),
        "losses_full": losses_full, "accuracies_full": accuracies_full,
    }
    cache[key] = result
    return result


def compare(results, label_key, label_fmt, path_prefix, title):
    for metric, ylabel in [("losses_full", "Perda"), ("accuracies_full", "Acurácia")]:
        plt.figure()
        for r in results:
            plt.plot(r[metric], label=label_fmt.format(r[label_key]))
        plt.grid(True, linestyle="--")
        plt.xlabel("Número de Épocas")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        suffix = "losses" if metric == "losses_full" else "accuracies"
        plt.savefig(os.path.join(RESULTS, f"{path_prefix}_{suffix}.png"))
        plt.close()


def experiment(exp_name, configs, label_key, label_fmt, title):
    print(f"== {exp_name} ==", flush=True)
    results = [dict(train(*c), experiment=exp_name) for c in configs]
    compare(results, label_key, label_fmt, exp_name, title)
    with open(os.path.join(RESULTS, "summary.csv"), "a", newline="") as f:
        w = csv.writer(f)
        for r in results:
            w.writerow([exp_name, r["name"], r["learning_rate"], r["hidden_size"], r["epochs"],
                        f"{r['final_loss']:.6f}", f"{r['final_accuracy']:.6f}",
                        f"{r['best_accuracy']:.6f}", r["seconds"], r["nan"]])
    return results


def best(results):
    valid = [r for r in results if not r["nan"]]
    return max(valid, key=lambda r: r["final_accuracy"])


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "summary.csv"), "w", newline="") as f:
        csv.writer(f).writerow(["experimento", "treino", "learning_rate", "hidden_size", "epochs",
                                "loss_final_treino", "acuracia_final_teste",
                                "melhor_acuracia_teste", "segundos", "teve_nan"])

    r_lr = experiment("1_taxa_aprendizado",
                      [(lr, BASE_HIDDEN, BASE_EPOCHS) for lr in [0.001, 0.01, 0.1, 0.5]],
                      "learning_rate", "lr = {}", f"{BASE_HIDDEN} neurônios, {BASE_EPOCHS} épocas")
    best_lr = best(r_lr)["learning_rate"]
    print(f"-> melhor taxa: {best_lr}", flush=True)

    r_h = experiment("2_neuronios",
                     [(best_lr, h, BASE_EPOCHS) for h in [4, 16, 64, 256]],
                     "hidden_size", "{} neurônios", f"lr = {best_lr}, {BASE_EPOCHS} épocas")
    best_h = best(r_h)["hidden_size"]
    print(f"-> melhor nº de neurônios: {best_h}", flush=True)

    r_e = experiment("3_epocas",
                     [(best_lr, BASE_HIDDEN, e) for e in [50, 100, 200, 500]],
                     "epochs", "{} épocas", f"lr = {best_lr}, {BASE_HIDDEN} neurônios")

    final = train(best_lr, best_h, BASE_EPOCHS)
    with open(os.path.join(RESULTS, "melhor_configuracao.json"), "w") as f:
        json.dump({k: v for k, v in final.items() if not k.endswith("_full")}, f, indent=2)
    print(f"== melhor configuração: {final['name']} -> acc {final['final_accuracy']:.4f} ==", flush=True)
