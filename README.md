# Optimization for Learning — Benchmarks

> **📄 [Read the full report (PDF)](report/main.pdf)**

An empirical study of randomized optimization, Adam optimizer ablations, and regularization strategies applied to neural network training on two classification tasks. Built on top of the [Supervised Learning Benchmarks](https://github.com/max-montes/supervised-learning-benchmarks) baselines.

## Datasets

| Dataset | Samples | Features | Task | Metric |
|---------|---------|----------|------|--------|
| **Adult Income** | 45,222 | 14 (→104 after encoding) | Binary classification | F1 Score |
| **Wine Quality** | 6,497 | 13 | 8-class classification | Macro-F1 |

## Experiments

### Part 1 — Randomized Optimization

Applied RHC, Simulated Annealing, and Genetic Algorithms to fine-tune the last two layers of pre-trained MLPs.

| Algorithm | Adult F1 | Wine Macro-F1 |
|-----------|----------|---------------|
| SL Baseline | 0.6726 | 0.3085 |
| RHC | 0.6677 | 0.3085 |
| SA | 0.6677 | **0.3395** |
| GA | **0.6758** | 0.3085 |

### Part 2 — Adam Ablations (Adult)

Compared 7 optimizers across 5 seeds with identical architecture.

| Optimizer | Test F1 | Grad Evals |
|-----------|---------|------------|
| SGD | 0.6707 | 12,100 |
| SGD+Momentum | **0.6783** | 9,680 |
| Nesterov | 0.6762 | 9,680 |
| Adam | 0.6744 | 2,565 |
| Adam (no bias correction) | 0.6677 | 2,565 |
| Adam (β₁=0) | 0.6703 | 2,565 |
| AdamW | 0.6745 | 2,565 |

### Part 3 — Regularization Study (Adult)

Systematically varied L2 weight decay, dropout, early stopping, and noise regularization under fixed Adam.

| Configuration | Test F1 |
|---------------|---------|
| Baseline (dropout=0.2) | 0.6677 |
| L2 (λ=10⁻³) | 0.6776 |
| Noise (LS=0.1, IN=0.05) | 0.6788 |
| **L2 + Dropout (0.4)** | **0.6814** |
| All four combined | 0.6692 |

### Part 4 — Integrated Best Combination

Composed Adam + L2 + Dropout with GA fine-tuning. GA fine-tuning did not improve over the regularized model alone, confirming that Adam + L2(10⁻³) + Dropout(0.4) is the optimal recipe (F1 = 0.6825).

## Key Findings

- **Randomized optimization can escape local minima** that gradient descent gets stuck in — SA improved Wine Macro-F1 by +10% over baseline.
- **Adam converges ~4× faster** than SGD+Momentum (2,565 vs. 9,680 gradient evaluations) for comparable final performance.
- **Regularization matters more than optimizer choice** — L2 + Dropout (F1 = 0.6814) outperforms the best optimizer ablation (SGD+Momentum, F1 = 0.6783).
- **More regularization isn't always better** — combining all four techniques degraded performance, over-constraining the network.
- **RO fine-tuning adds no value** when the model is already well-regularized.

## Repository Structure

```
├── data/                # Adult and Wine datasets (CSV)
├── notebooks/
│   └── ol_report.ipynb  # All experiments, figures, and analysis
├── report/
│   ├── main.pdf         # Compiled report
│   ├── main.tex         # LaTeX source
│   ├── references.bib
│   └── figures/         # Generated plots
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook ol_report.ipynb
```

## References

- Kingma, D. P. & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR*.
- Loshchilov, I. & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR*.
- Srivastava, N. et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *JMLR*, 15(56), 1929–1958.
- Kirkpatrick, S., Gelatt, C. D. & Vecchi, M. P. (1983). Optimization by Simulated Annealing. *Science*, 220(4598), 671–680.
- Mitchell, M. (1998). *An Introduction to Genetic Algorithms*. MIT Press.
- Müller, R., Kornblith, S. & Hinton, G. E. (2019). When Does Label Smoothing Help? *NeurIPS*, 32.
- Becker, B. & Kohavi, R. (1996). Adult Income Dataset. UCI Machine Learning Repository.
- Cortez, P. et al. (2009). Wine Quality Dataset. UCI Machine Learning Repository.
