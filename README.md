### Transformers develop separable posterior and value representations in a simple decision task

Code, trained weights and results for the paper *Transformers develop separable
posterior and value representations in a simple decision task*.

The repository reproduces every number, table and figure in the paper.

## Installation

```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
```

Dependencies are pinned in `pyproject.toml`. Training needs a CUDA GPU; the
toy-model analyses run on CPU or Apple MPS in minutes.

## Repository structure

```
jbo/           library (task, model, trainer, probes, steering, plotting)
experiments/   one script per analysis, 00_train ... 15_single_chain
tools/         pinning, results collection, and three figure scripts
configs/       architecture and training specs
worlds/, hmms/ task and hidden-Markov-chain specs
studies/       one folder per trained model: seeds, checkpoint, fitted instruments, pinned runs
results/       one folder per run: results.json, manifest.json, figures
figures/       figures drawn by tools/
```

## Usage

Every experiment takes a study and writes a new run directory under
`results/<study>/<experiment>/`:

```
python experiments/<script>.py --study <world>/<tag>
```

`--smoke` runs a small test version. Results are never overwritten unless
`--force` is given.

### Reproducing the main model

Trained weights are included, so the first two steps are optional.

```
S=mess3_m1m2/s0
python experiments/00_train.py --study $S --snapshots          # ~2 h on one GPU
python tools/pin_checkpoint.py --study $S --weights results/$S/00_train/<run>/checkpoint/weights.pt

python experiments/02_fit_instruments.py --study $S            # fit probes and steering planes
python experiments/01_behavior.py --study $S                   # choice vs. target policy
python experiments/01_behavior.py --study $S --snapshot 0      # the same at initialization
python experiments/03_decoder_views.py --study $S              # linear reads, withheld offers
python experiments/04_formation.py --study $S                  # reads across training
python experiments/05_dissociation.py --study $S               # card, mask and clue twins
python experiments/06_steering.py --study $S                   # posterior and value steering, mediation
python experiments/07_intrinsic_dim.py --study $S              # principal component attribution
python experiments/09_value_reads.py --study $S                # offer probability, payoff mass, joint posterior
python tools/pin_runs.py --study $S
python experiments/08_decoder_representation.py --study $S     # composes Figure 2 a-d
python experiments/10_joint_ablation.py --study $S             # joint posterior ablation
python experiments/11_pca_steering.py --study $S               # steering through principal components
python experiments/12_nonlinear_probe.py --study $S            # MLP probes
python experiments/13_probe_robustness.py --study $S           # probe construction
python experiments/14_heldout_cards.py --study $S              # withheld payoff cards
python tools/pin_runs.py --study $S
python tools/collect_results.py                                # optional: lists every reported number in RESULTS.md
```

`tools/pin_runs.py` records the newest run of each experiment in
`studies/<study>/pins.json`; `tools/collect_results.py` reads those runs.


## Included data

- `mess3_m1m2/s0`: weights, training snapshots, and every run with its data and figures.
- Other toy models: weights, and `results.json` and `manifest.json` for every run.
- Qwen3-1.7B-Base: results only (the checkpoint is not included), with
  the data its figures are drawn from. Re-running it needs an 80 GB GPU: about
  8 hours of training and 13 hours of analysis.

## Notation

| Code | Paper | Meaning |
|---|---|---|
| `etaA`, `etaB` | η_M, η_N | factor posteriors |
| `eta` | η | joint posterior over the nine states |
| `P` | P(X) | offer probability |
| `v` | m(X) | payoff mass |
| `D` | Q(X) | offer value |
| `beta` | β | inverse temperature of the target policy |
