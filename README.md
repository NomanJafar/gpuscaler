# gpuscaler

Queue-depth-driven autoscaler for GPU inference replicas (KEDA-style) with cold-start mitigation, validated against replayed traffic traces.

Project 11 of 15 in [inference-infra-portfolio](https://github.com/NomanJafar/inference-infra-portfolio).

## What this builds

KEDA-style scaling of inference replicas on pending-request queue depth, with cold-start mitigation.

## Why it matters

GPUs idling at 30% utilisation burn money. Autoscaling is FinOps.

## Status

Concept primer written (`CONCEPTS.md`). The controller and the trace replay are next; this project needs no GPU.

## Layout

```
CONCEPTS.md     the theory behind the project, written before the code
src/gpuscaler/   implementation
tests/          CPU-only tests, runnable on a laptop
notebooks/      GPU notebooks (Kaggle / Colab) that run the experiments and write results/
results/        measured numbers and plots
```

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

GPU experiments run from `notebooks/`; each notebook clones this repo, installs it and writes its outputs to `results/`.

## Hardware notes

Developed against free-tier GPUs (Kaggle and Colab T4, 16 GB). Where a T4 limits what can be measured (no FP8, Turing-era kernels), the README says so rather than papering over it.

## License

MIT
