# Installation

scmidas requires **Python ≥3.10** and is tested on Python 3.10 / 3.11 / 3.12 against PyTorch 2.5–2.10 and Lightning 2.4–2.6. We recommend creating a fresh environment to avoid conflicts with existing setups.

## Step 1: Create a new environment

```bash
conda create -n scmidas python=3.12
conda activate scmidas
```

## Step 2: Install scmidas

```bash
pip install scmidas
```

This pulls in the full set of dependencies (PyTorch, Lightning, scanpy, mudata, …). On Volta / Pascal GPUs (e.g. V100, P100, GTX 10xx), `import scmidas` emits a `UserWarning` with downgrade instructions if your CUDA install is incompatible with the resolved PyTorch wheel — newer torch (≥2.11) dropped those compute capabilities from its default `cu128/cu129` wheels.

**You're all set.** Continue with the [Tutorials](./tutorials/tutorial_index.html) — the [Quickstart](https://github.com/labomics/midas/blob/main/examples/quickstart.ipynb) gets you from `pip install` to a UMAP in about a minute.
