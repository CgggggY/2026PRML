# Week4: Transformer Reproduction and Ablation

This folder contains a small PyTorch reproduction of **Attention Is All You Need** and two ablation experiments for the PRML Week4 assignment.

The original paper uses large WMT machine translation datasets. To make the experiment runnable on a single GPU, this code uses synthetic sequence transduction tasks. The default task is `reverse`: given a random token sequence, the model predicts the reversed sequence.

## Files

```text
Week4/
├── README.md
├── requirements.txt
├── train.py
├── run_experiments.py
├── plot_results.py
└── transformer_lab/
    ├── __init__.py
    ├── data.py
    └── models.py
```

## Experiments

The code supports three models:

- `transformer`: a small encoder-decoder Transformer with sinusoidal positional encoding, multi-head attention, residual connections, LayerNorm, and FFN.
- `qk_only`: ablation for Q/K/V necessity. It keeps Q and K, but merges K and V by setting `V = K`.
- `cnn`: ablation for positional encoding + CNN. It uses the same token embedding and sinusoidal positional encoding, but replaces attention blocks with 1D convolution blocks.

## Environment

On the course server, the following environment was used:

```bash
/home/buaachenguanyu/anaconda3/envs/minimind_v/bin/python
```

Minimal dependencies:

```bash
pip install -r requirements.txt
```

## Quick Smoke Test

```bash
python run_experiments.py \
  --models transformer qk_only cnn \
  --steps 5 \
  --eval-every 5 \
  --eval-batches 2 \
  --batch-size 16 \
  --device cuda:0
```

## Full Experiment

```bash
python run_experiments.py \
  --models transformer qk_only cnn \
  --task reverse \
  --seq-len 8 \
  --steps 1000 \
  --eval-every 200 \
  --eval-batches 20 \
  --batch-size 128 \
  --dropout 0.0 \
  --lr 0.001 \
  --device cuda:0 \
  --output-dir runs_final
```

Run a deeper CNN baseline:

```bash
python train.py \
  --model cnn \
  --task reverse \
  --seq-len 8 \
  --layers 6 \
  --steps 1000 \
  --eval-every 200 \
  --eval-batches 20 \
  --batch-size 128 \
  --dropout 0.0 \
  --lr 0.001 \
  --device cuda:0 \
  --output-dir runs_final
```

## Plot Results

After training, draw result figures with:

```bash
python plot_results.py --runs-dir runs_final --out-dir figures
```

## Data

The training data is generated dynamically in `transformer_lab/data.py`. No external dataset is required.

