# Twenty-epoch loss comparison

This experiment keeps the original SFCNN system, codebook, training data, validation data, initialization seed, and batch order fixed while changing only the loss function.

Run from the repository root:

```bash
python loss_comparison/22_run_loss_comparison.py
```

The default comparison trains these four losses for 20 epochs:

- `residual_mse`: original notebook objective;
- `channel_nmse`: reconstructed-channel NMSE;
- `hybrid`: channel NMSE plus 0.10 times residual MSE;
- `channel_nmse_cvar`: channel NMSE plus 0.10 times the worst-20% sample loss.

All four models use one shared generated training dataset and validation dataset. Each model is initialized with the same seed and receives the same shuffled batch order. Evaluation resets NumPy to the same test seed before every model, so channel, pilot, and noise draws are paired across losses.

Outputs are written to `loss_comparison_results/`:

- one checkpoint per loss;
- `loss_ranking.csv` and `loss_comparison_report.json`;
- raw NMSE arrays in `loss_comparison_arrays.npz`;
- training-history and NMSE comparison plots.

The default evaluation is computationally expensive because it retains 50 test channel realizations for every combination of four path counts and seven SNR values. Use the smoke test before starting the full run:

```bash
python loss_comparison/22_run_loss_comparison.py --quick
```

To compare only selected losses:

```bash
python loss_comparison/22_run_loss_comparison.py --losses residual_mse channel_nmse hybrid
```

To set the number of deformation views, use `--m-views`. For example, eight
views produce `4*M = 32` model-input channels:

```bash
python loss_comparison/22_run_loss_comparison.py --m-views 8
```

For mixed-range deformation training, draw one morphing amplitude per
independent channel and apply it to all M views of that channel:

```bash
python loss_comparison/22_run_loss_comparison.py \
  --m-views 8 \
  --morphing-min 0.01 \
  --morphing-max 0.5
```

This samples `b/lambda` uniformly over the requested interval during training.
The standard evaluation remains fixed at `b/lambda = 0.02` so results remain
comparable with the original experiment.

## Evaluation-only morphing robustness sweep

Compare the existing fixed-range and mixed-range M=8 checkpoints without any
additional training:

```bash
python loss_comparison/24_evaluate_morphing_robustness.py \
  --fixed-checkpoint /path/to/fixed/sfcnn_channel_nmse_cvar_50epoch.pt \
  --mixed-checkpoint /path/to/mixed/sfcnn_channel_nmse_cvar_50epoch.pt \
  --morphing-ratios 0.01 0.02 0.10 0.30 0.50 \
  --snr-db 0 10 20 \
  --path-counts 3 \
  --eval-channels 10 \
  --m-views 8
```

The evaluator batches all 31 adjacent subcarrier pairs and feeds identical
generated observations to both models. It saves CSV, JSON, NPZ, and NMSE plots.

The morph-invariance loss is not included in this comparison yet. It requires paired observations of the same propagation scene at different morphing amplitudes plus the saved normalization scale for every pair.

## Higher-diversity dataset

Keep the physical grid at `K=32`, but train with 4,000 independent channels and eight randomly selected adjacent pairs per channel:

```bash
python loss_comparison/22_run_loss_comparison.py \
  --epochs 50 \
  --losses residual_mse channel_nmse_cvar \
  --train-channels 4000 \
  --val-channels 800 \
  --pairs-per-channel 8 \
  --evaluation-seed 2026 \
  --output loss_comparison_results_diverse
```

This produces 32,000 stored training examples, close to the original 31,000, but they come from four times as many independent propagation scenes. Enabling `--pairs-per-channel` also uses an observation-only normalization and stratifies the original `[0, 5, 10, 15, 20]` dB training SNRs across channel realizations.

## Controlled dataset ablation

Run the four combinations of channel diversity and normalization with the SNR
sampler, seeds, initialization, evaluation data, and losses held fixed:

```bash
python loss_comparison/23_run_dataset_ablation.py \
  --epochs 50 \
  --losses residual_mse channel_nmse_cvar \
  --output dataset_ablation_results
```

The four cells are 1,000 channels x 31 pairs and 4,000 channels x 8 pairs,
each tested with both target-assisted and observation-only normalization.
Target-assisted normalization is included only to reproduce and diagnose the
original experiment; observation-only normalization is the deployment-valid
choice. The top-level results folder contains a CSV/JSON summary and an L=3
comparison plot, while each cell subfolder contains its checkpoints and full
evaluation outputs.
