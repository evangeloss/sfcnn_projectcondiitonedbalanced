# Morph-conditioned M=8 SFCNN

This is a separate experiment project. It leaves the successful original
SFCNN project unchanged and adds explicit morphing-ratio conditioning.

## Why this architecture

The existing network receives 32 observation channels for eight deformation
views but is not told the commanded `b/lambda`. Fixed-range training is highly
accurate around 0.02 and fails at large deformation; uniform mixed training is
robust at 0.3-0.5 but loses precision around 0.01-0.02. One unconditional CNN
is therefore averaging incompatible correction regimes.

`MorphConditionedSFCNN` encodes `log10((b/lambda)/0.02)` with a small MLP and
uses FiLM scale-and-shift modulation in all five convolution blocks. This lets
the same convolutional features behave differently at different commanded
morphing amplitudes.

## Efficient initialization

The five convolutions, five batch-normalization layers, and output layer have
the same dimensions as the original M=8 SFCNN. They are copied exactly from
the existing mixed-morphing checkpoint. All FiLM heads start with zero weights
and biases, making the initial transformation an identity:

```text
conditioned output at initialization = existing checkpoint output
```

The training script verifies this numerically before generating the dataset.
Only the conditioning path is trainable for the first three epochs; the full
network is then unfrozen and fine-tuned at `1e-4` for 12 additional epochs.

## Anchored training distribution

For every independent propagation channel:

- 50% use exactly `b/lambda = 0.02`;
- 50% draw uniformly from `[0.01, 0.5]`.

This protects the original high-accuracy operating point while retaining broad
range coverage. The dataset uses M=8, 4,000 training channels, 800 validation
channels, eight adjacent pairs per channel, observation-only normalization,
and channel-NMSE plus CVaR.

## Kaggle

1. Create a GitHub repository containing this project and push it.
2. Add `sfcnn_m8_mixed_morphing_results.zip` as a Kaggle notebook input.
3. Enable Internet and a GPU.
4. In `kaggle_clone_and_run_conditioned.py`, set `REPO_URL` to the new GitHub
   repository URL.
5. Paste the launcher into a Kaggle cell and run it.

The downloadable result is:

```text
/kaggle/working/morph_conditioned_results.zip
```

It contains the best validation checkpoint, training plot, JSON report, and an
evaluation-only sweep at `b/lambda = 0.01, 0.02, 0.10, 0.30, 0.50` for L=3 and
SNR 0, 10, and 20 dB.

## Direct command

```bash
python morph_conditioned/train_conditioned.py \
  --checkpoint /path/to/sfcnn_channel_nmse_cvar_50epoch.pt \
  --epochs 15 \
  --conditioning-only-epochs 3 \
  --train-channels 4000 \
  --val-channels 800 \
  --pairs-per-channel 8 \
  --learning-rate 0.0001 \
  --morphing-min 0.01 \
  --morphing-max 0.5 \
  --anchor-ratio 0.02 \
  --anchor-probability 0.5 \
  --output morph_conditioned_results
```

Use `--quick` for a two-epoch, small-dataset smoke test before the complete run.

## Recommended range-balanced training

The second-generation trainer is:

```text
morph_conditioned/train_conditioned_balanced.py
```

It differs from the initial conditioned run in five important ways:

- it trains on `-10, -5, 0, 5, 10, 15, 20` dB;
- only 20% of channels are fixed at the `0.02` anchor;
- the remaining channels are balanced across six morphing intervals;
- the transferred CNN uses a smaller learning rate than the new FiLM path;
- checkpoints are selected using a fixed ratio/SNR validation grid and a
  penalty for the worst morphing interval.

On Kaggle, use the complete launcher:

```text
kaggle_clone_and_run_balanced_conditioned.py
```

Add the original mixed-morphing M=8 results ZIP as a Kaggle input, enable a
GPU and Internet, and run the launcher in one cell. It trains for 40 epochs,
runs the final 50-channel L=3 sweep, and creates:

```text
/kaggle/working/morph_conditioned_balanced_results.zip
```

The balanced trainer is the recommended next experiment. The paired-channel
physics-consistency objective described in the research notes is not included
in this run because it requires generating two deformations of every physical
scene and substantially increases dataset-generation time.
