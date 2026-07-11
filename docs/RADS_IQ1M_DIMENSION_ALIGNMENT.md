# RADs / I/Q-1M Dimension Alignment

## Scope

The transfer experiment uses one physical elevation channel and a RADs-like
Doppler span. Its canonical model input is:

```text
[Doppler, Azimuth, Elevation, Range, ComplexPhase]
[64,      256,     1,         256,   2]
```

RADs' 256 azimuth samples are simulator-generated and must not be interpreted
as 256 independent physical antenna channels. Matching `A=256` proves that GRT
can consume the same tensor geometry; it does not by itself prove equal angular
resolution or equal sensor statistics.

## A8 Transfer Baseline

`scripts/rads_map_checkpoint_infer.py` can reduce RADs `A=256` to I/Q-1M's
native `A=8`. The supported reducers are:

- `gaussian_coherent`
- `sector_coherent`
- `sector_energy_circular`
- `sector_max_energy`

On three fixed RADs frames, sector-coherent reduction preserved the RA energy
layout best, while sector-max-energy gave the best sparse-GT F1/recall at
`logit > 0`. Absolute F1 is not directly comparable to I/Q-1M LiDAR occupancy
because `RADs_gt` is sparse radar occupancy.

## A256 Model Path

The precomputed cache remains `[64,8,1,256,2]`. Creating a float16 A256 cache
for all 621,013 samples would require about 10.4 TB. Instead,
`AzimuthFFTTransformerEncoder` performs the following online:

1. Reconstruct complex values from `sqrt(abs)` and phase.
2. Apply inverse shift and inverse FFT along azimuth.
3. Zero pad the eight-element aperture to 256.
4. Apply FFT and shift along azimuth.
5. Re-encode `sqrt(abs)` and phase.

This matches a direct raw-IQ A256 FFT with relative complex L2 error
`4.05e-7` and phase MAE `5.76e-7` radians on the checked sample.

The A256 patch is `[8,32,1,8]`, yielding an `8 x 8 x 1 x 32 = 2048` token
grid. This keeps global-attention cost equal to the A8 baseline while exposing
eight angular tokens instead of one. Keeping the old `[2,8,1,4]` patch would
produce 65,536 tokens and is not practical for global attention.

## Official Initialization

The official single-elevation patch `[2,8,1,4]` is reordered into local
`D,A,E,R,C` feature order and trilinearly resized to `[8,32,1,8]`. A scale
exponent of `0.92` was calibrated on all 16 fixed validation samples because
zero-padded FFT bins are strongly correlated and violate independent fan-in
assumptions. Patch activation standard deviation after calibration was `0.843`
for A8 versus `0.823` for A256. All 124 requested encoder/decoder tensors load.

## Cropped RADs Inference

Range cropping shifts the entire complex 3D RAD cube before RA, RD, AD, model
input, or GT projections are derived. A256 checkpoints retain the resulting
azimuth axis directly. `RADs_gt` receives the same range shift and azimuth flip,
then max pooling to the model's `128 x 64` polar output grid.

```bash
python scripts/rads_map_checkpoint_infer.py \
  --checkpoint <best.ckpt> \
  --hparams <run>/hparams.yaml \
  --rads-root /root/autodl-tmp/data/RADs \
  --gt-root /root/autodl-tmp/data/RADs_gt \
  --modes crop_auto \
  --input-azimuth-bins auto \
  --out-dir <output>
```

`auto` resolves to A256 for `AzimuthFFTTransformerEncoder` checkpoints and A8
for the original `TransformerEncoder`.

## Best Verified Model-Selection Run

The best A256/E1 occupancy checkpoint produced on AutoDL is
`best-primary-017-9000.ckpt` (SHA256
`9572fa71a3a1434073a174c50a892604b625a101ae3dac832463e17d7b6a48b3`).
Its validation `map_f1` is `0.29325`.

The run used:

- official `base/small` occupancy weights as initialization;
- input `[64,256,1,256,2]` and patch `[8,32,1,8]`;
- batch size 32, AdamW, learning rate `1e-4`, warmup 100;
- `PolarOccupancy`, positive weight 16, range-weighted loss;
- `ptrain=0.9`, `pval=0.1`, and 2048 deterministic validation samples;
- no post-cache augmentation;
- `limit_train_batches=500` for model selection.

The cache itself is complete for all 19 configured outdoor traces: 621,013
samples, float16 `[64,8,1,256,2]` shards. The 90% training split contains
558,902 samples, or 17,465 drop-last batches at batch size 32. The 500-batch
cap therefore consumed only 16,000 samples (2.86%) per short epoch. This
checkpoint is a strong initialization and model-selection result, not proof of
complete full-dataset optimization.

## Recommended Full-Data Continuation

The next baseline should continue from the step-9000 checkpoint with the full
train dataloader. Do not restart from official weights and do not add seam or
refiner losses: all three output-smoothing ablations reduced validation F1.

Recommended first run:

- batch size 32, accumulation 1, bf16 mixed precision;
- AdamW learning rate `2e-5`, warmup 250;
- no `limit_train_batches`;
- three full epochs (or a four-hour maximum), validation once per epoch;
- retain the 2048-sample deterministic validation subset;
- select primarily by `map_f1/val`, while also saving best loss, depth, and
  `logit > 1` F1 checkpoints;
- keep augmentation disabled for this controlled coverage experiment. Test
  augmentation separately only after establishing the full-data baseline.

On the verified cache, one full epoch contains 17,465 optimizer steps. Recheck
the actual dataloader length at launch if the trace set or batch size changes.
