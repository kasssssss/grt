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

- `aperture_truncate`
- `aperture_window`
- `gaussian_coherent`
- `sector_coherent`
- `sector_energy_circular`
- `sector_max_energy`

On three fixed RADs frames, sector-coherent reduction preserved the RA energy
layout best, while sector-max-energy gave the best sparse-GT F1/recall at
`logit > 0`. Absolute F1 is not directly comparable to I/Q-1M LiDAR occupancy
because `RADs_gt` is sparse radar occupancy.

`aperture_truncate` is the only reducer algebraically matched to the A256
training representation. It applies inverse azimuth FFT, keeps the first eight
aperture coefficients, and transforms those coefficients back to A8. For an
A256 tensor produced by the I/Q-1M zero-padded-aperture path, this is an exact
inverse up to floating-point error. The sector reducers remain useful empirical
baselines for pseudo-angular RADs data, but they do not invert the training
transform.

The RADs audit found a boundary-straddling effective aperture. A window learned
from sequence 100 selected circular indices `[253,254,255,0,1,2,3,4]`, rather
than `[0..7]`. Use `aperture_window --aperture-start -3` for this globally
calibrated contract. Do not search the window independently for each evaluation
frame; calibration and evaluation sequences must remain disjoint.

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

## Failed Visibility-Only Supervision Ablation

Masking every voxel after the first LiDAR return was tested for one complete
558,902-sample training epoch from `best-primary-017-9000.ckpt`. It improved
validation first-hit depth MAE from `6.078` to `4.544` range bins, but collapsed
`map_f1` from `0.28725` to `0.04171` and filled the predicted BEV with dense
fan-shaped occupancy. The post-hit volume became unconstrained.

Do not use visibility-only masking. Retain dense BCE and test boundary-specific
supervision only as a low-weight auxiliary ray-termination loss. The auxiliary
term must remain disabled by default so official configurations and checkpoints
retain their original behavior.

The follow-up dense-BCE plus ray-termination experiment used a calibrated
weight of `0.005` (about 15% of baseline BCE at initialization). It improved
I/Q-1M `map_f1` from `0.28725` to `0.32066` and Chamfer from `1.939` to
`1.651`, but reduced recall from `0.695` to `0.525`, worsened depth MAE from
`6.078` to `6.282`, and reduced the fixed RADs three-frame `logit > 0` sparse
GT F1 from `0.01162` to `0.01102`. It also compressed high-confidence RADs
responses. This auxiliary loss is therefore not retained in the production
configuration.

## Half-Patch Shift Decoder Ablation

The 3D occupancy decoder emits disjoint `8 x 8 x 8` voxel patches. A
parameter-free shifted branch was tested to expose each interior voxel to a
second, half-patch-offset query grid. It reuses the existing decoder and
unpatch weights, blends only the interior volume, adds no state-dict keys, and
is bit-exact when `shift_blend: 0`. The optional ablation configuration uses a
conservative `shift_blend: 0.25`; the default model remains unchanged.

On the fixed 2,048-sample RADs-like I/Q-1M validation subset, applying this
branch to `best-primary-017-9000.ckpt` without updating any weights changed:

- `map_f1`: `0.29321 -> 0.30037`
- decoder patch-boundary/interior jump ratio: `1.988 -> 1.745`
- first-hit depth MAE: `6.988 -> 7.091` range bins
- BCE: `0.14148 -> 0.14160`

The fixed three-frame RADs sparse-GT F1 changed from
`0.00992/0.01162/0.01443` to `0.01028/0.01126/0.01610` at logit thresholds
`-1/0/+1`, respectively. The branch therefore improves source F1, patch
continuity, and high-confidence RADs responses, but is an inference-time
tradeoff rather than a universal transfer improvement.

Do not fine-tune the full model merely to adapt this branch. One full
558,902-sample epoch reduced `map_f1` to `0.28953`, raised the jump ratio to
`1.908`, and reduced all three fixed-frame RADs F1 values. A separate
3,000-step unpatch-only run also failed (`map_f1=0.28013`, depth MAE `6.761`).
Both checkpoints are rejected. Retain the original step-9000 checkpoint and
use the shifted branch only as the documented optional ablation.

## Doppler Contract Audit

I/Q-1M's 64 Doppler bins cover only about `[-1.22, 1.18] m/s`, while the
RADs-like target grid is the half-open interval `[-90, 90) m/s` with
`2.8125 m/s/bin`. The legacy transform maps all I/Q-1M bins to the center,
computes a complex mean, and applies a Gaussian. This can cancel opposite
phases and copies one phase estimate into unobserved velocities.

Three explicit variants are now separated:

- `rads_like.yaml`: legacy checkpoint-compatible mean plus complex Gaussian;
- `rads_like_native_doppler.yaml`: native index grid, single elevation, no blur;
- `rads_like_power_doppler.yaml`: physical mapping with RMS magnitude and
  strongest-bin phase, leaving unobserved target bins at zero;
- `rads_like_power_blur_doppler.yaml`: the same non-cancelling merge followed
  by power-preserving Gaussian blur and a documented sensor-scale calibration.

On the same 76 I/Q-1M frames with `best-primary-017-9000.ckpt`, the unmodified
legacy input scored `map_f1=0.27797`. Raw native and raw RMS inputs are strongly
out of distribution (`0.01937` and `0.09729`). Power-preserving blur plus a
stored-magnitude scale of `0.35` recovered `map_f1=0.27458` without training.
A scale of `0.25` achieved Chamfer `2.25157`, better than legacy `2.36450`, at
`map_f1=0.26605`. This is smoke-set calibration, not final model selection;
confirm it on a sequence-disjoint larger cache before full preprocessing.

The follow-up check used 256 later frames from each of `outdoor/baum` and
`outdoor/cmu.east`, excluding the first 256 frames. Legacy input scored
`map_f1=0.27951`; the power-preserving scale-0.35 input scored `0.28142`.
Depth and Chamfer remained worse (`11.21` and `3.21` versus `10.22` and
`2.89`), so this transform is a validated training candidate rather than a
drop-in improvement for every metric of the legacy checkpoint. A narrower
`sigma=0.4` kernel matched RADs' three-bin Doppler support but reduced the old
checkpoint's 76-frame F1 to about `0.236`; retain it only as a from-scratch
ablation.

## Cross-Domain Input Audit

`scripts/audit_rads_iq1m_input_distribution.py` compares model-input power
profiles and phase statistics on matched A8 tensors. On 512 held-out I/Q-1M
frames and 32 evenly sampled RADs frames:

- azimuth was the closest axis (`JS=0.082` for power-preserving I/Q-1M versus
  cropped RADs), supporting the matched aperture projection;
- Doppler still differed (`JS=0.230`): RADs had about three active bins while
  the checkpoint-compatible I/Q-1M transform had eleven;
- range differed most structurally (`JS=0.273`): after valid near-range
  cropping, RADs power had median/95th-percentile bins `5/18`, versus `39/249`
  for I/Q-1M;
- the A256 RADs tensor lost about `61.5%` relative L2 energy after fixed
  A8 projection and re-expansion.

A partial near-range shift was also rejected. With `crop_fraction=0.5`, the
RADs-to-power-I/Q-1M range JS divergence increased from `0.273` to `0.427`.
Keep the full automatically detected shift for the complete 3D cube; matching
only the median range bin does not match the full range-energy distribution.

The last result does not by itself justify a VAE. A sequence-disjoint complex
PCA rank-8 basis retained more than 99% of RADs energy, so the bottleneck is not
rank alone; it is preserving the A8 coordinate semantics expected by the GRT
checkpoint. A stochastic VAE would add phase noise and encourage smoothing.
If a learned adapter is needed, use a deterministic complex A256-to-A8
projection initialized from `aperture_truncate`, train it jointly with an A8
reconstruction decoder, and constrain its I/Q-1M output to the physical A8
target while optimizing the downstream occupancy/depth objective. This makes
the adapter testable against the fixed projection instead of hiding a changed
coordinate system inside an unconstrained latent code.

For the next controlled adaptation, retain the validated optimizer and batch
contract from `map_a256_dim_align_b32_20260711`: batch size 32, no gradient
accumulation, AdamW at `1e-4`, `ptrain/pval=0.9/0.1`, and 2,048 deterministic
validation samples. Validate every 2,000 optimizer steps. Changing the input
transform and effective batch size simultaneously would make the ablation
uninterpretable.
