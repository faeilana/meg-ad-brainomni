# Project reference

Complete technical reference for the MEG foundation model evaluation. Written so that someone
with no prior context can answer questions about any part of the repository.

Last updated 2026-10-16.

---

## 1. The research question

Do pretrained MEG foundation models capture Alzheimer's-related signal without disease-specific
training? And if so, do they beat the handcrafted spectral biomarkers the field already uses?

This is one half of a joint paper. Fae covers BrainOmni and MEG-Mamba, Sid covers MEG-GPT.
Advisor is Prof Srikantan Nagarajan at the UCSF Biomagnetic Imaging Lab.

**The short answer so far.** Frozen representations work, fine-tuning never improves on them,
three different foundation models land in the same performance band, and six handcrafted
spectral features beat all of them on the one cohort where both exist.

---

## 2. Background concepts

**MEG** measures brain activity by detecting the tiny magnetic fields produced by neural
currents. Sensors sit outside the head in a helmet. Each sensor records a mixture of activity
from the entire brain, like microphones around a concert hall each hearing the whole orchestra.

**Sensor level vs source level.** Sensor-level data is what the machine records, one time
series per physical detector. Source-level data is an estimate of what was happening *inside*
the brain, obtained by solving an inverse problem. This conversion is called **source
reconstruction**, and a **beamformer** is one method for it: for each candidate location in the
brain, it weights the sensors so that activity from that spot passes through and everything
else cancels. It needs an MRI, because the weights depend on head geometry.

**Parcellation** groups brain tissue into labelled anatomical regions. An **atlas** is the map
that defines them. A **voxel** is one small cube of a 3D brain image, typically 2mm. A region is
made of hundreds or thousands of voxels. A region's **centroid** is the average position of its
voxels.

**Atlases used here.**

| Atlas | Regions | Used by |
|---|---|---|
| Brainnetome | 246, of which 210 cortical | Pooja's data as delivered |
| Glasser52 | 52 | MEG-GPT |
| Schaefer100 | 100 | MEG-Mamba |

**Cortical vs subcortical.** The cortex is the wrinkled outer layer. Subcortical structures
(amygdala, hippocampus, caudate, globus pallidus, thalamus) sit beneath it. Brainnetome IDs
1 to 210 are cortical, 211 to 246 subcortical. MEG sees deep structures poorly because magnetic
field strength falls off sharply with distance.

**Sign ambiguity.** When source reconstruction builds a region's time series, it typically takes
the first principal component of the voxels in that region. PCA determines a direction but not a
sign, so the algorithm picks arbitrarily. A region's signal may therefore be inverted relative to
its neighbours. **Sign flipping** is the standard correction. Pooja's data did not receive it,
which is a finding of this project (see section 9).

---

## 3. Cohorts

| Cohort | N | AD | Control | Level | Channels |
|---|---|---|---|---|---|
| OSF | 148 | 78 | 70 | sensor | 271 gradiometers |
| Parcel | 161 | 77 | 84 | source | 246 Brainnetome regions |
| Combined | 309 | 155 | 154 | mixed | n/a |

**OSF** is the public release from Kudo et al., eLife 2023. Raw MEG plus MRI per subject in
`data/meg_mri/`, 297 files, 19 GB. Subject IDs look like `RSID0001`.

**Parcel** came from Pooja, already source reconstructed into Brainnetome regions by a method
that is not documented to us. Two files in `data/ts48/`, `tsAD.mat` and `tsCONT.mat`, 24 GB
including leadfields. Subject IDs look like `R1146`.

**The two cohorts use different anonymisation schemes**, so subject overlap cannot be checked by
ID matching. Zero ID overlap does not prove they are different people. This is unresolved.

**Combined** is simply the two concatenated, OSF first then parcel. Used to test whether more
data rescues fine-tuning.

---

## 4. Data files

### `data/meg_mri/` . OSF raw data, 19 GB
Per subject: `RSIDxxxx_meg_rest_60sec.mat` (60 seconds of resting-state MEG at 600 Hz, 271
channels, plus a `grad` struct holding sensor positions and orientations) and
`RSIDxxxx_mri_anon.mat` (the anatomical MRI, unused by this pipeline since no source
reconstruction is performed).

### `data/ts48/` . Pooja's data, 24 GB
```
tsAD.mat       matrix (36000, 246, 77) float64, labels, radid
tsCONT.mat     matrix (36000, 246, 84) float64, plus exclusion bookkeeping
leadfieldAllAD.mat    6.8 GB, forward models for 99 subjects
leadfieldAllCONT.mat  6.2 GB, forward models for 129 subjects
ts48AD.mat, ts48CONT.mat   a 48-region version, unused
```
`matrix` is time x region x subject. MATLAB v7.3, which is HDF5, so axes come out reversed when
read with h5py. `radid` holds subject codes. `tsCONT.mat` also carries `excludeRadid` listing 6
subjects, all of which were already removed before delivery, so nothing needs excluding.

**The leadfields** are the forward models, meaning step 2 of beamforming already computed. They
cover 228 subjects, 139 of whom are in the parcel cohort. They are not useful because that data
is already reconstructed, and they do not cover OSF at all.

**HDF5 chunking gotcha.** The `.mat` files are chunked `(1, 82, n_subj)`, so reading one subject
touches every chunk in the file and discards the rest. That costs ~25 seconds per subject.
Reading the whole array at once takes ~31 seconds total. `notebooks_mamba/02` reads in batches
of 25 for this reason.

### `data/preprocessed/` . cached sensor pipeline outputs, 3.9 GB
```
proj_out1.npy        (148, 16, 160, 256)  tiny, first 30 s segment
proj_out2.npy        (148, 16, 160, 256)  tiny, second segment
proj_out1_base.npy   (148, 16, 160, 512)  base
proj_out2_base.npy   (148, 16, 160, 512)
labels.npy           (148,)  0 = control, 1 = AD
split.npz            the 80/10/10 split indices
all_seg1.npy, all_seg2.npy, all_pos.npy, all_st.npy   STALE, only 71 subjects, 272 channels
```
The `all_*` files are leftovers from an abandoned partial run and are not inputs to anything.

### Atlas files
```
data/brainnetome_atlas/BN_Atlas_246_2mm.nii.gz    3D label volume, 2mm
data/glasser52/Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz   4D stack of 52 masks, 8mm
data/schaefer100/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz   3D labels, 2mm
```
Glasser52 came from Zenodo record 10401793. Schaefer100 was extracted from the `osl-ephys` pip
package, since it is not bundled with `osl-dynamics`.

### Spreadsheets
`eLife91044_processeddata_scalarmetrics.xlsx` is the one that matters. It supplied the `diagnosis`
column (verified to match `labels.npy` on all 148) and six columns that became the handcrafted
baseline: `Z AEC deltatheta/alpha/beta` and `Z Pow deltatheta/alpha/beta`. AEC is amplitude
envelope correlation, a connectivity measure. Pow is band power. The six `regionalmegmetric`
files are unused.

### Dead files in `data/`
`brainomni_features_tiny.npy`, `brainomni_features_base.npy` and `brainomni_features.csv` are
superseded. Both `.npy` files are 8192-dim and byte-identical, meaning both are base. See
section 9.

---

## 5. BrainOmni

NeurIPS 2025, Xiao et al. Code at github.com/OpenTSLab/BrainOmni. Local copy in `../BrainOmni/`.

### What makes it usable here

Most brain models learn an embedding per channel index, which locks them to one sensor layout.
BrainOmni instead encodes each channel's **physical properties**: 3D position, 3D orientation,
and type. Input is a triple `(X, L, S)` where `X` is the signal `(C, T)`, `L` is `(C, 6)` holding
position and orientation, and `S` is `(C,)` holding type (0 EEG, 1 gradiometer, 2 magnetometer).

**This is why the whole project is possible.** The model does not care what a channel is called
or how many there are, only where it sits. That is what allowed fabricated geometry from atlas
centroids to work, and it is the capability MEG-GPT and MEG-Mamba lack.

### Two stages

**BrainTokenizer** converts raw signal into discrete tokens. A SEANet convolutional encoder
extracts temporal features, a Sensor Encoder turns `L` and `S` into sensor embeddings, and a
cross-attention block with **16 learnable queries** compresses any channel count down to 16
**latent source variables**. Residual vector quantisation with 4 layers discretises them. Trained
as an autoencoder on 2-second segments, with a reconstruction loss combining time domain,
frequency domain and Pearson correlation terms.

The paper frames this as a learned version of source estimation. The tokenizer is the backward
solution, the reconstructor the forward solution.

**BrainOmni** is a Criss-Cross Transformer trained on those tokens with masked token prediction,
50% masking, on **30-second segments**. Criss-Cross means it splits features in half and runs
attention across time on one half and across channels on the other, so it models both when and
where. This is the thing MEG-GPT and MEG-Mamba cannot do, as they model each channel
independently.

### Sizes

| | lm_dim | heads | blocks | params |
|---|---|---|---|---|
| tiny | 256 | 8 | 12 | 8.4M |
| base | 512 | 16 | 12 | 33M |

They differ only in width. Depth is 12 for both, which is why `blocks[:-1]` runs 11 blocks for
either.

### The numbers you will see

```
256 Hz    sampling rate BrainOmni expects
512       tokenizer window in samples = 2 seconds at 256 Hz
160       time windows produced from a 30 s segment
16        neuro tokens / latent sources, always, regardless of input channels
256/512   hidden dimension, tiny / base
4096      16 x 256, feature size for tiny
8192      16 x 512, feature size for base
0.25      overlap ratio between tokenizer windows
```

### `encode()`

```python
def encode(self, x, pos, sensor_type):
    x, _ = self.tokenizer.tokenize(x, pos, sensor_type, self.overlap_ratio)
    neuro = self.tokenizer.encoder.neuros.detach().view(1, C, 1, -1)
    x = self.projection(x + neuro)
    for block in self.blocks[:-1]:
        x = block(x)
    return F.normalize(x, p=2.0, dim=-1, eps=1e-6)
```

`neuros` is `(16, 256)`, one learned vector per latent source, added so the transformer can tell
the 16 slots apart. Same idea as positional embeddings but for source identity.

**`proj_out` is captured partway through this**, after `projection` and before the blocks. This
is the expensive part to compute, so it is cached once and the cheap remainder is rerun as
needed.

**`blocks[:-1]` stops one block short** because the final block is specialised for the masked
prediction objective rather than producing a general representation. This is BrainOmni's own
choice, not an invention of this project.

### Table 9 of the paper, which will come up

The authors found fine-tuning *beats* freezing on their Alzheimer's dataset: tiny 0.774 to 0.795,
base 0.771 to 0.828. This project found the opposite. Three differences explain it:

- They train on 10-second **segments**, 5349 of them from 65 subjects, so thousands of training
  examples. This project treats each subject as one example, so 148 or 161.
- They **fully** fine-tune. This project unfroze only the last 1 to 3 blocks.
- They select the epoch on validation balanced accuracy, this project selects on validation loss.

The segment count is the strongest explanation.

---

## 6. MEG-GPT and MEG-Mamba

Both from the OHBA group at Oxford. Sid works on MEG-GPT.

### MEG-GPT
TensorFlow. Takes **Glasser52** parcellated source data, 52 channels. Tokenises amplitude into
discrete bins (quantile or mu-law tokenizers, 64 tokens), then autoregressive next-token
prediction. Channel identity comes from a learned per-parcel embedding, so there is no geometry
and the channel count is fixed. Pretrained on Cam-CAN, 612 healthy subjects. Pinned environment
in `osl-foundation/envs/oslf.yml` specifies TensorFlow 2.11.0 and Python 3.10.14, which is from
2022 and does not cooperate with modern CUDA. This is why Sid has had install trouble.

### MEG-Mamba
Preprint arXiv 2610.00746, repo github.com/OHBA-analysis/MEG-Mamba. **PyTorch**, not TensorFlow.
Takes **Schaefer100**, 100 channels, at 250 Hz.

```
n_parcels     100      fixed channel count
vocab_size    92       discrete amplitude tokens
d_model       256      hidden dimension
n_layers      4        Mamba-3 blocks, far smaller than BrainOmni's 12
d_state       64       internal state size
sess_emb_feat_dim 292  session feature dimension
```

**Mamba is a state space model, not a transformer.** It carries a fixed-size running state
updated one step at a time, so cost grows linearly with sequence length rather than
quadratically. The tradeoff is that it compresses history rather than retaining exact access to
it. Mamba-3 makes the state update input-dependent, which is what closed the quality gap with
transformers.

**Architecture.** Token embedding plus parcel embedding plus session embedding, all 256-dim and
summed, then 4 Mamba layers, then an output head predicting the next token.

**The session embedding is the zero-shot mechanism.** `compute_session_features.py` computes, per
parcel, the token unigram histogram and bigram transition matrix over the whole recording, then
applies a frozen sqrt → standardise → PCA transform shipped as
`MEG-Mamba/data/session_feature_transform.npz`. So a new recording is described by its own token
statistics rather than requiring retraining. This is what Chet means by "can be applied without
training to new data".

**Tokenizer** is a separate repo, EphysTokenizer, with the trained weights bundled in
`MEG-Mamba/tokenizer/model/`. Its codebook has 128 entries but only 91 are used, and `label_map`
compacts them to 0 to 91, and `tokenize_session(..., remap=True)` (the default) applies that. It
also z-scores each channel internally, so input scale does not matter.

**GPU requirement.** `mamba_ssm` provides `Mamba3` and uses CUDA plus Triton kernels. It will not
build on Apple Silicon. A Colab T4 (compute 7.5) compiles the kernels but crashes at runtime with
a device-side assert. An L4 (8.9) works. The package must be installed from commit `316ed60` of
state-spaces/mamba, since the PyPI release does not contain `Mamba3`.

---

## 7. Pipelines

### `src/`
`matio.py` (88 lines). unified `.mat` loader handling both classic (scipy) and v7.3 (h5py)
formats, fixing HDF5 axis reversal and dereferencing MATLAB string cell arrays. Imported by the
parcel notebooks.

`preprocessing.py` (50 lines). `preprocess_subject()` for the sensor cohort. Reads the `.mat`,
resamples 600 → 256 Hz via `resample_poly(up=32, down=75)`, aligns the 271 used channels against
the 301 listed in the helmet description, builds `(C, 6)` geometry from `chanpos` and `chanori`,
centres and scale-normalises the positions, sets all sensor types to 1 (verified: all 271 are
`meggrad`), and z-normalises each channel.

**Only `notebooks_archive/` imports it**, via `finetune_setup.ipynb` and
`extract_features.ipynb`. Nothing in the current pipeline does, so it is frozen, but every
sensor-level number traces back through it. The parcel notebooks reimplement the same logic
inline. It contains the epsilon bug described in section 9.

### `notebooks_archive/`
Superseded, kept as a record. **Do not run.** `finetune_setup.ipynb` produced `proj_out` and
`labels.npy` for the sensor cohort, using `src/preprocessing.py`, and `extract_features.ipynb`
produced `baseline_features.npy` from the spreadsheet and the now-dead `brainomni_features_*`
files. `finetune_neuralnet.ipynb` is the predecessor of `notebooks_sensor/02` and **writes the
same `finetune_shared.pt` but saves less**, so running it would destroy results.

### `notebooks_sensor/` . OSF cohort, 148 subjects, sensor level

```
00_extract_frozen_features   proj_out -> frozen features, both model sizes
01_baselines                 frozen vs handcrafted, 5-fold CV
02_finetune_tiny_nn          fine-tune tiny, 3 depths, NN and LR heads, CV + 2 single splits
03_finetune_base_nn          same for base
04_bootstrap_comparison      paired bootstrap CIs across all methods
05_algorithms_check          10 classifiers on frozen features
06_pca_algorithms            accuracy vs PCA components, 1 to 50, per classifier
07_head_nn_vs_lr             NN head vs LR head bar charts
08_head_lr_single_splits     LR head on the two single splits, rebuilt from saved weights
```

`00` loads each checkpoint separately in one loop and asserts the resulting arrays differ, which
is the fix for the duplicate-file bug.

`02` and `03` hold the training machinery: `CachedMEGDataset`, `train_finetune`, `run_cv`,
`run_split`. They save `finetune_shared.pt` and `finetune_shared_base.pt` containing out-of-fold
predictions, per-fold histories, and `best_weights` for every configuration.

`08` depends on those `best_weights`: it rebuilds each fine-tuned backbone and fits an LR head
without retraining, which is why it runs in minutes.

### `notebooks_parcel/` . parcel cohort, 161 subjects, source level

```
00_parcel_geometry       centroids from Brainnetome -> (210, 6) geometry
01_parcel_setup          tokenise with that geometry -> proj_out
02_parcel_baselines      frozen features and 5-fold CV
03_combined              pool with OSF to 309 subjects
04_finetune_combined     fine-tune the combined cohort, 3 depths x 2 heads x 2 sizes
05_bootstrap_ci          paired bootstrap, and the 210 vs 52 comparison
06_plot_comparisons      charts
07_algorithm_sweep       classifier sweep
08_pca_curve             PCA curve
09_cohort_balance        checks the batch effect cannot inflate the pooled result
10_glasser52_mapping     Brainnetome 210 -> Glasser52 weight matrix, plus the sign diagnosis
11_glasser52_setup       tokenise the Glasser52 version
```

`09_head_comparison.ipynb` was never run and contributes nothing.

`04` has fold-level checkpointing with `set_seed(seed + fold)`, so an interrupted run resumes
and reproduces an uninterrupted one exactly. `RESUME = True`, and the driver skips configurations
already present in `cv_results`.

### `notebooks_mamba/` . MEG-Mamba on the parcel cohort

```
00_schaefer100_mapping   Brainnetome 210 -> Schaefer100 weight matrix and geometry
01_feasibility           does the tokenizer recognise remapped data? compares against noise
02_tokenize_all          tokenise all 161, build session features
colab_extract            runs on a Colab L4, extracts hidden states
03_mamba_baselines       evaluation and paired comparison against BrainOmni
```

`colab_extract` is the only part that leaves the local machine. It runs `model.embed(...)` then
`model.backbone(...)`, stopping before `readout` since that projects to next-token logits. Pools
over 12 windows of 1000 tokens, giving 100 x 256 = 25600 features per subject.

---

## 8. Methods

**Cross-validation.** Stratified 5-fold, `shuffle=True`, `random_state=123`, everywhere. For
fine-tuning, a nested scheme: within each training fold, a further 15% is carved out as an inner
validation set used solely for epoch selection. The test fold is evaluated every epoch and logged
for the learning curves but **never influences any decision**. The selection line is
`if vll < best_vl` where `vll` is inner-validation loss, not the monitored test loss.

**Metrics.** Two conventions appear and differ by a few thousandths:
- *mean of folds*: average the five per-fold scores. This is what `cross_validate` returns.
- *pooled out of fold*: collect all 148 or 161 held-out predictions and compute one score.

The bootstrap requires pooled, so pooled is the stated convention. Example of the gap: frozen
tiny AUC is 0.859 by mean-of-folds and 0.866 pooled.

**Bootstrap.** 2000 resamples, seed 123. For method comparisons it is **paired**: one resample
per iteration, both methods scored on it, differences collected. This is tighter than comparing
overlapping confidence intervals, because subject-level variance cancels.

**Fine-tuning recipe.** AdamW, betas (0.9, 0.99), eps 1e-6, weight decay 0.05, label smoothing
0.1, warmup-cosine with 10% warmup, gradient clipping 1.0, differential learning rates with
backbone 1e-5 and head 1e-4. Depth is set by `n_frozen`: with 12 blocks, `n_frozen=10` trains
`blocks[10:-1]` which is last-1, `n_frozen=9` is last-2, `n_frozen=8` is last-3.

**Two deviations from the paper** worth knowing: differential learning rates are not specified by
the authors (they tried 3e-6, 1e-5, 3e-5 uniformly), and they select on validation balanced
accuracy rather than loss.

**PCA merge for parcellation mapping.** When merging regions into coarser parcels, averaging
fails because of arbitrary signs. Instead the first principal component of the member regions is
taken, which is invariant to input signs. Measured: averaging retains 0.64 of input amplitude on
average and 0.23 at worst. PC1 retains 1.54 and 0.93. This is also what `osl-ephys` does.

---

## 9. Known issues

### The epsilon bug . unresolved, affects sensor cohort only

```python
std = signal.std(axis=1, keepdims=True) + 1e-13
```

MEG is measured in Tesla, so raw per-channel standard deviations are around 1.4e-13. The epsilon
guarding against division by zero is the same order of magnitude, so the denominator is roughly
double what it should be. Normalised channels come out with standard deviation 0.41 to 0.72
instead of 1.0, and the distortion is **channel-dependent**, compressing relative amplitudes.

The parcel pipeline is unaffected: Pooja's data is order 1, so 1e-13 is 13 orders of magnitude
too small to matter and normalisation lands at exactly 1.000.

**Impact.** Every subject and method went through the identical transform, so all comparisons
remain valid. What is unknown is whether the absolute sensor-cohort numbers would change if
fixed. The recommended next step is a 20-subject test extracting features both ways and
comparing, which is about 5 minutes, before deciding whether a rerun is warranted.

### Sign flipping absent from Pooja's data . found, worked around

Across 8 subjects, only **10.9%** of within-parcel region pairs keep the same correlation sign.
Genuine anticorrelation would be consistent between people. Random sign per subject is the
signature of an uncorrected arbitrary convention. Within-parcel correlations are strong in
magnitude (mean |r| 0.47) but 41% are negative.

Explicit sign flipping was attempted and failed, moving negatives only from 41% to 37%, because
estimating flips from one subject's noisy data is unreliable. PCA merging sidesteps the problem
instead.

**Implication.** The 210-region results are likely a floor. Properly sign-flipped data could do
better. Worth requesting from Pooja.

### Duplicate feature file bug . fixed

`brainomni_features_tiny.npy` and `brainomni_features_base.npy` in `data/` are byte-identical and
both 8192-dim, meaning both are base. The original `extract_features.ipynb` was parameterised by
a `model_name` variable that had to be changed in step with the checkpoint path, and at some
point they diverged. `notebooks_sensor/00` fixed this by looping over both in one pass with a
distinctness assertion. The stale files remain in `data/` and should be deleted.

### Unresolved questions for Sri

1. **Was sign flipping applied to Pooja's data?** Confirmed absent by measurement, but worth
   asking whether a corrected version exists.
2. **Do the cohorts share subjects?** Cannot be checked by ID. `notebooks_parcel/04` supports
   `SUBJECT_GROUPS` with `StratifiedGroupKFold` but it is off for lack of a mapping.
3. **Cortical only, or include subcortical?** Hippocampus matters for Alzheimer's.
4. **Was the OSF data already filtered?** `preprocessing.py` applies no bandpass or notch,
   while the BrainOmni paper applies 0.1 to 96 Hz plus 50/60 notch before resampling.
5. **Can Sid share source-reconstructed OSF data?** This would put all three models on the same
   148 subjects, which is the weakest part of the study as it stands.

---

## 10. Results

All frozen plus logistic regression, stratified 5-fold, seed 123, pooled out-of-fold. Every
number below was verified by recomputation from the saved predictions.

| Method | Cohort | N | Accuracy | AUC |
|---|---|---|---|---|
| Handcrafted spectral, 6 features | OSF | 148 | 0.872 | 0.925 |
| BrainOmni-tiny, Glasser52 | parcel | 161 | 0.870 | 0.904 |
| BrainOmni-tiny, Brainnetome 210 | parcel | 161 | 0.851 | 0.913 |
| BrainOmni-base, sensor | OSF | 148 | 0.831 | 0.878 |
| BrainOmni-tiny, sensor | OSF | 148 | 0.831 | 0.866 |
| BrainOmni-tiny, combined | combined | 309 | 0.832 | 0.899 |
| BrainOmni-base, combined | combined | 309 | 0.825 | 0.903 |
| MEG-Mamba, Schaefer100 | parcel | 161 | 0.826 | 0.890 |

Chance is 0.527 on OSF, 0.522 on parcel, 0.502 on combined.

### Fine-tuned, sensor cohort (tiny)

| Depth | Head | Acc | AUC |
|---|---|---|---|
| last-1 | NN | 0.791 | 0.860 |
| last-1 | LR | 0.797 | 0.872 |
| last-2 | NN | 0.784 | 0.859 |
| last-2 | LR | 0.791 | 0.853 |
| last-3 | NN | 0.777 | 0.859 |
| last-3 | LR | 0.777 | 0.852 |

All below the 0.831 frozen baseline. Paired against frozen, last-3 is significantly worse:
−0.054 accuracy, 95% CI [−0.095, −0.014].

### Fine-tuned, combined cohort

| Depth | Head | tiny Acc | tiny AUC | base Acc | base AUC |
|---|---|---|---|---|---|
| last-1 | NN | 0.845 | 0.899 | 0.841 | 0.893 |
| last-1 | LR | 0.838 | 0.902 | 0.838 | 0.902 |
| last-2 | NN | 0.825 | 0.884 | 0.828 | 0.883 |
| last-2 | LR | 0.848 | 0.890 | 0.828 | 0.895 |
| last-3 | NN | 0.832 | 0.877 | 0.835 | 0.876 |
| last-3 | LR | 0.845 | 0.885 | 0.828 | 0.893 |

Every paired interval against frozen contains zero. The best configuration's AUC difference is
−0.000.

### Key paired comparisons

```
52 Glasser parcels vs 210 Brainnetome regions, same 161 subjects
   accuracy  +0.019  [-0.012, +0.050]   not significant
   AUC       -0.009  [-0.033, +0.015]   not significant

MEG-Mamba vs BrainOmni Glasser52, same 161 subjects
   accuracy  -0.043  [-0.099, +0.012]   not significant
   AUC       -0.014  [-0.050, +0.023]   not significant
```

### Five findings

1. **Reconstructed geometry works.** Parcellated input reaches 0.913 AUC against 0.866 for the
   same model on raw sensor data. Different cohorts, so not controlled, but the model clearly
   handles fabricated geometry.
2. **Fine-tuning never beats frozen.** 14 configurations. Significantly worse at 148 subjects,
   indistinguishable at 309. More data removed the harm but produced no benefit. Training curves
   show why: training accuracy reaches 0.993 to 1.000 while validation loss rises.
3. **Pooling cohorts costs accuracy.** A classifier separates the two cohorts at **AUC 1.000**,
   median standardised difference 1.78 per dimension. Per-cohort standardisation recovers AUC
   from 0.898 to 0.909 but does not close the gap. `09_cohort_balance` confirms the batch effect
   cannot be inflating the result, since a cohort-only classifier reaches just 0.524.
4. **Spatial resolution barely matters.** 52 parcels match 210 regions. Matching Sid's
   parcellation therefore does not handicap BrainOmni.
5. **Three foundation models land in the same band**, 0.826 to 0.870, none separating from the
   others, and handcrafted spectral features beat all of them on the one cohort where both exist.

### Caveats on the comparisons

- Parcellation is confounded with model in the MEG-Mamba comparison, Schaefer100 against
  Glasser52.
- MEG-Mamba features are 25600-dimensional against BrainOmni's 4096, so it was judged under
  harder conditions for the same classifier. PCA to equal dimensions would remove this objection.
- The parcel cohort's input to MEG-Mamba was **remapped**, not beamformed, to Schaefer100.
  `01_feasibility` confirms the tokenizer still recognises it (4.84 bits entropy against 3.57 for
  white noise) but this is not equivalent to proper beamforming.
- Handcrafted exists only for OSF, so it cannot be compared on the parcel cohort.
- A single-feature check showed delta-theta power alone reaches 0.883 AUC. Removing it, the other
  five still reach 0.915, so the handcrafted result is not carried by one feature. But those six
  were published by the eLife authors *because* they separate AD in this cohort, so the baseline
  is tuned to the data.

---

## 11. Environment

`conda activate brainomni`. PyTorch 2.12.0 on Apple Silicon MPS. Sibling repositories under
`Neuroimaging/`: `BrainOmni`, `EphysTokenizer`, `MEG_Mamba`, `osl-foundation`. None are pip
installable, so notebooks add them with `sys.path.insert`, and `.vscode/settings.json` lists them in
`python.analysis.extraPaths` so VSCode can resolve imports.

`*.npy` and `*.pt` are gitignored. They total several GB and are regenerable.

**Gotchas encountered.** The tokenizer defaults to `cuda if available else cpu`, so on a Mac it
silently runs on CPU unless `device='mps'` is passed. Loading a full 2.7 GB array plus accumulating
outputs pushed a 16 GB machine into swap, collapsing throughput from 1.1 to 5.5 minutes per
subject, and batched reads fixed it.
