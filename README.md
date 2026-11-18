# MEG foundation models for Alzheimer's classification

Evaluating whether pretrained MEG foundation models capture Alzheimer's-related
signal without disease-specific training. Part of a joint paper with Sid, who covers
MEG-GPT.

All results use the same protocol so they are directly comparable: stratified 5-fold
cross-validation, seed 123, logistic regression, metrics computed on pooled out-of-fold
predictions, and 2000-resample bootstrap confidence intervals.

---

## Cohorts

| Cohort | N | Level | Source |
|---|---|---|---|
| OSF | 148 | sensor, 271 gradiometers | Kudo et al. eLife 2023 |
| Parcel | 161 | source, Brainnetome 246 | Pooja, already reconstructed |
| Combined | 309 | mixed | the two concatenated |

The two cohorts use different anonymisation schemes (`RSID0001` vs `R1146`), so
subject overlap cannot be checked by ID matching.

---

## Notebook folders

Each folder is a pipeline over one cohort. Run in numeric order.

### `notebooks_sensor/` . OSF cohort, sensor level

Starts from cached `proj_out` in `data/preprocessed/`, not from raw `.mat`.

```
00  frozen features for tiny and base, each from its own checkpoint
01  baselines: frozen features and handcrafted spectral features
02  fine-tune tiny, 3 unfreezing depths, NN and LR heads
03  fine-tune base, same sweep
04  paired bootstrap confidence intervals
05  comparison plots
06  algorithm sweep beyond logistic regression
07  accuracy vs number of PCA components
08  NN head vs LR head
09  LR head across train/val/test splits
```

### `notebooks_parcel/` . Pooja's cohort, source level

```
00  geometry from Brainnetome centroids       position + orientation per region
01  tokenise with that geometry               -> proj_out
02  frozen features and evaluation
03  pool with OSF, 309 subjects
04  fine-tune the combined cohort, 3 depths x 2 heads x 2 model sizes
05  paired bootstrap, and the 210 vs 52 comparison
06  comparison plots
07  algorithm sweep
08  PCA curve
09  cohort balance, checks the batch effect cannot inflate the pooled result
10  map Brainnetome 210 -> Glasser52, to match Sid's parcellation
11  tokenise the Glasser52 version
```

### `notebooks_mamba/` . MEG-Mamba on the parcel cohort

```
00  map Brainnetome 210 -> Schaefer100
01  feasibility: does the tokeniser recognise remapped data?
02  tokenise all subjects, build session features
    --- colab_extract.ipynb runs on a Colab GPU, see below ---
03  evaluation and paired comparison against BrainOmni
```

`colab_extract.ipynb` runs elsewhere because `mamba_ssm` needs CUDA and an Ampere
GPU or newer. It will not build on Apple Silicon, and a Colab T4 compiles the kernels
but crashes at runtime. Upload `tokens_schaefer100.npy` and `session_feats.npy`,
download `mamba_features.npy` back into `results_mamba/`.

### `notebooks_archive/`

Superseded, but it is the provenance for every sensor-level number.
`finetune_setup.ipynb` produced `data/preprocessed/`, and `extract_features.ipynb`
produced `data/baseline_features.npy`. Those two are the only code that imports
`src/preprocessing.py`. Not part of the pipeline, do not rerun.

---

## Key method choices

**Reconstructed geometry.** BrainOmni needs a position and orientation per channel.
Parcellated data has no sensor layout, so geometry is built from the atlas: position is
the region centroid, orientation is the normalised origin-to-centroid vector,
approximating the outward cortical surface normal. Centred and scale-normalised to match
what `src/preprocessing.py` does to real sensor coordinates.

**PCA merge, not averaging.** Pooja's regions carry arbitrary signs, a step her pipeline
skipped. Across 8 subjects only 11% of within-parcel region pairs keep the same
correlation sign, which is the signature of arbitrary signs rather than real
anticorrelation. Averaging such regions cancels signal, one parcel retained 23% of its
input amplitude. PC1 is invariant to input signs, and matches what `osl-ephys` does.

**Cortical only.** 210 of Brainnetome's 246 regions. MEG has poor sensitivity to deep
sources, and the origin-to-centroid orientation is meaningless for a structure near the
centre of the brain.

**No selection on the test fold.** Fine-tuning epochs are chosen on an inner validation
split carved from the training fold. The test fold is logged for the learning curves but
never influences any decision.

**Paired bootstrap.** Method comparisons draw one resample per iteration and evaluate
both methods on it, so subject-level variance cancels. Comparing overlapping confidence
intervals would be weaker.

---

## Results

Frozen features plus logistic regression, pooled out-of-fold.

| Method | Cohort | N | Accuracy | AUC |
|---|---|---|---|---|
| Handcrafted spectral, 6 features* | OSF | 148 | 0.872 | 0.925 |
| BrainOmni-tiny, Glasser52 | parcel | 161 | 0.870 | 0.904 |
| BrainOmni-tiny, Brainnetome 210 | parcel | 161 | 0.851 | 0.913 |
| BrainOmni-base, sensor | OSF | 148 | 0.831 | 0.878 |
| BrainOmni-tiny, sensor | OSF | 148 | 0.831 | 0.866 |
| BrainOmni-tiny, combined | combined | 309 | 0.832 | 0.899 |
| MEG-Mamba, Schaefer100 | parcel | 161 | 0.826 | 0.890 |

Chance is 0.522 on the parcel cohort, 0.527 on OSF.

*The handcrafted features are not computed here. They are the published `Z Pow` and
`Z AEC` columns from the Kudo et al. supplement, so the baseline is that paper's own
tuned result on its own data.*

### Findings

**Reconstructed geometry works.** Parcellated input reaches 0.913 AUC against 0.866 for
the same model on raw sensor recordings. Different cohorts, so not a controlled
comparison, but the model clearly handles fabricated geometry.

**Fine-tuning never beats frozen.** 14 configurations across two model scales, three
unfreezing depths and two heads. On the 148-subject cohort it was significantly worse,
by up to 0.054 accuracy. On the 309-subject cohort every paired interval contains zero.
More data removed the harm but produced no benefit. Training curves show why: training
accuracy reaches 0.99 to 1.00 while validation loss rises.

**Pooling cohorts costs accuracy.** Combined underperforms parcel alone. A classifier
separates the two cohorts at AUC 1.000 with a median standardised difference of 1.78 per
dimension, so domain shift exceeds the disease signal. Per-cohort standardisation
recovers part of it but does not close the gap. Notebook `09` confirms the batch effect
cannot be inflating the pooled result, since a cohort-only classifier reaches just 0.524.

**Spatial resolution barely matters.** 52 Glasser parcels match 210 Brainnetome regions,
paired difference +0.019 accuracy [−0.012, +0.050], not significant. Matching Sid's
parcellation therefore does not handicap BrainOmni.

**Three foundation models land in the same band**, 0.826 to 0.870, none separating from
the others. On the one cohort where a handcrafted baseline exists, six spectral features
beat all of them.

---

## Caveats

- Parcellation is confounded with model in the MEG-Mamba comparison, Schaefer100 against
  Glasser52.
- MEG-Mamba features are 25600-dimensional against BrainOmni's 4096, so it was evaluated
  under harder conditions for the same classifier.
- The parcel cohort's input to MEG-Mamba was remapped rather than beamformed to
  Schaefer100, which is not how its tokeniser was trained. Notebook `01` checks the
  tokeniser still recognises it, but this is not equivalent.
- The handcrafted baseline exists only for OSF, so it cannot be compared on the parcel
  cohort.
- Pooja's data was never sign flipped, so the parcel results are likely a floor.
- Whether the two cohorts share subjects is unresolved.

---

## Layout

```
data/
  meg_mri/            OSF raw MEG and MRI, 148 subjects
  ts48/               Pooja's parcellated data and leadfields
  preprocessed/       cached proj_out and labels for the OSF cohort
  brainnetome_atlas/  Brainnetome 246
  glasser52/          Glasser52, from Zenodo 10401793
  schaefer100/        Schaefer100, extracted from the osl-ephys package
src/
  matio.py            unified .mat loader, classic and v7.3/HDF5
  preprocessing.py    sensor preprocessing, used only by notebooks_archive
results_sensor/       OSF cohort outputs
results_parcel/       parcel and combined cohort outputs
results_mamba/        MEG-Mamba outputs
writeup/              paper draft
```

`*.npy` and `*.pt` are gitignored. They are regenerable and run to several GB.

## Environment

`conda activate brainomni`. PyTorch on Apple Silicon MPS. BrainOmni and the
EphysTokenizer live outside this repo, in sibling folders under `Neuroimaging/`.
