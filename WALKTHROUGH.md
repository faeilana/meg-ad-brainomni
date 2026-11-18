# Cell index

What every cell does, one line each. Numbers are zero-indexed and count markdown cells,
so `[4]` is the fifth thing you scroll past in Jupyter.

For concepts, data, model internals and known issues, see `REFERENCE.md`.

---

# notebooks_sensor

The OSF cohort, 148 subjects, sensor level. Starts from the cached tokenizer output in
`data/preprocessed/`, not from raw recordings.

## 00_extract_frozen_features.ipynb

- `[0]` Heading.
- `[1]` Imports, picks the GPU, loads the labels so `N` is known.
- `[2]` Runs every subject through the frozen model and saves one 4096-dim feature vector each. Does this twice, once for tiny and once for base.
- `[3]` Checks the two feature files are not accidentally identical, then runs a quick logistic regression on each to see roughly where they land.

## 01_baselines.ipynb

- `[0]` Heading.
- `[1]` Loads the three feature sets being compared. Frozen tiny, frozen base, and the six handcrafted spectral features.
- `[2]` Logistic regression on all three. This is the cell that produces the headline numbers, frozen 0.831 against handcrafted 0.872.
- `[3]` Repeats it with eight different classifiers, to check logistic regression is not the thing holding the score down.

## 02_finetune_tiny_nn.ipynb

The long one. It answers whether updating BrainOmni's weights on this dataset beats
leaving them frozen.

- `[0]` Heading.
- `[1]` Imports for training, metrics and plotting.
- `[2]` Paths, the logging helpers, the epoch budget per unfreezing depth, and the two train/val/test splits. Also loads the cached tokenizer output.
- `[3]` All the model plumbing. The dataset wrapper, the classifier with its neural net head, the learning rate schedule, and the caching helper that runs the frozen blocks once instead of every epoch.
- `[4]` The training loop. Trains, tracks the best epoch by validation loss, and restores those weights at the end.
- `[5]` Wraps that in nested 5-fold cross-validation. Picks the epoch on a slice carved out of the training data, never on the test fold.
- `[6]` Section heading.
- `[7]` Runs the two single splits across three unfreezing depths. Six runs.
- `[8]` Section heading.
- `[9]` Runs the cross-validation across the same three depths. Saves after each one so a crash does not cost the whole sweep.
- `[10]` Training curves for the cross-validated runs. This is the figure that shows the overfitting, training accuracy near 1.0 while validation loss climbs.
- `[11]` Section heading.
- `[12]` Bar chart of accuracy per depth, with the frozen and handcrafted baselines drawn as horizontal lines. Every bar sits below both.
- `[13]` Section heading.
- `[14]` The same training curves for the single-split runs. Skips itself cleanly if you only ran cross-validation.
- `[15]` Section heading.
- `[16]` Confusion matrices, one per depth.
- `[17]` Section heading.
- `[18]` ROC curves, all three depths on one plot.
- `[19]` Section heading.
- `[20]` Saves everything to `finetune_shared.pt`. Notebooks 04 through 08 read this file instead of retraining, which is why they run in seconds.

## 03_finetune_base_nn.ipynb

The same notebook cell for cell, pointed at the base checkpoint instead of tiny. Its head
is wider, 512 rather than 256, and it writes to its own log and checkpoint files. Kept as
a separate copy so an edit aimed at base cannot silently change the tiny results.

## 04_bootstrap_comparison.ipynb

- `[0]` Heading.
- `[1]` Loads both fine-tuning runs and both baselines, and refits the baselines fold by fold so every subject has one held-out prediction. Bootstrapping needs per-subject predictions, which is why this step exists.
- `[2]` Confidence intervals by resampling subjects 2000 times and taking the middle 95 percent.
- `[3]` Section heading.
- `[4]` The paired comparison against the handcrafted baseline. Each resample scores both methods, so subject difficulty cancels out and the difference is what gets measured. This is the cell behind the claim that fine-tuning is significantly worse on 148 subjects.

## 05_algorithms_check.ipynb

- `[0]` Heading.
- `[1]` Loads whichever feature set you point it at.
- `[2]` Builds a list of roughly fifteen classifiers, each wrapped in a pipeline so scaling happens inside the fold rather than before the split.
- `[3]` Cross-validates all of them and prints them ranked. The useful result is negative, nothing beats logistic regression by enough to matter, so the ceiling is in the features.

## 06_pca_algorithms.ipynb

- `[0]` Heading.
- `[1]` Setup, and the range of component counts to try, 1 to 50.
- `[2]` The same classifier list.
- `[3]` For each classifier, compresses the features to n components and re-scores, sweeping n. PCA sits inside the pipeline so it refits per fold.
- `[4]` Plots every curve, with logistic regression in thick black and the baselines as reference lines. Accuracy rises to about 10 components then flattens, which says 4096 dimensions on 148 subjects is far past the point of diminishing returns.
- `[5]` Prints each classifier's best score and the component count that got there.

## 07_head_nn_vs_lr.ipynb

- `[0]` Heading.
- `[1]` Loads both saved fine-tuning runs.
- `[2]` Bar chart comparing the two single splits against the cross-validated mean. Shows how much a single split bounces around, which is the argument for trusting the cross-validated number.
- `[3]` Bar chart comparing the neural net head against a logistic regression head on the same fine-tuned features. They land on top of each other, so the head is not the bottleneck.
- `[4]` Prints every configuration as text, for copying into the spreadsheet.

## 08_head_lr_single_splits.ipynb

Fills the one gap in notebook 07. The logistic regression head was computed for the
cross-validation folds but not for the single splits, so that panel was missing.

- `[0]` Heading.
- `[1]` Imports, paths, features and labels.
- `[2]` Redefines the classifier class. It has to match notebooks 02 and 03 exactly or loading the saved weights fails. This is the most fragile link in the repo.
- `[3]` Reloads the saved best weights for each model and depth, re-extracts features, and fits logistic regression on each split. Caches the result so reruns are instant.
- `[4]` The matching bar chart.

---

# notebooks_parcel

Pooja's cohort, 161 subjects, already source-reconstructed into brain regions. There are
no sensors, so this folder fabricates the position and orientation BrainOmni needs out of
the atlas itself. That was Sri's suggestion and it is the main methodological piece.

## 00_parcel_geometry.ipynb

- `[0]` Heading.
- `[1]` Imports, and the switch that decides whether to keep subcortical regions.
- `[2]` Reads the 246 region names out of the data file and pulls the region number from each. Asserts they cover 1 to 246, so the geometry cannot end up assigned to the wrong rows.
- `[3]` Loads the Brainnetome atlas and finds the centre point of each region in millimetres.
- `[4]` Builds the geometry array. Drops the 36 deep regions, centres the coordinates, points each region's orientation outward from the brain centre, and rescales so the numbers match what the model saw during pretraining.
- `[5]` Plots the result from three angles and saves the picture. This is the check, because wrong geometry would not crash anything, it would just quietly produce worse numbers. A brain-shaped cloud in all three views means it worked.
- `[6]` Saves the geometry and the row indices every later notebook uses to pick the same 210 regions in the same order.

## 01_parcel_setup.ipynb

Turns raw region signal into tokenizer output. The expensive notebook here.

- `[0]` Heading.
- `[1]` Config. Which model, which region set, and the 30-second segment length the checkpoint requires.
- `[2]` Loads the geometry and converts it to tensors once, since it is the same for every subject.
- `[3]` Loads BrainOmni frozen and defines the function that stops at the tokenizer instead of running the whole model. Stopping there is what makes fine-tuning cheap later.
- `[4]` The preprocessing. Resamples 600 Hz to 256 Hz, normalises each region, and cuts 60 seconds into two 30-second segments. Then streams the file one subject at a time so memory stays flat.
- `[5]` Runs it on the AD file then the control file, printing elapsed time. The slow cell.
- `[6]` Stacks everything and saves. Prints the value range, which is a real check, the parcel data landed at -3.30 to 3.79 against -3.30 to 3.78 for real sensor data.

## 02_parcel_baselines.ipynb

- `[0]` Heading.
- `[1]` Config. Has to match whichever version of notebook 01 or 11 you ran last.
- `[2]` Extracts frozen features the same way the sensor pipeline does, so the two sets of numbers are comparable.
- `[3]` Cross-validated logistic regression, printing per-fold scores as well as the mean. A single bad fold is visible there and invisible in the mean.
- `[4]` Compares the score against just guessing the majority class, and prints a what-to-check list if it is too close. A guard against writing up a null result as a finding.
- `[5]` Section heading.
- `[6]` Compares 210 regions against 52 parcels on the same subjects. The difference is +0.019 accuracy with the interval crossing zero, so matching Sid's parcellation costs nothing measurable.

## 03_combined.ipynb

- `[0]` Heading.
- `[1]` Config and paths for both cohorts.
- `[2]` Loads both and asserts their token shapes match. They do, because the tokenizer always produces 16 latent sources no matter how many input channels there were. 271 sensors and 210 regions both collapse to the same shape, which is the property this whole project leans on.
- `[3]` Extracts features for all 309 subjects together.
- `[4]` Cross-validates, choosing a grouped split if subject overlap is known. It currently assumes the cohorts are disjoint, which is unverified and one of the open questions for Sri.
- `[5]` Learning curve over increasing training set size. The `shuffle=True` here was a bug fix, without it the first 20 percent of the data is all one class.
- `[6]` Scores OSF alone, parcel alone, and the two pooled. Pooling turns out to be worse than parcel alone, which is the domain shift result.

## 04_finetune_combined.ipynb

Repeats the fine-tuning sweep on 309 subjects, to test whether it failed the first time
just because the dataset was small.

- `[0]` Heading.
- `[1]` Config. Model, learning rates, epoch budget, logging.
- `[2]` Loads and pools both cohorts, with an error message that names the likely cause if the shapes disagree.
- `[3]` The dataset wrapper and the classifier.
- `[4]` The training loop, same recipe as the sensor notebooks.
- `[5]` Cross-validation, saving after every fold rather than every configuration. One configuration here is about fifteen hours.
- `[6]` The driver, with resume.
- `[7]` Training curves, one figure per depth.
- `[8]` Prints the combined results next to the sensor-only numbers. The answer is that more data stopped fine-tuning from hurting but did not make it help.

## 05_bootstrap_ci.ipynb

- `[0]` Heading.
- `[1]` Gathers every method into one list of predictions so they can be compared on equal terms.
- `[2]` Confidence intervals for each.
- `[3]` Paired differences against the first method in the list.
- `[4]` Horizontal bar chart with the intervals drawn as error bars. They are asymmetric on purpose, because bootstrap intervals are not symmetric and drawing them that way would misrepresent them.
- `[5]` The full paired comparison between frozen and fine-tuned on the combined cohort, with figures.
- `[6]` The same for the other model size.

## 06_plot_comparisons.ipynb

- `[0]` Heading.
- `[1]` Loads the fine-tuning results and the frozen features.
- `[2]` One bar chart per head, across unfreezing depths, with baselines marked. These are the presentation figures.

## 07_algorithm_sweep.ipynb

- `[0]` Heading.
- `[1]` Setup and features.
- `[2]` Runs the fifteen-classifier sweep on parcel features. Confirms again that the classifier is not the limiting factor.

## 08_pca_curve.ipynb

- `[0]` Heading.
- `[1]` Setup, classifier list, component range.
- `[2]` Sweeps PCA components for each classifier.
- `[3]` The figure, with the sensor-level score drawn in for comparison.

## 09_cohort_balance.ipynb

Written after the combined number came out suspiciously stable. It asks whether that
result is real or an artifact of mixing two datasets.

- `[0]` Heading.
- `[1]` Setup, and tags each subject with which cohort they came from.
- `[2]` Section heading.
- `[3]` Works out how well you could score by ignoring the brain data entirely and just guessing each cohort's majority class. Also tests whether the two cohorts differ in AD rate, since if they did, knowing the cohort would leak the label.
- `[4]` Section heading.
- `[5]` Scores the real classifier separately on each cohort. A big gap would mean the pooled number is averaging over two different problems.
- `[6]` Section heading.
- `[7]` Plots those scores against the guessing ceiling.
- `[8]` Section heading.
- `[9]` Trains a classifier to predict the cohort instead of the diagnosis. It gets AUC 1.000, meaning the two datasets are perfectly distinguishable. That is a severe batch effect and explains why pooling costs accuracy.
- `[10]` Notes on reading the output.
- `[11]` Section heading.
- `[12]` Projects the features to two dimensions and colours by cohort. The two clouds separate cleanly, which is the picture version of that 1.000.

The resolution these cells reach together. The batch effect is real and harmful, but it
cannot be inflating the result, because guessing from cohort alone only reaches 0.524.
Those are two separate claims and this notebook keeps them apart.

## 09_head_comparison.ipynb

- `[0]` Heading.
- `[1]` Setup only.

Never run. Shares a number with `09_cohort_balance` and was superseded by notebook 06.
Safe to delete.

## 10_glasser52_mapping.ipynb

Builds the map from Brainnetome regions to Sid's 52 Glasser parcels. This is the
"down project to match Sid" instruction.

- `[0]` Heading.
- `[1]` Loads both atlases. They are on different grids, 2mm against 8mm.
- `[2]` For each region, counts how many of its voxels fall inside each Glasser parcel. Goes through millimetre coordinates because the two voxel grids do not line up, so millimetres are the only common language.
- `[3]` Normalises those counts into weights and saves them. Prints how many regions feed each parcel and vice versa, which is the check that the map is sane rather than degenerate.
- `[4]` Builds the 52-parcel geometry the same way notebook 00 did for 210.
- `[5]` Plots the 52 centroids from three angles.
- `[6]` Section heading.
- `[7]` The sign-flipping diagnosis. Correlates region pairs inside the same parcel. Only 11 percent of them keep a consistent sign across subjects, and 41 percent are negative. Neighbouring cortex should be positively correlated, so strong correlations with random signs mean the signs are arbitrary rather than real. Pooja's pipeline skipped the step that fixes this.
- `[8]` Section heading.
- `[9]` The workaround. Merges regions into parcels using the first principal component instead of the average. Flipping an input's sign leaves the component unchanged, so arbitrary signs stop mattering. Averaging cancelled signal badly, one parcel kept only 23 percent of its amplitude.

## 11_glasser52_setup.ipynb

Notebook 01 again, with the merge step inserted.

- `[0]` Heading.
- `[1]` Config, now pointed at the 52-parcel version.
- `[2]` Loads the geometry and the mapping weights, asserting their shapes. Catches running this before notebook 10 finished.
- `[3]` Loads the model and the tokenizer-only function.
- `[4]` The principal-component merge from notebook 10.
- `[5]` The same preprocessing with one extra step. 246 rows down to 210, merged to 52, then resampled and normalised. The merge goes first because normalising per channel beforehand would wipe out the relative amplitudes the merge depends on.
- `[6]` Runs it.
- `[7]` Saves, then compares the value range against the 210-region version. Two different channel counts through the same model should land in similar ranges, and a big gap would mean the merge broke the scale.

---

# notebooks_mamba

A third model, added after Chet Gohil emailed Sri about it. It wants 100 Schaefer parcels
at 250 Hz rather than 52 Glasser parcels at 256 Hz, so the mapping work gets redone.

## 00_schaefer100_mapping.ipynb

- `[0]` Heading.
- `[1]` Loads the Brainnetome and Schaefer atlases.
- `[2]` The same voxel overlap counting, now to 100 parcels.
- `[3]` Section heading.
- `[4]` The merge function, with a guard for empty parcels. Some Schaefer parcels get no Brainnetome voxels at all, which is expected when the boundaries disagree, and without the guard it would crash.
- `[5]` Plots the 100 centroids.

## 01_feasibility.ipynb

Written to answer one question before committing days to this. MEG-Mamba's tokenizer was
trained on properly beamformed Schaefer data. Ours is Brainnetome regions remapped into
Schaefer parcels, which is not the same thing. Does the tokenizer still see structure or
does it just produce noise.

- `[0]` Heading.
- `[1]` Setup. Note the target is 250 Hz here, not the 256 BrainOmni uses.
- `[2]` The merge, plus a helper that prepares one subject end to end.
- `[3]` Loads the tokenizer, with a clear error and a clone command if the package is missing.
- `[4]` Tokenizes a few subjects and measures how many distinct tokens get used and how spread out they are.
- `[5]` Notes on reading those numbers.
- `[6]` The control that makes the whole check mean something. Tokenizes pure random noise and measures the same things. High variety on real data only proves anything if noise looks different, and it did. Real data used fewer tokens, less randomly.

## 02_tokenize_all.ipynb

- `[0]` Heading.
- `[1]` Setup.
- `[2]` The merge, 210 regions down to 100 parcels.
- `[3]` Tokenizes everyone. Rewritten twice for speed. The files are stored in a layout where reading one subject costs the same as reading 25, so it reads 25 at a time. Reading all of them at once was worse, it pushed the machine into swap and throughput collapsed. It also forces the GPU, because the tokenizer quietly defaults to CPU on a Mac.
- `[4]` Builds the session features MEG-Mamba uses to handle an unseen recording without training. They come from token statistics, which is the clever part of its design.
- `[5]` Saves everything and prints file sizes. The first two files get uploaded to Colab.

## colab_extract.ipynb

Runs on Colab, not here. The Mamba kernels need CUDA and a reasonably modern NVIDIA GPU,
and will not build on Apple Silicon at all.

- `[0]` Heading.
- `[1]` The install, which took several attempts. Pinned to a specific commit because the public release does not contain the kernels this model needs.
- `[2]` Loads the uploaded files and the pretrained model.
- `[3]` Notes on what gets extracted.
- `[4]` Runs each subject through in windows, capped at 12 windows for memory, and saves the features. The cap means the last fifth of each recording goes unused, which is a stated limitation.

A Colab T4 is not enough. It compiles the kernels fine and then crashes at runtime. An L4
works, which on Colab means Pro.

## 03_mamba_baselines.ipynb

- `[0]` Heading.
- `[1]` Loads the features that came back from Colab, checking the file is actually there.
- `[2]` Cross-validated logistic regression. 0.826 accuracy, 0.890 AUC.
- `[3]` Rebuilds predictions in the pooled form the rest of the project uses, plus the bootstrap helper.
- `[4]` Section heading.
- `[5]` Compares MEG-Mamba against both BrainOmni versions on the same subjects. Asserts the subject order matches first, because a paired test is meaningless if row 5 is a different person in each file. All three models land between 0.826 and 0.870 with none pulling away.
