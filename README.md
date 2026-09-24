# Few-Shot Android Malware Family Classification

A graph-based few-shot learning system for classifying Android malware families from APKs. Given only a handful of labeled examples per family (N-way K-shot), the system learns to classify unseen APKs into their correct malware family.

---

## How It Works

Each APK is parsed into a **HiGraph** — a two-level graph representation:

- **CFG (Control Flow Graph)**: basic blocks as nodes with opcode/feature vectors, connected by control flow edges
- **FCG (Function Call Graph)**: functions as nodes with structural features (cyclomatic complexity, degree profile), connected by call edges

A two-level GNN backbone encodes this structure into per-function embeddings. A pooling head then aggregates function embeddings into a single graph-level embedding, which is used for **prototypical nearest-neighbour classification** at inference time — no retraining needed for new families.

### Pooling Heads (SDL)

The core contribution is **Structure-Discriminative Landmark (SDL)** weighting — functions are weighted by how discriminative they are before pooling:

| Variant          | Formula                                                   |
| ---------------- | --------------------------------------------------------- |
| `self_attention` | Learnable MLP attention                                   |
| `sdl_complexity` | `sigmoid(λa·sim_in + λb·cyclomatic_complexity − sim_out)` |
| `sdl_degree`     | `sigmoid(λa·sim_in + λb·in_degree − sim_out)`             |
| `sdl_pagerank`   | `sigmoid(λa·sim_in + λb·pagerank − sim_out)`              |

λa and λb are learned scalars. `sim_in` = similarity to the function's class prototype; `sim_out` = similarity to other classes. The three SDL variants differ only in the structural prior (`cyclomatic_complexity`, `in_degree`, `pagerank`); PageRank is computed per episode by power iteration over the FCG.

### Graph-pooling baselines

Learned readouts trained on the same pre-trained backbone, for comparison against SDL:

| Variant      | Description                                                              |
| ------------ | ------------------------------------------------------------------------ |
| `gmt`        | Graph Multiset Transformer — `GMPool_G → SelfAtt → GMPool_1`             |
| `gmpool`     | Graph Multiset Pooling only — `GMPool_G → mean over seeds`               |
| `topk_pool`  | TopKPooling — learned node scoring, gated mean readout                    |
| `diff_pool`  | DiffPool — soft cluster assignment, with link-prediction + entropy losses |

---

## Datasets

Three datasets are supported:

| Dataset        | Type               | Samples | Families |
| -------------- | ------------------ | ------- | -------- |
| **HiGraph**    | Yearly (2012–2021) | ~34,000 | 84       |
| **MalDroid**   | Flat               | ~7,676  | 25       |
| **KronoDroid** | Flat               | ~11,500 | 43       |

Ground truth labels live in `dataset_refined_truth/`:

```
dataset_refined_truth/
  higraph/
    labelled_samples_with_family_{year}.csv   # hash, family — one per year
    global_families_mapping.json              # family name → integer id
  maldroid/
    labelled_samples_with_family_maldroid_apks.csv
    family_to_idx.json
    category_to_idx.json
  kronodroid/
    labelled_samples_with_family_kronodroid_apks.csv
    family_to_idx.json
    category_to_idx.json
```

---

## Project Structure

```
.
├── dataset_refined_truth/      # Ground truth labels and family mappings
├── dataset/                    # Processed PyG graphs, AndroFIM images, FAMCF features
│   ├── higraph/graph/          # Cached PyG graph files
│   ├── higraph/androfim/       # AndroFIM PNG images + manifest
│   ├── higraph/famcf/          # FAMCF feature vectors
│   ├── maldroid/...
│   └── kronodroid/...
├── raw_format/                 # Raw APK files (see below)
├── dataset_config_split/       # Train/val/test family splits (generated)
├── trained_models/             # Model checkpoints
├── results/                    # Evaluation logs, figures, result sheets
├── src/                        # All source code, grouped by pipeline stage
│   ├── core/                   # Models, backbone, datasets, config, shared utils
│   ├── data/                   # APK → graph / image / feature extraction, splits
│   ├── pretrain/               # Contrastive backbone pre-training, DINO
│   ├── train/                  # Episodic heads and baseline training
│   ├── eval/                   # Unified few-shot evaluation
│   ├── robustness/             # Open-set and obfuscation evaluation
│   └── analysis/               # Interpretability, weight stability, result sheets
└── scripts/                    # Training and evaluation shell scripts
```

---

## Organizing Raw APKs (`raw_format/`)

APKs must be placed in `raw_format/` before running any preprocessing. The expected layout depends on the dataset:

### HiGraph (yearly structure)

```
raw_format/
  higraph/
    2012/
      <sha256>.apk
      ...
    2013/
      <sha256>.apk
    ...
    2021/
      <sha256>.apk
```

HiGraph APKs can be downloaded from [AndroZoo](https://androzoo.uni.lu) using the provided script:

```bash
python -m src.data.download_apks --api_key YOUR_ANDROZOO_KEY
python -m src.data.download_apks --api_key YOUR_ANDROZOO_KEY --year 2019   # single year
python -m src.data.download_apks --api_key YOUR_ANDROZOO_KEY --workers 4   # parallel
```

### Flat datasets (MalDroid, KronoDroid)

MalDroid and KronoDroid APKs must be obtained from their respective sources:

- **MalDroid**: [https://www.unb.ca/cic/datasets/maldroid-2020.html](https://www.unb.ca/cic/datasets/maldroid-2020.html)
- **KronoDroid**: [https://github.com/aleguma/kronodroid](https://github.com/aleguma/kronodroid)

We use a subset of each dataset. The exact samples used are listed in:
- `dataset_refined_truth/maldroid/labelled_samples_with_family_maldroid_apks.csv`
- `dataset_refined_truth/kronodroid/labelled_samples_with_family_kronodroid_apks.csv`

After obtaining the full datasets, filter to only the APKs whose SHA256 hash appears in the corresponding CSV and place them in `raw_format/`:

```
raw_format/
  maldroid/
    <sha256>.apk
    ...
  kronodroid/
    <sha256>.apk
    ...
```

No year subdirectories — all APKs sit directly in the dataset folder.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Step-by-Step Pipeline

### 1. Convert APKs to HiGraphs

Parses each APK into a JSON graph file using Androguard:

```bash
# HiGraph (all years)
python -m src.data.apk_to_higraph

# HiGraph (specific years)
python -m src.data.apk_to_higraph --years 2019 2020 2021

# MalDroid / KronoDroid
python -m src.data.apk_to_higraph --dataset maldroid
python -m src.data.apk_to_higraph --dataset kronodroid
```

Output:
- HiGraph: `dataset/higraph/graph/{year}/raw/{sha256}.json`
- Flat: `dataset/{name}/graph/raw/{sha256}.json`

### 2. Generate Family Splits (required before any training)

**This must be run before training any model.** It computes a disjoint 60/20/20 train/val/test split at the family level and saves it to `dataset_config_split/`. All models (backbone, episodic heads, FAMCF, AndroFIM) read from this file to ensure they train and evaluate on identical partitions.

```bash
python -m src.data.generate_splits --config trial_config_v3_higraph_no_overlap
python -m src.data.generate_splits --config trial_config_v3_maldroid_no_overlap
python -m src.data.generate_splits --config trial_config_v3_kronodroid_no_overlap
```

If this step is skipped, the graph training pipeline will generate the split as a side effect, but AndroFIM and FAMCF will fail with a `FileNotFoundError` since they only read — they do not generate splits.

### 3. Extract FAMCF Features

Extracts permission, API call, and opcode feature vectors from APKs. **Required before running the training pipeline** — `run_training.sh` will fail at the FAMCF stage if this has not been run.

```bash
# HiGraph
python -m src.data.extract_famcf_features --apk_root raw_format/higraph --out_dir dataset/higraph/famcf

# MalDroid / KronoDroid
python -m src.data.extract_famcf_features --apk_root raw_format/maldroid --out_dir dataset/maldroid/famcf
```

### 4. Convert APKs to AndroFIM Images

Converts APK bytecode to RGB visualisation images. **Required before running the training pipeline** — `run_training.sh` will fail at the AndroFIM stage if this has not been run.

```bash
# HiGraph
python -m src.data.apk_to_androFIM_image

# Flat datasets
python -m src.data.apk_to_androFIM_image --dataset maldroid
python -m src.data.apk_to_androFIM_image --dataset kronodroid
```

Output layout:

- HiGraph: `dataset/higraph/androfim/{year}/{sha256}.png`
- Flat: `dataset/{name}/androfim/images/{sha256}.png`

### 5. Train

Once steps 1–4 are complete, run the full training pipeline:

```bash
bash scripts/run_training.sh          # full pipeline
bash scripts/run_training.sh splits   # step 0: generate splits only
bash scripts/run_training.sh full_backbone
bash scripts/run_training.sh episodic
bash scripts/run_training.sh famcf
bash scripts/run_training.sh androfim
bash scripts/run_training.sh fcgat
```

Training stages in order:

1. **Generate splits** — disjoint family train/val/test split
2. **Whole-graph backbone** — instance-level NT-Xent contrastive pre-training
3. **Full backbone** — instance + set-level NT-Xent contrastive pre-training
4. **Episodic heads** — N-way K-shot fine-tuning (`self_attention`, `sdl_complexity`, `sdl_degree`, `sdl_pagerank`)
5. **FAMCF encoders** — triplet contrastive training on permission/API/opcode features
6. **AndroFIM** — DINO self-supervised pre-training → fine-tune on base families
7. **FCGAT** — supervised pre-training of the adapted FCGAT baseline

Model checkpoints are saved to:

```
trained_models/{config_id}/{dataset}/{split_type}/{variant}/model.pth
```

To train **everything** in dependency order — all backbones, all baselines, all episodic
heads and the granular-ablation ladder — use the master script instead:

```bash
bash scripts/run_full_training.sh          # full pipeline
bash scripts/run_full_training.sh reset    # clear progress markers and start over
```

`run_full_training.sh`, and the other long-running scripts, record completed steps in a
`*_progress.txt` file at the repository root, so an interrupted run resumes where it
stopped rather than retraining from scratch.

The graph-pooling baselines and the granular ablation have their own scripts:

```bash
bash scripts/run_graph_pooling_training.sh      # gmt, gmpool, topk_pool, diff_pool
bash scripts/run_rebuttal_ablation_training.sh  # flat-FCG / set-only / instance-only ladder
```

---

## Evaluation

`episodic_evaluate.py` runs a unified few-shot evaluation across all trained models on **identical episodes** for a fair comparison.

### What it does

For each episode it:

1. Samples N classes, K support examples per class, and Q query examples per class
2. **Support phase** (labels known): computes per-function weights → builds node prototypes and graph prototypes
3. **Query phase** (labels unknown): computes weights using node prototypes as pseudo-labels → pools to graph level
4. Classifies each query by cosine similarity to the nearest graph prototype

All models are evaluated on the same support/query split per episode.

### Models evaluated

- Episodic heads: `self_attention`, `sdl_complexity`, `sdl_degree`, `sdl_pagerank` (and SDL ablations)
- Graph-pooling baselines: `gmt`, `gmpool`, `topk_pool`, `diff_pool`
- Pretrained backbones: `whole_graph_backbone`, `pretrained_backbone`
- Granular-ablation backbones and heads: `flat_fcg`, `set_only`, `instance_only`, `random_init`
- `FCGAT` baseline
- `AndroFIM` baseline (image-based ResNet-12, multi-crop ProtoNet)
- `FAMCF` baseline (soft-voting across permission/API/opcode encoders)

### Output

- Mean accuracy ± 95% CI per model per K-shot setting
- Per-family accuracy breakdown
- Parameter counts, wall time, GPU time, peak memory (resource report)
- Optional per-function weight records for analysis (`--export_weights_dir`)

```bash
bash scripts/run_evaluation.sh

# Or directly:
python -m src.eval.episodic_evaluate \
    --config trial_config_v3_higraph_no_overlap \
    --n_shots 1 5 \
    --n_episodes 500
```

Three model selections are available:

| Flag              | Models evaluated                                                  |
| ----------------- | ----------------------------------------------------------------- |
| *(none)*          | Main comparison: SDL variants, self-attention, pooling baselines   |
| `--ablation_only` | SDL component ablations only (no FCGAT / FAMCF / AndroFIM)         |
| `--all_models`    | Everything, incl. granular-ablation backbones and heads            |

Use `--all_models` when results must be directly comparable — it evaluates every model on the
*same* episodes in a single run, avoiding the sampling differences that arise across separate
runs:

```bash
bash scripts/run_full_evaluation.sh
```

Logs are timestamped (`results/…/evaluate_all_models_{timestamp}.log`) and a running
mean ± CI is printed every 20 episodes, so a long run can be monitored while in progress.

---

## Temporal Evaluation

Trains on pre-2019 data and evaluates on 2019–2021, measuring generalization to families that
emerge *after* the training period. The test pool is restricted to families that do not occur
anywhere in 2012–2018, so this setting is temporally separated **and** family-disjoint: it is
strictly stricter than `no_overlap`, which is family-disjoint but has no temporal constraint.

```bash
python -m src.data.generate_splits --config trial_config_v3_higraph_temporal
bash scripts/run_temporal_training.sh      # backbone + all heads + baselines
bash scripts/run_temporal_evaluation.sh    # per-year (2019, 2020, 2021) + combined
```

The split file records the exact train/val sample hashes, plus the indices and names of the
unseen test families, in `dataset_config_split/higraph/trial_config_v3/temporal_split.json`.
FAMCF and AndroFIM filter on those hashes so every model sees the same samples.

`run_temporal_evaluation.sh` reports a combined 2019–2021 result and a per-year breakdown.
A year is skipped when it contains fewer unseen families than `--n_way`, which is expected:
most families appearing in a given year were already present before 2019, so only the
combined pool is reliably large enough for 5-way episodes.

---

## Open-Set Evaluation

Measures whether a model can tell a known family from one it has never seen. Each episode adds
*intruder* queries drawn from families outside the episode's support set, and the decision
threshold is calibrated on the validation split to accept 95% of known queries.

```bash
python -m src.robustness.episodic_openset_evaluate \
    --config trial_config_v3_higraph_no_overlap \
    --n_shots 1 5 --n_intruder 15 --n_episodes 500 --n_cal 200
```

Reports, per model and shot count: closed-set accuracy on known queries, AUROC for
known-vs-intruder, FPR at 95% TPR, and open-set macro-F1 at the calibrated threshold.

---

## Obfuscation Robustness

Evaluates degradation under three structural transformations applied with
[Obfuscapk](https://github.com/ClaudiuGeorgiu/Obfuscapk) — `ArithmeticBranch` (dead-code
insertion), `CallIndirection` (call-graph pollution), and `Reorder` (basic-block reordering).
Obfuscapk is an external tool and must be installed separately.

Each obfuscated APK is paired with its clean counterpart and evaluated under identical
episodes, so the reported Δ isolates the effect of the transformation rather than of
episode sampling.

```bash
# 1. stage the novel-family APKs that still need obfuscating
python -m src.data.prepare_novel_apks

# 2. run Obfuscapk over the staged APKs (all years, or a year range)
bash scripts/obfuscate_higraph.sh
bash scripts/obfuscate_higraph.sh 2019 2021

# 3. collect the newly obfuscated APKs into a novel-only folder per method
python -m src.data.prepare_novel_apks --collect_only

# 4. extract graphs, then FAMCF features and AndroFIM images, for each method
bash scripts/run_obfuscation_extraction.sh
bash scripts/run_obfuscation_famcf_androfim_extraction.sh

# 5. evaluate
python -m src.robustness.episodic_obfuscated_evaluate --config trial_config_v3_higraph_no_overlap
python -m src.robustness.episodic_obfuscated_query_evaluate --config trial_config_v3_higraph_no_overlap
```

Step 4 encodes FAMCF features with the **clean** vocabulary (`--existing_vocab`) so that
feature positions still line up with the trained encoders.

The two evaluators differ in threat model:

- `episodic_obfuscated_evaluate` — both support and query obfuscated (homogeneous).
- `episodic_obfuscated_query_evaluate` — clean support, obfuscated query. This matches an
  evasion setting, where a defender holds a curated clean reference set and the incoming
  sample is obfuscated.

---

## Interpretability and Weight Stability

Inspects which functions SDL actually up-weights, and how stable those weights are under
obfuscation.

```bash
# pick consistently-correct and consistently-misclassified APKs, saving a replay episode
python -m src.analysis.select_interpretability_apks

# dump per-function weight decomposition for each selected APK
python -m src.analysis.extract_sdl_weights

# figures
python -m src.analysis.plot_sdl_weights
python -m src.analysis.plot_sdl_bars_simple

# weight stability on matched functions, clean vs obfuscated
python -m src.analysis.measure_weight_stability
```

`extract_sdl_weights` reports, per function:
`weight | sim_max | sim_others | norm_complexity | raw_complexity | function_name`,
so the contribution of the semantic and structural terms can be read off individually.

Two further utilities:

```bash
python -m src.analysis.generate_result_sheets      # Excel summaries of the evaluation logs
python -m src.analysis.count_placeholder_apks      # functions with no recoverable CFG
```

---

## Experiment Configs

Configs are defined in `src/core/experiment_configs.py`. The main configs are:

| Config                                  | Dataset             | Split             |
| --------------------------------------- | ------------------- | ----------------- |
| `trial_config_v3_higraph_no_overlap`    | HiGraph (2012–2021) | Disjoint families |
| `trial_config_v3_maldroid_no_overlap`   | MalDroid            | Disjoint families |
| `trial_config_v3_kronodroid_no_overlap` | KronoDroid          | Disjoint families |
| `trial_config_v3_higraph_temporal`      | HiGraph (2012–2021) | Temporal          |

The `no_overlap` configs use a 60/20/20 base/val/novel family split with seed 42, so evaluation
families are never seen in training. The `temporal` config adds a time constraint on top: it
trains on an 80/20 sample-level split of 2012–2018 and evaluates only on families from
2019–2021 that never appear in 2012–2018.

---

## Reproducing the Reported Results

Each result table maps to one command, assuming the corresponding training step has completed.

| Result                                   | Command                                                        |
| ---------------------------------------- | -------------------------------------------------------------- |
| Main comparison across all three corpora | `bash scripts/run_evaluation.sh`                                 |
| Additional pooling comparisons           | `bash scripts/run_full_evaluation.sh`                            |
| Temporal generalization (unseen families)| `bash scripts/run_temporal_evaluation.sh`                        |
| SDL structural-prior ablations           | `bash scripts/run_ablation_evaluation.sh`                        |
| Pretraining component ablation           | `bash scripts/run_full_evaluation.sh`                            |
| Open-set evaluation                      | `python -m src.robustness.episodic_openset_evaluate`             |
| SDL weight decomposition                 | `python -m src.analysis.extract_sdl_weights`                     |
| Per-episode inference cost               | any evaluation run — see `resource_report*.txt` in the log dir   |
| Paired clean vs. obfuscated accuracy     | `python -m src.robustness.episodic_obfuscated_evaluate`          |
| Weight stability under obfuscation       | `python -m src.analysis.measure_weight_stability`                |

The three reviewer-response experiments — weight stability, the random-init control, and the
clean-support/obfuscated-query evaluation — can be run together:

```bash
bash scripts/run_reviewer_experiments.sh
```

---

## Dependencies

- Python 3.13 (see `.python-version`)
- PyTorch 2.10 · PyTorch Geometric 2.7 · torchvision 0.25
- Androguard 4.1.3 (APK parsing)
- scikit-learn (open-set AUROC), matplotlib (figures), openpyxl (result sheets)
- timm (required by the AndroFIM DINO pre-training entrypoint)
- See `requirements.txt` for the full pinned list

All entrypoints are run as modules from the repository root, e.g.
`python -m src.train.episodic_train --config …`.
