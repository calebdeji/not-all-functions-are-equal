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

λa and λb are learned scalars. `sim_in` = similarity to the function's class prototype; `sim_out` = similarity to other classes.

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
├── src/                        # All source code
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
python src/download_apks.py --api_key YOUR_ANDROZOO_KEY
python src/download_apks.py --api_key YOUR_ANDROZOO_KEY --year 2019   # single year
python src/download_apks.py --api_key YOUR_ANDROZOO_KEY --workers 4   # parallel
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
python src/apk_to_higraph.py

# HiGraph (specific years)
python src/apk_to_higraph.py --years 2019 2020 2021

# MalDroid / KronoDroid
python src/apk_to_higraph.py --dataset maldroid
python src/apk_to_higraph.py --dataset kronodroid
```

Output:
- HiGraph: `dataset/higraph/graph/raw_data/higraph_base_data/{year}/{sha256}.json`
- Flat: `dataset/{name}/graph/raw/{sha256}.json`

### 2. Generate Family Splits (required before any training)

**This must be run before training any model.** It computes a disjoint 60/20/20 train/val/test split at the family level and saves it to `dataset_config_split/`. All models (backbone, episodic heads, FAMCF, AndroFIM) read from this file to ensure they train and evaluate on identical partitions.

```bash
python src/generate_splits.py --config trial_config_v3_higraph_no_overlap
python src/generate_splits.py --config trial_config_v3_maldroid_no_overlap
python src/generate_splits.py --config trial_config_v3_kronodroid_no_overlap
```

If this step is skipped, the graph training pipeline will generate the split as a side effect, but AndroFIM and FAMCF will fail with a `FileNotFoundError` since they only read — they do not generate splits.

### 3. Extract FAMCF Features

Extracts permission, API call, and opcode feature vectors from APKs. **Required before running the training pipeline** — `run_training.sh` will fail at the FAMCF stage if this has not been run.

```bash
# HiGraph
python src/extract_famcf_features.py --apk_root raw_format/higraph --out_dir dataset/higraph/famcf

# MalDroid / KronoDroid
python src/extract_famcf_features.py --apk_root raw_format/maldroid --out_dir dataset/maldroid/famcf
```

### 4. Convert APKs to AndroFIM Images

Converts APK bytecode to RGB visualisation images. **Required before running the training pipeline** — `run_training.sh` will fail at the AndroFIM stage if this has not been run.

```bash
# HiGraph
python src/apk_to_androFIM_image.py

# Flat datasets
python src/apk_to_androFIM_image.py --dataset maldroid
python src/apk_to_androFIM_image.py --dataset kronodroid
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
4. **Episodic heads** — N-way K-shot fine-tuning (`self_attention`, `sdl_complexity`, `sdl_degree`)
5. **FAMCF encoders** — triplet contrastive training on permission/API/opcode features
6. **AndroFIM** — DINO self-supervised pre-training → fine-tune on base families
7. **FCGAT** — supervised pre-training of the adapted FCGAT baseline

Model checkpoints are saved to:

```
trained_models/{config_id}/{dataset}/{split_type}/{variant}/model.pth
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

- Episodic heads: `self_attention`, `sdl_complexity`, `sdl_degree` (and ablations)
- Pretrained backbones: `whole_graph_backbone`, `pretrained_backbone`
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
python src/episodic_evaluate.py \
    --config trial_config_v3_higraph_no_overlap \
    --n_shots 1 5 \
    --n_episodes 500
```

---

## Experiment Configs

Configs are defined in `src/experiment_configs.py`. The main configs are:

| Config                                  | Dataset             | Split             |
| --------------------------------------- | ------------------- | ----------------- |
| `trial_config_v3_higraph_no_overlap`    | HiGraph (2012–2021) | Disjoint families |
| `trial_config_v3_maldroid_no_overlap`   | MalDroid            | Disjoint families |
| `trial_config_v3_kronodroid_no_overlap` | KronoDroid          | Disjoint families |

All use a 60/20/20 base/val/novel family split with seed 42.

---

## Dependencies

- Python 3.10+
- PyTorch 2.10
- PyTorch Geometric 2.7
- Androguard 4.1.3 (APK parsing)
- See `requirements.txt` for full list
