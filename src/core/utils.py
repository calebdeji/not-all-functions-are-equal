import os
import json
import random
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import  WeightedRandomSampler
from enum import Enum

class DatasetType(Enum):
    NoOverlap = 'no_overlap'
    Temporal = 'temporal'




@dataclass
class ModelParameters:
    id: str
    cfg_dim:int
    func_feat_dim: int
    cfg_hidden_channels: int
    fcg_hidden_channels: int
    final_channels: int
    dropout: float
    backbone_batch_size: int
    backbone_epochs : int
    backbone_lr : float
    backbone_projection_head_hidden_channels: int
    backbone_projection_head_output_channels: int
    backbone_eta_lr_min:float
    backbone_set_level_features_top_k : int
    backbone_pretrain_instance_level_projection_temperature : float
    backbone_pretrain_set_level_projection_temperature : float
    training_years : list[str]
    test_years : list[str]

    dataset_split_type: DatasetType = DatasetType.NoOverlap
    dataset_name: str = 'higraph'



    # Episodic fine-tuning
    episodic_n_way: int = 5
    episodic_k_shot: int = 5
    episodic_n_query: int = 15
    episodic_episodes_per_epoch: int = 200
    episodic_epochs: int = 100
    episodic_lr: float = 0.0001
    episodic_proto_scale: float = 16.0
    episodic_supcon_temp: float = 0.1
    episodic_lambda_sup: float = 0.5
    episodic_eval_every: int = 5
    episodic_eval_episodes: int = 200
    episodic_patience: int = 6


_FLAT_DATASET_NAMES = {'maldroid', 'kronodroid'}


class Utils:

    @staticmethod
    def get_project_root():
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    @staticmethod
    def get_graph_root(dataset: str) -> str:
        """
        Returns the PyG dataset root for the given dataset name.
        Edit this function if the dataset layout changes.

        dataset: 'higraph' | 'maldroid' | 'kronodroid'
        """
        return os.path.join(Utils.get_project_root(), 'dataset', dataset, 'graph')

    @staticmethod
    def get_androfim_root(dataset: str) -> str:
        """
        Returns the AndroFIM image directory for the given dataset name.
        Edit this function if the dataset layout changes.

        dataset: 'higraph' | 'maldroid'
        """
        return os.path.join(Utils.get_project_root(), 'dataset', dataset, 'androfim')

    @staticmethod
    def get_androfim_manifest(dataset: str) -> str:
        """
        Returns the path to androFIM_manifest.csv for the given dataset name.

        dataset: 'higraph' | 'maldroid'
        """
        return os.path.join(Utils.get_androfim_root(dataset), 'androFIM_manifest.csv')

    @staticmethod
    def get_famcf_root(dataset: str) -> str:
        """
        Returns the FAMCF features directory for the given dataset name.
        Edit this function if the dataset layout changes.

        dataset: 'higraph' | 'maldroid'
        """
        return os.path.join(Utils.get_project_root(), 'dataset', dataset, 'famcf')

    @staticmethod
    def get_refined_truth_root(dataset: str) -> str:
        """
        Returns the dataset_refined_truth directory for the given dataset name.
        Edit this function if the dataset layout changes.

        dataset: 'higraph' | 'maldroid' | 'kronodroid'
        """
        return os.path.join(Utils.get_project_root(), 'dataset_refined_truth', dataset)

    @staticmethod
    def get_raw_apk_root(dataset: str) -> str:
        """
        Returns the raw APK directory for the given dataset name.
        Edit this function if the dataset layout changes.

        dataset: 'higraph' | 'maldroid'
        """
        return os.path.join(Utils.get_project_root(), 'raw_format', dataset)

    @staticmethod
    def get_device():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        return device

    @staticmethod
    def get_balanced_sampler(dataset):
        print('[*] Generating balanced sampler')

        from torch.utils.data import Subset
        if isinstance(dataset, Subset):
            all_labels = torch.cat([d.get_labels() for d in dataset.dataset.datasets])
            all_labels = all_labels[torch.tensor(dataset.indices)]
        else:
            all_labels = torch.cat([d.get_labels() for d in dataset])


        unique_families = sorted(set(all_labels.tolist()))
        print(f"[*] Number of unique families {len(unique_families)} and dataset {dataset}")

        y_all = all_labels

        class_counts = torch.bincount(y_all)

        class_counts = class_counts.float()

        # I don't think there should be any class with class_counts 0
        class_counts[class_counts == 0] = 1.0

        class_weights = 1.0 / class_counts


        sample_weights = class_weights[y_all]

        # 7. Create Sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        return sampler, all_labels, len(unique_families)

    @staticmethod
    def augment_all_graph_data(data, edge_perturb=0.2, node_perturb=0.2):
        """
        Applies BOTH edge dropping and feature masking to make the task harder
        and prevent the model from learning "augmentation shortcuts".
        """
        data_aug = data.clone()

        # 1. Edge Perturbation (Drop Edges) — FCG and CFG
        if data_aug.fcg_edge_index.numel() > 0:
            mask = torch.rand(data_aug.fcg_edge_index.size(1), device=data_aug.fcg_edge_index.device) > edge_perturb
            data_aug.fcg_edge_index = data_aug.fcg_edge_index[:, mask]

        if data_aug.edge_index.numel() > 0:
            mask = torch.rand(data_aug.edge_index.size(1), device=data_aug.edge_index.device) > edge_perturb
            data_aug.edge_index = data_aug.edge_index[:, mask]

        # 2. CFG Block Feature Masking (per feature, not per node)
        if data_aug.x.numel() > 0:
            mask_x = (torch.rand_like(data_aug.x) > node_perturb).float()
            data_aug.x = data_aug.x * mask_x

        # 3. Function Feature Masking (per feature)
        if data_aug.function_features.numel() > 0:
            mask_ff = (torch.rand_like(data_aug.function_features) > node_perturb).float()
            data_aug.function_features = data_aug.function_features * mask_ff

        return data_aug

    @staticmethod
    def augment_graph(data, edge_perturb=0.2, node_perturb=0.2):
        """
        Applies BOTH edge dropping and feature masking to make the task harder
        and prevent the model from learning "augmentation shortcuts".
        """
        data_aug = data.clone()

        # 1. Edge Perturbation (Drop Edges) — FCG and CFG
        if data_aug.fcg_edge_index.numel() > 0:
            mask = torch.rand(data_aug.fcg_edge_index.size(1), device=data_aug.fcg_edge_index.device) > edge_perturb
            data_aug.fcg_edge_index = data_aug.fcg_edge_index[:, mask]

        if data_aug.edge_index.numel() > 0:
            mask = torch.rand(data_aug.edge_index.size(1), device=data_aug.edge_index.device) > edge_perturb
            data_aug.edge_index = data_aug.edge_index[:, mask]

        # 2. CFG Block Feature Masking (per feature, not per node)
        if data_aug.x.numel() > 0:
            mask_x = (torch.rand_like(data_aug.x) > node_perturb).float()
            data_aug.x = data_aug.x * mask_x

        # 3. Function Feature Masking (per feature)
        # if data_aug.function_features.numel() > 0:
        #     mask_ff = (torch.rand_like(data_aug.function_features) > node_perturb).float()
        #     data_aug.function_features = data_aug.function_features * mask_ff

        return data_aug

    @staticmethod
    def nt_xent_loss(z1, z2, temperature=0.5):
        from torch.nn import functional
        # z1, z2: [Batch_Size, Dim]
        # batch_size = z1.size(0)

        # out = torch.cat([z1, z2], dim=0)
        #
        # out = torch.nn.functional.normalize(out, dim=1)
        #
        # sim_matrix = torch.mm(out, out.t()) / temperature
        #
        # mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
        # sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
        #
        # labels = torch.cat([
        #     torch.arange(batch_size, device=z1.device) + batch_size,
        #     torch.arange(batch_size, device=z1.device)
        # ], dim=0)
        #
        # loss = torch.nn.functional.cross_entropy(sim_matrix, labels)

        z1 = functional.normalize(z1, dim=1)
        z2 = functional.normalize(z2, dim=1)
        n = z1.size(0)
        sim = torch.mm(z1, z2.t()) / temperature
        labels = torch.arange(n, device=z1.device)
        loss = (functional.cross_entropy(sim, labels) + functional.cross_entropy(sim.t(), labels)) / 2

        return loss

    @staticmethod
    def memory_intense_nt_xent_loss(z1, z2, temperature=0.5):
        from torch.nn import functional

        # 1. L2 Normalize
        z1 = functional.normalize(z1, dim=1)
        z2 = functional.normalize(z2, dim=1)

        N = z1.size(0)
        device = z1.device

        # 2. Concatenate views to form the 2N pool
        z = torch.cat([z1, z2], dim=0)  # Shape: (2N, D)

        # 3. Compute 2N x 2N similarity matrix
        sim = torch.mm(z, z.t()) / temperature

        # 4. Mask out the self-similarity on the diagonal
        sim.fill_diagonal_(-1e9)

        # 5. Create labels for positive pairs
        # The positive pair for index `i` is `i + N`
        # The positive pair for index `i + N` is `i`
        labels = torch.cat([
            torch.arange(N, 2 * N, device=device),
            torch.arange(0, N, device=device)
        ], dim=0)

        # 6. Calculate cross-entropy loss
        loss = functional.cross_entropy(sim, labels)

        return loss

    @staticmethod
    def extract_training_validation_and_test(years: list[str], split_file=None):
        from src.core.parse_dataset import HiGraphDataset, HiGraphData, get_dataset_root
        from torch.utils.data import ConcatDataset, Subset

        year_datasets = [HiGraphDataset(root=get_dataset_root(), year=year) for year in years]
        combined = ConcatDataset(year_datasets)

        all_labels = torch.cat([d.get_labels() for d in year_datasets])

        # Split at the family level to guarantee no class overlap across splits
        unique_families = sorted(set(all_labels.tolist()))
        num_families = len(unique_families)

        family_perm = torch.randperm(num_families, generator=torch.Generator().manual_seed(42)).tolist()
        shuffled_families = [unique_families[i] for i in family_perm]

        train_end = int(0.6 * num_families)
        val_end = train_end + int(0.2 * num_families)

        train_families = set(shuffled_families[:train_end])
        val_families = set(shuffled_families[train_end:val_end])
        test_families = set(shuffled_families[val_end:])

        # Save family-name split to JSON so baselines (AndroFIM, FAMCF) can load
        # the exact same partition without re-deriving it from the graph data.
        if split_file is not None:
            split_file = Path(split_file)
            if not split_file.exists():
                family_to_idx = HiGraphDataset.get_family_mapping()
                idx_to_family = {v: k for k, v in family_to_idx.items()}
                split = {
                    "base":  [idx_to_family[i] for i in shuffled_families[:train_end]],
                    "val":   [idx_to_family[i] for i in shuffled_families[train_end:val_end]],
                    "novel": [idx_to_family[i] for i in shuffled_families[val_end:]],
                }
                split_file.parent.mkdir(parents=True, exist_ok=True)
                with open(split_file, "w") as f:
                    json.dump(split, f, indent=2)
                print(f"[Split] Saved -> {split_file}  "
                      f"(base={len(split['base'])}, val={len(split['val'])}, novel={len(split['novel'])})")

        labels_list = all_labels.tolist()
        train_indices = [i for i, y in enumerate(labels_list) if y in train_families]
        val_indices   = [i for i, y in enumerate(labels_list) if y in val_families]
        test_indices  = [i for i, y in enumerate(labels_list) if y in test_families]

        train_ds = Subset(combined, train_indices)
        val_ds   = Subset(combined, val_indices)
        test_ds  = Subset(combined, test_indices)

        return train_ds, val_ds, test_ds

    @staticmethod
    def extract_temporal_split(training_years: list, test_years: list, split_file=None):
        """
        Temporal evaluation split:
          train / val  — 80/20 sample-level split from training_years, all families present in both.
          test         — samples from test_years belonging ONLY to families that do not occur
                         anywhere in the training period, so temporal generalization is measured
                         on families the model has never seen.

        A family counts as "seen" if it appears anywhere in training_years, not merely in the
        80% train portion; the 20% validation slice is drawn from the same period and so carries
        no additional families.
        """
        from src.core.parse_dataset import HiGraphDataset, get_dataset_root
        from torch.utils.data import ConcatDataset, Subset

        train_year_datasets = [HiGraphDataset(root=get_dataset_root(), year=y) for y in training_years]
        train_combined = ConcatDataset(train_year_datasets)
        n_train_total = sum(len(d) for d in train_year_datasets)

        perm = torch.randperm(n_train_total, generator=torch.Generator().manual_seed(42)).tolist()
        cut = int(0.8 * n_train_total)
        train_ds = Subset(train_combined, perm[:cut])
        val_ds   = Subset(train_combined, perm[cut:])

        # Build flat hash list from raw_file_names (avoids loading full graph data)
        all_hashes = []
        for ds in train_year_datasets:
            for fname in ds.raw_file_names:
                all_hashes.append(fname[:-5])  # strip '.json'
        train_hashes = [all_hashes[i] for i in perm[:cut]]
        val_hashes   = [all_hashes[i] for i in perm[cut:]]

        # Families observed anywhere in the training period
        seen_families = set()
        for ds in train_year_datasets:
            seen_families.update(ds.get_labels().tolist())

        # Test pool: only families absent from the training period
        test_year_datasets = [HiGraphDataset(root=get_dataset_root(), year=y) for y in test_years]
        test_combined = ConcatDataset(test_year_datasets)
        test_labels = torch.cat([ds.get_labels() for ds in test_year_datasets])
        novel_indices = [i for i, lbl in enumerate(test_labels.tolist()) if lbl not in seen_families]
        novel_families = {test_labels[i].item() for i in novel_indices}
        test_ds = Subset(test_combined, novel_indices)

        print(f"[Split] Temporal test pool: {len(novel_indices)} samples from "
              f"{len(novel_families)} families absent from {training_years[0]}-{training_years[-1]} "
              f"(dropped {len(test_labels) - len(novel_indices)} samples of seen families)")

        if split_file is not None:
            split_file = Path(split_file)
            if not split_file.exists():
                split_file.parent.mkdir(parents=True, exist_ok=True)
                with open(split_file, 'w') as f:
                    json.dump({
                        'training_years': training_years,
                        'test_years': test_years,
                        'train_indices': perm[:cut],
                        'val_indices': perm[cut:],
                        'train_hashes': train_hashes,
                        'val_hashes': val_hashes,
                        'test_novel_indices': novel_indices,
                        'test_novel_families': sorted(novel_families),
                    }, f, indent=2)
                print(f"[Split] Saved -> {split_file}  "
                      f"(train={cut}, val={n_train_total - cut}, "
                      f"test={len(novel_indices)} samples from {len(novel_families)} "
                      f"unseen families in {test_years})")

        return train_ds, val_ds, test_ds

    @staticmethod
    def extract_stratified_training_validation_and_test(years: list[str]):
        from src.core.parse_dataset import HiGraphDataset, HiGraphData, get_dataset_root
        from torch.utils.data import ConcatDataset, Subset
        from collections import defaultdict

        year_datasets = [HiGraphDataset(root=get_dataset_root(), year=year) for year in years]
        combined = ConcatDataset(year_datasets)

        all_labels = torch.cat([d.get_labels() for d in year_datasets]).tolist()

        # Group indices by family
        label_to_indices = defaultdict(list)
        for idx, label in enumerate(all_labels):
            label_to_indices[label].append(idx)

        train_indices, val_indices, test_indices = [], [], []
        generator = torch.Generator().manual_seed(42)

        for label, indices in label_to_indices.items():
            n_samples = len(indices)

            perm = torch.randperm(n_samples, generator=generator).tolist()
            shuffled_indices = [indices[i] for i in perm]

            # 60 / 20 / 20 per family
            train_end = int(0.6 * n_samples)
            val_end = train_end + int(0.2 * n_samples)

            train_indices.extend(shuffled_indices[:train_end])
            val_indices.extend(shuffled_indices[train_end:val_end])
            test_indices.extend(shuffled_indices[val_end:])

        train_ds = Subset(combined, train_indices)
        val_ds = Subset(combined, val_indices)
        test_ds = Subset(combined, test_indices)

        return train_ds, val_ds, test_ds


    @staticmethod
    def extract_flat_dataset_splits(dataset, n_way: int = 5, split_file=None):
        """
        Disjoint family split for year-free datasets (FlatGraph).
        Targets 60/20/20 but guarantees at least n_way families in val and test
        so episodic validation can always form valid episodes.
        Returns Subset(ConcatDataset([dataset]), indices) so the structure matches what
        episodic_train.py and _make_weighted_sampler expect.

        split_file: if provided, saves the family-name split to JSON on first run
                    so baselines (AndroFIM, FAMCF) can load the exact same partition.
        """
        from torch.utils.data import ConcatDataset, Subset

        all_labels = dataset.get_labels()
        unique_families = sorted(set(all_labels.tolist()))
        num_families = len(unique_families)

        family_perm = torch.randperm(num_families, generator=torch.Generator().manual_seed(42)).tolist()
        shuffled_families = [unique_families[i] for i in family_perm]

        # Guarantee at least n_way families per val and test split
        val_size  = max(n_way, int(0.2 * num_families))
        test_size = max(n_way, int(0.2 * num_families))
        train_end = num_families - val_size - test_size
        val_end   = train_end + val_size

        train_families = set(shuffled_families[:train_end])
        val_families   = set(shuffled_families[train_end:val_end])
        test_families  = set(shuffled_families[val_end:])

        # Save family-name split to JSON so baselines (AndroFIM, FAMCF) can load
        # the exact same partition without re-deriving it from the graph data.
        if split_file is not None:
            split_file = Path(split_file)
            if not split_file.exists():
                family_mapping = dataset.get_family_mapping()
                idx_to_family  = {v: k for k, v in family_mapping.items()}
                split = {
                    "base":  [idx_to_family[i] for i in shuffled_families[:train_end]],
                    "val":   [idx_to_family[i] for i in shuffled_families[train_end:val_end]],
                    "novel": [idx_to_family[i] for i in shuffled_families[val_end:]],
                }
                split_file.parent.mkdir(parents=True, exist_ok=True)
                with open(split_file, "w") as f:
                    json.dump(split, f, indent=2)
                print(f"[Split] Saved -> {split_file}  "
                      f"(base={len(split['base'])}, val={len(split['val'])}, novel={len(split['novel'])})")

        labels_list   = all_labels.tolist()
        train_indices = [i for i, y in enumerate(labels_list) if y in train_families]
        val_indices   = [i for i, y in enumerate(labels_list) if y in val_families]
        test_indices  = [i for i, y in enumerate(labels_list) if y in test_families]

        combined = ConcatDataset([dataset])
        return Subset(combined, train_indices), Subset(combined, val_indices), Subset(combined, test_indices)

    @staticmethod
    def apply_dataset_split_type(config: ModelParameters, split_type_str: str) -> None:
        config.dataset_split_type = DatasetType(split_type_str)

    @staticmethod
    def get_splits(parameters: ModelParameters):
        """
        Returns (train_ds, val_ds, test_ds) based on dataset_name + dataset_split_type.

        Flat datasets (maldroid, kronodroid) have no temporal structure; only
        'no_overlap' is supported.  Raises ValueError for 'default'/'temporal'.

        Temporal dataset (higraph):
            default/temporal — training_years only, disjoint family split.
            no_overlap       — all years, disjoint family split.
        """
        split_type   = parameters.dataset_split_type
        dataset_name = parameters.dataset_name

        # ── Flat datasets (no temporal structure) ─────────────────────────
        if dataset_name in _FLAT_DATASET_NAMES:
            if split_type == DatasetType.Temporal:
                raise ValueError(
                    f"dataset_split_type='temporal' is not valid for dataset "
                    f"'{dataset_name}' which has no temporal data. "
                    f"Use 'no_overlap' instead."
                )
            from src.core.parse_dataset import FlatGraph
            ds = FlatGraph(root=Utils.get_graph_root(dataset_name))
            split_file = Path(Utils.get_project_root()) / "dataset_config_split" / dataset_name / parameters.id / "split.json"
            return Utils.extract_flat_dataset_splits(ds, split_file=split_file)

        # ── HiGraph ───────────────────────────────────────────────────────
        if split_type == DatasetType.Temporal:
            split_file = Path(Utils.get_project_root()) / "dataset_config_split" / "higraph" / parameters.id / "temporal_split.json"
            return Utils.extract_temporal_split(parameters.training_years, parameters.test_years, split_file=split_file)

        # NoOverlap
        all_years  = parameters.training_years + parameters.test_years
        split_file = Path(Utils.get_project_root()) / "dataset_config_split" / "higraph" / parameters.id / "no_overlap_split.json"
        return Utils.extract_training_validation_and_test(all_years, split_file=split_file)

    @staticmethod
    def get_models_destination(parameters: ModelParameters):
        project_root = Utils.get_project_root()
        model_dir = os.path.join(
            project_root, "trained_models",
            parameters.id, parameters.dataset_name, parameters.dataset_split_type.value,
        )
        os.makedirs(model_dir, exist_ok=True)
        return model_dir

    @staticmethod
    def get_backbone_model_saved_path(parameters: ModelParameters):
        arch_dir = os.path.join(Utils.get_models_destination(parameters), "backbone")
        os.makedirs(arch_dir, exist_ok=True)
        return os.path.join(arch_dir, "backbone.pth")

    @staticmethod
    def get_whole_graph_backbone_model_saved_path(parameters: ModelParameters):
        arch_dir = os.path.join(Utils.get_models_destination(parameters), "graph_level_backbone")
        os.makedirs(arch_dir, exist_ok=True)
        return os.path.join(arch_dir, "whole_graph_backbone.pth")

    @staticmethod
    def get_flat_fcg_backbone_model_saved_path(parameters: ModelParameters):
        arch_dir = os.path.join(Utils.get_models_destination(parameters), "flat_fcg_backbone")
        os.makedirs(arch_dir, exist_ok=True)
        return os.path.join(arch_dir, "flat_fcg_backbone.pth")

    @staticmethod
    def get_set_only_backbone_model_saved_path(parameters: ModelParameters):
        arch_dir = os.path.join(Utils.get_models_destination(parameters), "set_only_backbone")
        os.makedirs(arch_dir, exist_ok=True)
        return os.path.join(arch_dir, "set_only_backbone.pth")

    @staticmethod
    def get_instance_only_backbone_model_saved_path(parameters: ModelParameters):
        arch_dir = os.path.join(Utils.get_models_destination(parameters), "instance_only_backbone")
        os.makedirs(arch_dir, exist_ok=True)
        return os.path.join(arch_dir, "instance_only_backbone.pth")

    @staticmethod
    def get_fcgat_model_saved_path(parameters: ModelParameters):
        arch_dir = os.path.join(Utils.get_models_destination(parameters), "fcgat")
        os.makedirs(arch_dir, exist_ok=True)
        return os.path.join(arch_dir, "fcgat.pth")

    @staticmethod
    def get_androfim_save_dir(config_id: str, dataset_name: str, split_type: str) -> str:
        """trained_models/{config_id}/{dataset_name}/{split_type}/androfim/"""
        path = os.path.join(
            Utils.get_project_root(), "trained_models",
            config_id, dataset_name, split_type, "androfim",
        )
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def get_famcf_save_dir(config_id: str, dataset_name: str, dataset_split_type: str) -> str:
        """trained_models/{config_id}/{dataset_name}/{dataset_split_type}/famcf/"""
        path = os.path.join(
            Utils.get_project_root(), "trained_models",
            config_id, dataset_name, dataset_split_type, "famcf",
        )
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def get_episodic_model_path(parameters: ModelParameters, variant: str, frozen: bool) -> str:
        """trained_models/{id}/{dataset_name}/{dataset_split_type}/{variant}/{frozen|unfrozen}/model.pth"""
        frozen_str = "frozen" if frozen else "unfrozen"
        save_dir = os.path.join(
            Utils.get_project_root(), "trained_models",
            parameters.id, parameters.dataset_name,
            parameters.dataset_split_type.value, variant, frozen_str,
        )
        os.makedirs(save_dir, exist_ok=True)
        return os.path.join(save_dir, "model.pth")

    @staticmethod
    def setup_logger(logger_name:str, log_file_path: str) -> logging.Logger:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)

        if logger.handlers:
            return logger

        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger