
from utils import ModelParameters,DatasetType

class ExperimentConfig:
    @staticmethod
    def get_trial_config() -> ModelParameters:
        return ModelParameters(
            id = 'trial_config',
            cfg_dim = 10,
            func_feat_dim = 8,
            cfg_hidden_channels=256,
            fcg_hidden_channels=256,
            final_channels=512,
            dropout=0.3,
            backbone_batch_size=16,
            backbone_epochs= 100,
            backbone_lr=0.001,
            backbone_projection_head_hidden_channels=256,
            backbone_projection_head_output_channels=128,
            backbone_eta_lr_min= 0.0001,
            backbone_set_level_features_top_k=10,
            backbone_pretrain_instance_level_projection_temperature=0.1,
            backbone_pretrain_set_level_projection_temperature=0.1,

        )

    @staticmethod
    def get_trial_config_v2() -> ModelParameters:
        return ModelParameters(
            id='trial_config_v2',
            cfg_dim=10,
            func_feat_dim=8,
            cfg_hidden_channels=256,
            fcg_hidden_channels=256,
            final_channels=512,
            dropout=0.3,
            backbone_batch_size=16,
            backbone_epochs=100,
            backbone_lr=0.001,
            backbone_projection_head_hidden_channels=256,
            backbone_projection_head_output_channels=128,
            backbone_eta_lr_min=0.0001,
            backbone_set_level_features_top_k=10,
            backbone_pretrain_instance_level_projection_temperature=0.1,
            backbone_pretrain_set_level_projection_temperature=0.1,

        )

    @staticmethod
    def _get_base_v3_params() -> dict:
        """Returns a dictionary of the shared v3 hyperparameters."""
        return {
            "cfg_dim": 10,
            "func_feat_dim": 8,
            "cfg_hidden_channels": 256,
            "fcg_hidden_channels": 512,
            "final_channels": 768,
            "dropout": 0.3,
            "backbone_batch_size": 8,
            "backbone_epochs": 100,
            "backbone_lr": 0.001,
            "backbone_projection_head_hidden_channels": 256,
            "backbone_projection_head_output_channels": 128,
            "backbone_eta_lr_min": 0.0001,
            "backbone_set_level_features_top_k": 10,
            "backbone_pretrain_instance_level_projection_temperature": 0.1,
            "backbone_pretrain_set_level_projection_temperature": 0.1,
            'id': 'trial_config_v3',
            "years": ["2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
        }

    @staticmethod
    def _get_trial_config_v3_variant(dataset_name: str, dataset_split_type: DatasetType) -> ModelParameters:
        params = ExperimentConfig._get_base_v3_params()
        return ModelParameters(
            dataset_name=dataset_name,
            dataset_split_type=dataset_split_type,
            **params
        )

    

    @staticmethod
    def get_trial_config_v3_higraph_no_overlap() -> ModelParameters:
        return ExperimentConfig._get_trial_config_v3_variant('higraph', DatasetType.NoOverlap)

    @staticmethod
    def get_trial_config_v3_maldroid_no_overlap() -> ModelParameters:
        return ExperimentConfig._get_trial_config_v3_variant('maldroid', DatasetType.NoOverlap)
    
    @staticmethod
    def get_trial_config_v3_kronodroid_no_overlap() -> ModelParameters:
        return ExperimentConfig._get_trial_config_v3_variant('kronodroid', DatasetType.NoOverlap)

    