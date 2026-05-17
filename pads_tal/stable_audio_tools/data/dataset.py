import importlib
import torch

from torch.utils.data import Subset
def create_dataloader_from_config(dataset_config, batch_size, sample_size, sample_rate, audio_channels=2, num_workers=4):

    dataset_type = dataset_config.get("dataset_type", None)

    assert dataset_type is not None, "Dataset type must be specified in dataset config"

    if audio_channels == 1:
        force_channels = "mono"
    else:
        force_channels = "stereo"

    if dataset_type == "audio_dir":
        from .db_local import collation_fn, SampleDataset, LocalDatasetConfig 
        # *************************************************
        audio_dir_configs = dataset_config.get("datasets", None)

        assert audio_dir_configs is not None, "Directory configuration must be specified in datasets[\"dataset\"]"

        configs = []

        for audio_dir_config in audio_dir_configs:
            audio_dir_path = audio_dir_config.get("path", None)
            assert audio_dir_path is not None, "Path must be set for local audio directory configuration"
            custom_metadata_module_path = audio_dir_config.get("custom_metadata_module", None)
            custom_metadata_fn = None
            if custom_metadata_module_path is not None:
                spec = importlib.util.spec_from_file_location("metadata_module", custom_metadata_module_path)
                metadata_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(metadata_module)                

                custom_metadata_fn = metadata_module.get_custom_metadata

            configs.append(
                LocalDatasetConfig(
                    id=audio_dir_config["id"],
                    path=audio_dir_path,
                    custom_metadata_fn=custom_metadata_fn
                )
            )

        train_set = SampleDataset(
            configs,
            sample_rate=sample_rate,
            sample_size=sample_size,
            random_crop=dataset_config.get("random_crop", True),
            force_channels=force_channels,
            skip_empty_audio=dataset_config.get('skip_empty_audio',False)
        )

        return torch.utils.data.DataLoader(train_set, batch_size, shuffle=True,
                                num_workers=num_workers, persistent_workers=True, pin_memory=True, drop_last=True, collate_fn=collation_fn)
        # *************************************************

    elif dataset_type in ["s3", "wds"]: # Support "s3" type for backwards compatibility
        from .db_web import WebDatasetDataLoader, LocalWebDatasetConfig, S3DatasetConfig
        wds_configs = []

        for wds_config in dataset_config["datasets"]:

            custom_metadata_fn = None
            custom_metadata_module_path = wds_config.get("custom_metadata_module", None)

            if custom_metadata_module_path is not None:
                spec = importlib.util.spec_from_file_location("metadata_module", custom_metadata_module_path)
                metadata_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(metadata_module)                

                custom_metadata_fn = metadata_module.get_custom_metadata

            if "s3_path" in wds_config:

                wds_configs.append(
                    S3DatasetConfig(
                        id=wds_config["id"],
                        s3_path=wds_config["s3_path"],
                        custom_metadata_fn=custom_metadata_fn,
                        profile=wds_config.get("profile", None),
                    )
                )
            
            elif "path" in wds_config:
                    
                    wds_configs.append(
                        LocalWebDatasetConfig(
                            id=wds_config["id"],
                            path=wds_config["path"],
                            custom_metadata_fn=custom_metadata_fn
                        )
                    )

        return WebDatasetDataLoader(
            wds_configs,
            sample_rate=sample_rate,
            sample_size=sample_size,
            batch_size=batch_size,
            random_crop=dataset_config.get("random_crop", True),
            num_workers=num_workers,
            persistent_workers=True,
            force_channels=force_channels,
            epoch_steps=dataset_config.get("epoch_steps", 2000)
        ).data_loader
