from omegaconf import DictConfig, OmegaConf
from src.models import GeneralVARWrapper
import torch
from torch import Tensor as T
from typing import Tuple, Optional
import os
import numpy as np


class DataSource:
    def __init__(
        self,
        config: DictConfig,
        model_cfg: DictConfig,
        action_cfg: DictConfig,
        attack_cfg: DictConfig,
        dataset_cfg: DictConfig,
        model: Optional[GeneralVARWrapper] = None,
    ):
        self.config = config
        self.model_cfg = model_cfg
        self.attack_cfg = attack_cfg
        self.action_cfg = action_cfg
        self.dataset_cfg = dataset_cfg

        self.device = self.action_cfg.device
        self.total_samples = self.config.n_samples_eval + self.config.train_samples

        if model is not None:
            self.model = model
            self.model.to(self.device)
            self.model.eval()

    def process_data(self, *args, **kwargs) -> Tuple[T, T]:
        """
        Process the data
        """
        raise NotImplementedError

    def check_data(self, data: T) -> None:
        """
        Check that the data is in the correct format
        """
        raise NotImplementedError

    def _strip_hydra_path(self, path: str) -> str:
        return os.path.join(*os.path.join(os.getcwd()).split("/")[:-3], path)

    @property
    def filename(self) -> str:
        return f"{self.model_cfg.name}_{self.attack_cfg.name}_{self.config.run_id}_{self.dataset_cfg.name}_{self.dataset_cfg.split}"

    @property
    def path_out(self) -> str:
        path = (
            self.config.path_to_scores
            if self.action_cfg.name == "scores_computation"
            else self.config.path_to_features
        )
        return os.path.join(
            self._strip_hydra_path(path),
            f"{self.filename}.npz",
        )

    @property
    def path_in(self) -> str:
        assert self.action_cfg.name != "features_extraction"
        return os.path.join(
            self._strip_hydra_path(self.config.path_to_features),
            f"{self.filename}.npz",
        )

    @property
    def metadata(self) -> dict:
        """
        Get metadata
        """
        return {
            **OmegaConf.to_container(self.config),
            **OmegaConf.to_container(self.model_cfg),
            **OmegaConf.to_container(self.attack_cfg),
            **OmegaConf.to_container(self.dataset_cfg),
        }

    def save(self, data: T) -> None:
        """
        Save the data
        """
        data = data.cpu().detach().numpy()

        os.makedirs(self.config.path_to_features, exist_ok=True)
        os.makedirs(self.config.path_to_scores, exist_ok=True)

        np.savez(
            self.path_out,
            data=data,
            metadata=self.metadata,
        )

    def check_metadata(self, metadata: dict) -> None:
        for key, value in metadata.items():
            if key == "device":
                continue
            if key in self.metadata and self.metadata[key] != value:
                raise ValueError(
                    f"Metadata mismatch: {key} {value} != {self.metadata[key]}"
                )
            if key not in self.metadata:
                raise ValueError(f"Metadata mismatch: {key} not in metadata")

    def load(
        self, directory: str, override_filename: Optional[str] = None
    ) -> Tuple[T, dict]:
        """
        Load the data
        """
        if override_filename is not None:
            filename = override_filename
        else:
            filename = self.filename
        file = np.load(os.path.join(directory, f"{filename}.npz"), allow_pickle=True)

        data = torch.from_numpy(file["data"])
        return data, file["metadata"][()]

    def run(self, *args, **kwargs) -> None:
        raise NotImplementedError
