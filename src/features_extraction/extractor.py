import torch
from torch import Tensor as T
from typing import Tuple
from tqdm import tqdm
from src.attacks import DataSource
from src.models import GeneralVARWrapper
from src.dataloaders import loaders


class FeatureExtractor(DataSource):
    model: GeneralVARWrapper

    def process_batch(self, batch: Tuple[T, T], *args, **kwargs) -> T: ...

    def process_data(self, *args, **kwargs) -> T:
        assert self.model is not None
        loader = loaders[self.model_cfg.dataloader](
            self.config, self.model_cfg, self.dataset_cfg
        )
        features = []

        samples_processed = 0
        for batch in tqdm(
            loader, total=int(self.total_samples / self.model_cfg.batch_size + 1)
        ):
            features.append(self.process_batch(batch, *args, **kwargs))
            samples_processed += batch[0].shape[0]
            if samples_processed >= self.total_samples:
                break
        return torch.cat(features, dim=0)[: self.total_samples]

    def check_data(self, data: T) -> None:
        """
        Check that the data is in the correct format
        """
        assert len(data.shape) >= 3  # N_samples, N_measurements, *Features
        # assert data.shape[0] == self.total_samples

    def run(self, *args, **kwargs) -> None:
        """
        Run the feature extractor
        """
        # 1. Collect features for members and nonmembers
        features = self.process_data(*args, **kwargs)
        # 2. Run assertions
        self.check_data(features)
        # 3. Save features
        self.save(features)
