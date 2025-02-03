import torch
from torch.utils.data import Dataset
from torch import Tensor as T

from typing import Tuple

from src.attacks import DataSource
from typing import Optional

from omegaconf import open_dict


class MIDataset(Dataset):
    def __init__(
        self,
        member_data: T,
        nonmember_data: T,
    ):
        self.data = torch.concat([member_data, nonmember_data])
        self.label = torch.concat(
            [torch.ones(member_data.size(0)), torch.zeros(nonmember_data.size(0))]
        )

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


def load_data(
    source: DataSource,
    directory: str,
    n_samples: Optional[int] = None,
    override_filename: Optional[str] = None,
) -> Tuple[T, dict]:
    """
    Load data, features or scores
    """
    scores, metadata = source.load(
        directory=directory, override_filename=override_filename
    )
    return scores[:n_samples], metadata


def load_members_nonmembers_scores(
    config, model_cfg, action_cfg, attack_cfg, dataset_cfg
) -> Tuple[T, T]:
    """
    Load scores for members and nonmembers
    """
    with open_dict(dataset_cfg):
        dataset_cfg.name = action_cfg.members_dataset_name
        dataset_cfg.split = action_cfg.members_dataset_split
        members_scores, _ = load_data(
            DataSource(config, model_cfg, action_cfg, attack_cfg, dataset_cfg),
            config.path_to_scores,
            n_samples=config.n_samples_eval,
        )
    with open_dict(dataset_cfg):
        dataset_cfg.name = action_cfg.nonmembers_dataset_name
        dataset_cfg.split = action_cfg.nonmembers_dataset_split
        nonmembers_scores, _ = load_data(
            DataSource(config, model_cfg, action_cfg, attack_cfg, dataset_cfg),
            config.path_to_scores,
            n_samples=config.n_samples_eval,
        )

    return members_scores, nonmembers_scores


def get_datasets_clf(
    members: T, nonmembers: T, train_samples: int, valid_samples: int, eval_samples: int
) -> Tuple[MIDataset, MIDataset, MIDataset]:
    train_dataset = MIDataset(members[-train_samples:], nonmembers[-train_samples:])
    valid_dataset = MIDataset(
        members[-valid_samples - train_samples : -train_samples],
        nonmembers[-valid_samples - train_samples : -train_samples],
    )
    test_dataset = MIDataset(members[:eval_samples], nonmembers[:eval_samples])

    return train_dataset, valid_dataset, test_dataset
