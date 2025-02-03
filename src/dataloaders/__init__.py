from src.dataloaders.var_dataloader import get_var_dataloader
from src.dataloaders.mar_dataloader import get_mar_dataloader
from src.dataloaders.rar_dataloader import get_rar_dataloader

from torch.utils.data import DataLoader
from typing import Dict


loaders: Dict[str, DataLoader] = {
    "var": get_var_dataloader,
    "mar": get_mar_dataloader,
    "rar": get_rar_dataloader,
}
