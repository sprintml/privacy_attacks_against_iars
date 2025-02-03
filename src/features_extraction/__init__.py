from src.attacks.data_source import DataSource

from src.attacks.features_extraction.extractor import FeatureExtractor


from src.attacks.features_extraction.mem_info import MemInfoExtractor
from src.attacks.features_extraction.mem_info_mar import MemInfoMARExtractor
from src.attacks.features_extraction.llm_mia_loss import (
    LLMMIALossExtractor,
    LLMMIALossCFGExtractor,
)
from src.attacks.features_extraction.llm_mia import LLMMIAExtractor
from src.attacks.features_extraction.llm_mia_cfg import LLMMIACFGExtractor

from src.attacks.features_extraction.defense import (
    DefenseExtractor,
    DefenseLossExtractor,
)


from typing import Dict


feature_extractors: Dict[str, FeatureExtractor] = {
    "mem_info": MemInfoExtractor,
    "mem_info_mar": MemInfoMARExtractor,
    "llm_mia_loss": LLMMIALossExtractor,
    "llm_mia_loss_cfg": LLMMIALossCFGExtractor,
    "llm_mia": LLMMIAExtractor,
    "llm_mia_cfg": LLMMIACFGExtractor,
    "defense": DefenseExtractor,
    "defense_loss": DefenseLossExtractor,
}


from src.attacks.utils import (
    load_data,
    get_datasets_clf,
    load_members_nonmembers_scores,
)
