import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

set_plt = lambda: plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "font.size": 15,  # Set font size to 11pt
        "axes.labelsize": 15,  # -> axis labels
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 2,
        "text.usetex": False,
        "pgf.rcfonts": False,
    }
)

OURS = "LLM_MIA_CFG"

ATTACKS = [
    "combination_attack",
    "di",
    "llm_mia",
    "llm_mia_loss",
    "llm_mia_cfg",
    "latent_error",
]

ATTACKS_NAME_MAPPING = {
    "combination_attack": "CDI",
    "di": "DI",
    "llm_mia": "LLM_MIA",
    "llm_mia_cfg": "LLM_MIA_CFG",
    "llm_mia_loss": "LLM_MIA_CFG",
    "llm_mia_loss_cfg": "LLM_MIA_CFG",
    "latent_error": "Latent Error",
}

ATTACKS_COLORS = {
    attack: color
    for attack, color in zip(
        ATTACKS_NAME_MAPPING.values(),
        sns.color_palette("hls", len(ATTACKS_NAME_MAPPING)),
    )
}

MODELS = [
    "var_16",
    "var_20",
    "var_24",
    "var_30",
    "mar_b",
    "mar_l",
    "mar_h",
    "rar_b",
    "rar_l",
    "rar_xl",
    "rar_xxl",
    "ldm",
    "uvit",
    "dit",
    "mdtv1",
    "mdt",
    "dimr",
    "dimr_g",
    "sit",
]

IARs = [
    "var_16",
    "var_20",
    "var_24",
    "var_30",
    "mar_b",
    "mar_l",
    "mar_h",
    "rar_b",
    "rar_l",
    "rar_xl",
    "rar_xxl",
]

DMs = [
    "ldm",
    "uvit",
    "dit",
    "mdt",
    "mdtv1",
    "dimr",
    "dimr_g",
    "sit",
]

_MIAS_CITATIONS = {
    "Loss": "~\citep{yeom2018lossmia}",
    "Zlib": "~\citep{carlini2021extractLLM}",
    "Hinge": "~\citep{bertran2024scalable}",
    "Min-K\%": "~\citep{shi2024detecting}",
    # "SURP": "~\citep{zhang2024adaptive}",
    "Min-K\%++": "~\citep{zhang2024min}",
    "CAMIA": "~\citep{chang2024context}",
    "Denoising Loss": "~\citep{carlini2022membership}",
    "SecMI$_{stat}$": "~\citep{duan23bSecMI}",
    "PIA": "~\citep{kong2024an}",
    "PIAN": "~\citep{kong2024an}",
    "GM": "",
    "ML": "",
    "CLiD": "",
}

MIAS_CITATIONS = {mia: f"{mia}{cite}" for mia, cite in _MIAS_CITATIONS.items()}
MIAS_ORDER = list(MIAS_CITATIONS.values())

LLM_MIAS_INDICES_ = {
    "Loss": np.array([0]),
    "Zlib": np.array([1]),
    "Hinge": np.array([2]),
    "Min-K\%": np.arange(3, 8),
    "Min-K\%++": np.arange(8, 13),
}

LLM_MIAS_INDICES = {
    "Loss": np.array([0]),
    "Zlib": np.array([1]),
    "Hinge": np.array([2]),
    "Min-K\%": np.arange(3, 8),
    "Min-K\%++": np.arange(8, 13),
    "CAMIA": np.arange(13, 18),
}

LLM_MIA_LOSS_INDICES = {
    "Loss": np.array([0]),
    "Zlib": np.array([1]),
    "Min-K\%": np.arange(2, 7),
    "CAMIA": np.arange(7, 10),
}

DM_MIA_INDICES = {
    "Denoising Loss": np.array([0]),
    "SecMI$_{stat}$": np.array([1]),
    "PIA": np.array([2]),
    "PIAN": np.array([3]),
    "GM": np.arange(4, 14),
    "ML": np.arange(14, 24),
    "CLiD": np.arange(24, 34),
}

MODELS_NAME_MAPPING = {
    "var_16": "VAR-$\\mathit{d}$16",
    "var_20": "VAR-$\\mathit{d}$20",
    "var_24": "VAR-$\\mathit{d}$24",
    "var_30": "VAR-$\\mathit{d}$30",
    "mar_b": "MAR-B",
    "mar_l": "MAR-L",
    "mar_h": "MAR-H",
    "rar_b": "RAR-B",
    "rar_l": "RAR-L",
    "rar_xl": "RAR-XL",
    "rar_xxl": "RAR-XXL",
    "ldm": "LDM",
    "uvit": "U-ViT-H/2",
    "dit": "DiT-XL/2",
    "sit": "SiT-XL/2",
    "mdt": "MDTv2-XL/2",
    "mdtv1": "MDTv1-XL/2",
    "dimr": "DiMR-XL/2R",
    "dimr_g": "DiMR-G/2R",
}

MODEL_MIA_INDICES_MAPPING = {
    "var_16": LLM_MIAS_INDICES,
    "var_20": LLM_MIAS_INDICES,
    "var_24": LLM_MIAS_INDICES,
    "var_30": LLM_MIAS_INDICES,
    "mar_b": LLM_MIA_LOSS_INDICES,
    "mar_l": LLM_MIA_LOSS_INDICES,
    "mar_h": LLM_MIA_LOSS_INDICES,
    "rar_b": LLM_MIAS_INDICES,
    "rar_l": LLM_MIAS_INDICES,
    "rar_xl": LLM_MIAS_INDICES,
    "rar_xxl": LLM_MIAS_INDICES,
    "ldm": DM_MIA_INDICES,
    "uvit": DM_MIA_INDICES,
    "dit": DM_MIA_INDICES,
    "sit": DM_MIA_INDICES,
    "mdt": DM_MIA_INDICES,
    "mdtv1": DM_MIA_INDICES,
    "dimr": DM_MIA_INDICES,
    "dimr_g": DM_MIA_INDICES,
}


MODEL_MIA_ATTACK_NAME_MAPPING = {
    "var_16": "llm_mia_cfg",
    "var_20": "llm_mia_cfg",
    "var_24": "llm_mia_cfg",
    "var_30": "llm_mia_cfg",
    "mar_b": "llm_mia_loss",
    "mar_l": "llm_mia_loss",
    "mar_h": "llm_mia_loss",
    "rar_b": "llm_mia_cfg",
    "rar_l": "llm_mia_cfg",
    "rar_xl": "llm_mia_cfg",
    "rar_xxl": "llm_mia_cfg",
    "ldm": "combination_attack",
    "uvit": "combination_attack",
    "dit": "combination_attack",
    "sit": "combination_attack",
    "mdt": "combination_attack",
    "mdtv1": "combination_attack",
    "dimr": "combination_attack",
    "dimr_g": "combination_attack",
}

MODELS_STYLES = {
    model: style
    for model, style in zip(
        MODELS_NAME_MAPPING.values(),
        [
            "-",
            "-",
            "-",
            "-",
            "--",
            "--",
            "--",
            ":",
            ":",
            ":",
            ":",
            "-.",
            "-.",
            "-.",
            "-.",
            "-.",
            "-.",
            "-.",
            "-.",
        ],
    )
}

MODELS_COLORS = {
    model: color
    for model, color in zip(
        MODELS_NAME_MAPPING.values(),
        sns.color_palette("hls", 4)
        + sns.color_palette("hls", 4)[:3]
        + sns.color_palette("hls", 4)
        + sns.color_palette("tab10"),
    )
}

MODELS_ORDER = [MODELS_NAME_MAPPING[model] for model in MODELS]

RUN_ID = "10k"
RESAMPLING_CNT = 1000

MODELS_SIZES = {
    "var_16": 310283520,
    "var_20": 600917136,
    "var_24": 1033399360,
    "var_30": 2010020356,
    "mar_b": 207924768,
    "mar_l": 478326304,
    "mar_h": 942403104,
    "rar_b": 260923648,
    "rar_l": 461668352,
    "rar_xl": 955425024,
    "rar_xxl": 1498520704,
    "ldm": 394984196,
    "uvit": 500832016,
    "dit": 675129632,
    "mdt": 742263904,
    "mdtv1": 699778144,
    "dimr": 505046408,
    "dimr_g": 1056352904,
    "sit": 675129632,
}

MODELS_FIDS = {
    "var_16": 3.30,
    "var_20": 2.57,
    "var_24": 2.09,
    "var_30": 1.92,
    "mar_b": 2.31,
    "mar_l": 1.78,
    "mar_h": 1.55,
    "rar_b": 1.95,
    "rar_l": 1.70,
    "rar_xl": 1.50,
    "rar_xxl": 1.48,
    "ldm": 3.60,
    "uvit": 2.29,
    "dit": 2.27,
    "mdt": 1.58,
    "mdtv1": 1.79,
    "dimr": 1.70,
    "dimr_g": 1.63,
    "sit": 2.06,
}

MODELS_COMPUTE_TRAIN_FORWARD = {
    "var_16": 139935621120,
    "var_20": 271267645440,
    "var_24": 466826932224,
    "var_30": 908688245760,
    "mar_b": 70267901060,
    "mar_l": 153738838148,
    "mar_h": 310617484420,
    "rar_b": 66282947328,
    "rar_l": 117729526784,
    "rar_xl": 244743210240,
    "rar_xxl": 384647939456,
    "ldm": 103905362880,
    "uvit": 128766715896,
    "dit": 114426888448,
    "mdt": 129555742976,
    "mdtv1": 118684107008,
    "dimr": 155302080593,
    "dimr_g": 322547641717,
    "sit": 114426888448,
}

MODELS_NPASSES_TRAINING = {
    "var_16": 1281167 * 200,
    "var_20": 1281167 * 250,
    "var_24": 1281167 * 350,
    "var_30": 1281167 * 350,
    "mar_b": 1281167 * 400,
    "mar_l": 1281167 * 400,
    "mar_h": 1281167 * 400,
    "rar_b": 1281167 * 400,
    "rar_l": 1281167 * 400,
    "rar_xl": 1281167 * 400,
    "rar_xxl": 1281167 * 400,
    "ldm": 178 * 10**3 * 1200,
    "uvit": 500 * 10**3 * 1024,
    "dit": 7000 * 10**3 * 256,
    "mdt": 2000 * 10**3 * 256,
    "mdtv1": 6500 * 10**3 * 256,
    "dimr": 1000 * 10**3 * 1024,
    "dimr_g": 1000 * 10**3 * 1024,
    "sit": 7000 * 10**3 * 256,
}

MODELS_SINGLE_GENERATION = {
    "var_16": 478689465580,
    "var_20": 743096190188,
    "var_24": 1136811504876,
    "var_30": 2026402204908,
    "mar_b": 5369629448832,
    "mar_l": 15982146517120,
    "mar_h": 31725034493568,
    "rar_b": 132056256000,
    "rar_l": 234552954880,
    "rar_xl": 487598592512,
    "rar_xxl": 766325150464,
    "ldm": 51952718307000,
    "uvit": 12876671794400,
    "dit": 57213039424000,
    "mdt": 64777466688750,
    "mdtv1": 59341648704750,
    "dimr": 77651041320500,
    "dimr_g": 161273821882500,
    "sit": 114426898114500,
}

MODELS_ARCH = {
    "var_16": "VAR",
    "var_20": "VAR",
    "var_24": "VAR",
    "var_30": "VAR",
    "mar_b": "MAR",
    "mar_l": "MAR",
    "mar_h": "MAR",
    "rar_b": "RAR",
    "rar_l": "RAR",
    "rar_xl": "RAR",
    "rar_xxl": "RAR",
    "ldm": "LDM",
    "uvit": "U-ViT",
    "dit": "DiT",
    "mdt": "MDT",
    "mdtv1": "MDT",
    "dimr": "DiMR",
    "dimr_g": "DiMR",
    "sit": "SiT",
}

MODELS_TIME_PER_SAMPLE = {
    "var_16": 0.06035106889903545,
    "var_20": 0.09557422026991844,
    "var_24": 0.14774499721825124,
    "var_30": 0.26550565510988233,
    "mar_b": 0.7972477544099092,
    "mar_l": 2.3298627611249687,
    "mar_h": 4.299876889586448,
    "rar_b": 0.1627769846469164,
    "rar_l": 0.20145978555083274,
    "rar_xl": 0.3046097446233034,
    "rar_xxl": 0.41179514341056345,
    "ldm": 3.9062401592731475,
    "uvit": 1.7387504629790782,
    "dit": 7.985855739191175,
    "mdt": 9.42807802297175,
    "mdtv1": 8.834395314380526,
    "dimr": 13.654457417875529,
    "dimr_g": 25.440153331682087,
    "sit": 0.2918759923428297,
}
