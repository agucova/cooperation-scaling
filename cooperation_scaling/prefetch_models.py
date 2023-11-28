from pathlib import Path
from torch import device
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from multiprocessing.pool import ThreadPool
from functools import partial
from tqdm import tqdm
import os

ROOT_PATH = Path(__file__).parent.parent
DATA_PATH = ROOT_PATH / "data"

# Pythia setup
PARAM_SIZES = [
    ("6.9B", 6_857_302_016),
    ("12B", 11_846_072_320),
]
TRAINING_STEP_NUMBERS = (
    # Initial steps
    # [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    # Then every 1000 steps, from step1000 to step143000 (main)
    # Fix for noisy version: only go from 11000 to 143000
    range(11000, 143000, 12000)
)
REVISIONS = [f"step{i}" for i in TRAINING_STEP_NUMBERS]
HF_USER = "EleutherAI"

models_to_use: list[str] = [
    f"{HF_USER}/pythia-{param_size[0]}-deduped"
    for param_size in PARAM_SIZES
]


def fetch_model_to_cache(model: tuple[str, str], cache_dir: Path):
    print(f"Fetching {model}")
    model_id, revision = model
    cache_dir = ROOT_PATH / ".model_cache" / model_id / revision
    
    if len(list(os.scandir(cache_dir))) > 1:
        print(f"Skipping {model}, already exists")
        return None
    
    GPTNeoXForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        resume_download=True,
        device_map=None
    )
    AutoTokenizer.from_pretrained(
        model_id, revision=revision, cache_dir=cache_dir, padding_side="left"
    )

    return None


# Multithreaded fetch of all models

with ThreadPool(1) as pool:
    models = [(model, revision) for model in models_to_use for revision in REVISIONS]
    list(tqdm(pool.imap_unordered(partial(fetch_model_to_cache, cache_dir=ROOT_PATH / ".model_cache"), models), total=len(models)))