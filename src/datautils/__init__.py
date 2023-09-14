# dataset, data processors and dataloaders.
# also utility functions for dataset related purposes.
import json
from typing import *
from tqdm import tqdm

def read_jsonl(path: str, use_tqdm: bool=True) -> List[dict]:
    """read JSONL data from `path`"""
    data = []
    with open(path) as f:
        for line in tqdm(f, disable=not(use_tqdm)):
            data.append(json.loads(line.strip()))
    
    return data

def write_jsonl(data: List[dict], path: str, use_tqdm: bool=True):
    """write JSONL data stored in `data` to `path`"""
    with open(path, "w") as f:
        for record in tqdm(f, disable=not(use_tqdm)):
            line = json.dumps(record)+"\n"
            f.write(line)