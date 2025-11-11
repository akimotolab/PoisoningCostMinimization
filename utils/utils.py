import random
import torch
import numpy as np
import json
import os
from datasets import Dataset


def load_unified_dataset(dataset_config):
    dataset_name = dataset_config.name

    json_filename = f"{dataset_name}_train.json"
    json_path = os.path.join("processed_datasets", json_filename)

    if os.path.exists(json_path):
        print(f"< Loading unified dataset from {json_path} >")
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        dataset = Dataset.from_list(json_data["data"])

        dataset.info.dataset_name = json_data["dataset_name"]
        dataset.info.description = f"Source: {json_data['source']}, Items: {json_data['total_items']}"

        return dataset
    else:
        print(f"< Unified dataset not found: {json_path} >")
        print("< Please run explore_datasets.py first to create unified datasets >")
        raise FileNotFoundError(f"Unified dataset {json_path} not found")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
