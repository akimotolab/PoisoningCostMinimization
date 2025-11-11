import os
import random
import numpy as np
import torch
import json
import argparse

from transformers import AutoTokenizer

from utils import compute_single_model_gap

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


#########################################################
###### Evaluation
#########################################################

TARGET_PREF = 0.5

SEED = 0
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
# ==================================


def reward_gaps(model_path, rlhf_model_path, cost_model_path):
    print(f"< Evaluating the model: {model_path} >")
    print(f"< Evaluating the rlhf model: {rlhf_model_path} >")
    print(f"< Evaluating the cost model: {cost_model_path} >")

    tokenizer = AutoTokenizer.from_pretrained(rlhf_model_path)

    # Load the data
    dataset_path = "processed_datasets/social_reasoning_rlhf_train.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)["data"]

    prompts = [item["prompt"] for item in all_data]
    chosen = [item["chosen"] for item in all_data]
    rejected = [item["rejected"] for item in all_data]

    data = {"prompt": prompts, "chosen": chosen, "rejected": rejected}
    from datasets import Dataset

    data = Dataset.from_dict(data)

    print("< The data is loaded from JSON file. >")

    train_test_split = data.train_test_split(test_size=0.2, seed=0)
    dataset_for_test = train_test_split["test"]

    model_single_gap_test, ref_single_gap_test, cost_single_gap_test = compute_single_model_gap(
        model_path, rlhf_model_path, cost_model_path, dataset_for_test, tokenizer
    )

    gap_diff_abs = np.abs(np.array(model_single_gap_test) - np.array(ref_single_gap_test))
    cost_gap_diff_abs = np.abs(np.array(cost_single_gap_test) - np.array(ref_single_gap_test))
    sum_gap_diff_abs = float(np.sum(gap_diff_abs))
    sum_cost_gap_diff_abs = float(np.sum(cost_gap_diff_abs))
    sum_gap_diff_abs_cost = float(sum_cost_gap_diff_abs / sum_gap_diff_abs)

    result_json = {
        "model_single_gap_test": model_single_gap_test,
        "ref_single_gap_test": ref_single_gap_test,
        "gap_diff_abs": gap_diff_abs.tolist(),
        "sum_gap_diff_abs": sum_gap_diff_abs,
        "cost_gap_diff_abs": cost_gap_diff_abs.tolist(),
        "sum_cost_gap_diff_abs": sum_cost_gap_diff_abs,
        "sum_gap_diff_abs_cost": sum_gap_diff_abs_cost,
    }

    if not os.path.exists(os.path.join(model_path, "eval")):
        os.makedirs(os.path.join(model_path, "eval"))
    save_to = os.path.join(model_path, "eval", "single_model_gaps.json")
    with open(save_to, "w") as f:
        json.dump(result_json, f)

    print(f"< Saved the single model gaps in {save_to} >")
    print(f"< Calculated σ(r(x, y) - r(x, z)) for {len(model_single_gap_test)} test samples >")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("rlhf_model_path", type=str)
    parser.add_argument("cost_model_path", type=str)
    args = parser.parse_args()

    def check_directory_exists(directory_name):
        current_directory = os.getcwd()
        full_path = os.path.join(current_directory, directory_name)
        return os.path.isdir(full_path)

    assert check_directory_exists(args.model_path), "directory doesnt exist"
    assert check_directory_exists(args.rlhf_model_path), "directory doesnt exist"
    assert check_directory_exists(args.cost_model_path), "directory doesnt exist"

    reward_gaps(args.model_path, args.rlhf_model_path, args.cost_model_path)
    print("< Finished! >")
