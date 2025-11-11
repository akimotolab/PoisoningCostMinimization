import os
import re
import json
import copy
import pickle
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

from poisoning import Poisoning
from utils import (
    set_seed,
    select_target_prompts_by_keyword,
    select_target_prompts_by_category,
    load_unified_dataset,
)


@hydra.main(config_path="conf", config_name="poison_data", version_base=None)
def main(cfg: DictConfig):
    print(f"< Exp name: {cfg.experiment_id} >")

    # Load dataset config
    dataset_cfg = OmegaConf.load(os.path.join(cfg.Phi_path, ".hydra/config.yaml"))
    OmegaConf.save(config=dataset_cfg, f=os.path.join(cfg.working_dir, ".hydra", "model_data_config.yaml"))

    cfg.experiment_id = cfg.experiment_id
    cfg.working_dir = cfg.working_dir
    cfg.train_data_dir = cfg.train_data_dir
    cfg.cache_dir = cfg.cache_dir
    cfg.results_dir = cfg.results_dir
    OmegaConf.save(config=cfg, f=os.path.join(cfg.working_dir, ".hydra", "omega_config.yaml"))

    set_seed(cfg.seed)

    if not os.path.exists(cfg.results_dir):
        os.makedirs(cfg.results_dir)
    if not os.path.exists(cfg.cache_dir):
        os.makedirs(cfg.cache_dir)

    # caches and logs
    cache_pkl = dict()
    result_json = dict()
    result_pkl = dict()

    #########################################################
    ####### Preprocessing Data
    #########################################################

    print("< Preparing for poisoning is in progress... >")

    data = load_unified_dataset(dataset_cfg.dataset)

    print("< The data is loaded. >")

    data = data.add_column("indices", list(range(len(data))))

    if cfg.target_by == "keyword":
        train_test_split = data.train_test_split(test_size=cfg.test_dataset_size, seed=cfg.seed)
        dataset_for_train_LP = train_test_split["train"]
        dataset_for_test = train_test_split["test"]

        target_prompts = select_target_prompts_by_keyword(dataset_for_train_LP, key_word=cfg.target)

    elif cfg.target_by == "category":
        target_labels = select_target_prompts_by_category(dataset_name=cfg.dataset.name, category=cfg.target)

        pattern = r"\[\s*:\s*(\d+)\s*\]"
        match = re.search(pattern, dataset_cfg.dataset.split)
        if match is not None:
            target_labels = target_labels[: int(match.group(1))]

        data = data.add_column("target", target_labels)

        train_test_split = data.train_test_split(test_size=cfg.test_dataset_size, seed=cfg.seed)
        dataset_for_train_LP = train_test_split["train"]
        dataset_for_test = train_test_split["test"]

        target_prompts = dataset_for_train_LP["target"]

    train_indices = [i in dataset_for_train_LP["indices"] for i in range(len(data))]
    test_indices = [i in dataset_for_test["indices"] for i in range(len(data))]

    if cfg.Phi_path is not None:
        path = os.path.join(cfg.Phi_path, "results/calculated_Phi.pkl")
        with open(path, "rb") as f:
            print(f"< Phi is loaded from {path}. >")
            Phi = np.array(pickle.load(f))
        assert len(Phi) == len(data), "The length of Phi is not equal to the length of the dataset. Use old main.py"
        Phi_test = Phi[test_indices]
        Phi = Phi[train_indices]
        Phi_test = Phi_test.T
        Phi = Phi.T
    else:
        raise ValueError("cfg.Phi_path is None.")

    if cfg.Phi_sft_path is not None:
        path = os.path.join(cfg.Phi_sft_path, "results/calculated_Phi.pkl")
        with open(path, "rb") as f:
            print(f"< Phi is loaded from {path}. >")
            Phi_sft = np.array(pickle.load(f))
        assert len(Phi_sft) == len(data), "The length of Phi is not equal to the length of the dataset. Use old main.py"
        Phi_test = Phi_sft[test_indices]
        Phi_sft = Phi_sft[train_indices]
        Phi_test = Phi_test.T
        Phi_sft = Phi_sft.T
    else:
        raise ValueError("cfg.Phi_path is None.")

    p = Poisoning(data=dataset_for_train_LP, cfg=cfg, Phi=Phi, Phi_sft=Phi_sft)

    if cfg.existing_r_o_path is not None:
        print(f"< Loading existing r_o model from {cfg.existing_r_o_path} >")
        p.load_r_o(cfg.existing_r_o_path)
    else:
        raise ValueError("cfg.existing_r_o_path is None.")

    #########################################################
    ####### Poisoning
    #########################################################

    cache_pkl["Phi"] = p.Phi
    cache_pkl["Phi_test"] = Phi_test
    if hasattr(p, "r_o_coef"):
        cache_pkl["r_o_coef"] = p.r_o_coef
    if hasattr(p, "r_a_coef"):
        cache_pkl["opt_r_coef"] = p.r_a_coef
    with open(f"{cfg.cache_dir}/cache.pkl", "wb") as f:
        pickle.dump(cache_pkl, f)

    if cfg.calculate_lambda_only:
        exit()

    prepoison_data = copy.deepcopy(p.data)
    prepoison_pref = prepoison_data["preference_prop"]

    print("< Preparing is completed. >")
    print("< Poisoning is in progress... >")

    p.poisoning()

    print("< Poisoning is completed. >")

    prepoison_data_path = f"{cfg.train_data_dir}/{prepoison_data.info.dataset_name}"
    prepoison_data.save_to_disk(prepoison_data_path)
    print(f"< The pre-poison data for LP is saved to {prepoison_data_path}. >")

    poisoned_data_path = f"{cfg.train_data_dir}/{p.data.info.dataset_name}"
    p.save_data_to_disk(poisoned_data_path)
    print(f"< The poisoned data for LP is saved to {poisoned_data_path}. >")

    result_json["result"] = p.result
    result_json["poisoned_pref"] = p.poisoned_pref
    result_json["target_prompts"] = target_prompts
    result_json["theta_A"] = p.theta_A
    result_json["prepoison_pref"] = prepoison_pref

    _result_json = {"keys": list(result_json.keys())}
    _result_json.update(result_json)
    result_json = _result_json

    with open(f"{cfg.results_dir}/poisoning.json", "w") as f:
        json.dump(result_json, f)

    result_pkl["LP_result"] = p.LP_result
    with open(f"{cfg.results_dir}/poisoning.pkl", "wb") as f:
        pickle.dump(result_pkl, f)

    print(f"< The results are saved in {cfg.results_dir}. >")


if __name__ == "__main__":
    main()

    print("< Finished! >")
