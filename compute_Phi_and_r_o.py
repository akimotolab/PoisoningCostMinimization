import hydra
from omegaconf import DictConfig
import json
import os

from train_reward import FeatureExtractor
from poisoning import feature_extract, train_r_o
from utils import load_unified_dataset


@hydra.main(config_path="conf", config_name="compute_Phi_and_r_o", version_base=None)
def main(cfg: DictConfig):
    feature_extractor_name = cfg.feature_extractor.provider + "/" + cfg.feature_extractor.name

    #########################################################
    ####### Preprocessing Data
    #########################################################

    data = load_unified_dataset(cfg.dataset)

    print("< The data is loaded. >")

    print("< Computing Phi is in progress... >")

    feature_extract(data=data, feature_extractor=FeatureExtractor(feature_extractor_name), save_dir=cfg.working_dir)

    print("< Training r_o model is in progress... >")

    train_r_o(data, feature_extractor_name, cfg.only_last_layer, save_dir=cfg.working_dir)

    dataset_info = {
        "dataset_name": data.info.dataset_name if hasattr(data.info, "dataset_name") else cfg.dataset.name,
        "dataset_size": len(data),
        "feature_extractor": feature_extractor_name,
        "working_dir": cfg.working_dir,
    }

    with open(os.path.join(cfg.working_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    print(f"< Preprocessing is completed. (saved in {cfg.working_dir}) >")


if __name__ == "__main__":
    main()

    print("< Finished! >")
