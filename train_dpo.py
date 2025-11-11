import os
import time
import argparse
from omegaconf import OmegaConf, DictConfig
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig

from utils import set_seed, data_for_RLHF

MODEL_NAME = "dpo_model"


def update_duplicate_num(x):
    x["number"] = cfg.duplication_n
    return x


def main(cfg: DictConfig):
    print(f"< Exp name: {cfg.experiment_id} >")

    model_saved_to = os.path.join(cfg.working_dir, MODEL_NAME)
    if not os.path.exists(model_saved_to):
        os.makedirs(model_saved_to)
    else:
        raise FileExistsError(f"Trained model {model_saved_to} already exists.")

    set_seed(cfg.seed)

    #########################################################
    ####### DPO
    #########################################################
    print(
        f"< Training DPO model with the poisoned dataset is in progress... (train_only_last_layer is {cfg.dpo_configs.train_only_last_layer}) >"
    )
    start_time = time.time()

    # Load the poisoned data
    dataset_for_LP = load_from_disk(f"{cfg.train_data_dir}/poisoned_data_LP")

    dataset_for_LP = dataset_for_LP.map(update_duplicate_num)
    dataset_for_training = data_for_RLHF(dataset_for_LP, cfg.experiment_id)

    # Load the model
    feature_extractor = cfg.feature_extractor.provider + "/" + cfg.feature_extractor.name
    print(f"< Loading model from {feature_extractor} >")
    model = AutoModelForCausalLM.from_pretrained(feature_extractor, trust_remote_code=True, device_map="auto")

    for param in model.parameters():
        param.requires_grad = False if cfg.dpo_configs.train_only_last_layer else True
    for param in model.lm_head.parameters():
        param.requires_grad = True

    # Train
    training_args = DPOConfig(
        beta=cfg.dpo_configs.beta,
        output_dir="temp",
        torch_compile=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
    )

    peft_config = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["o_proj", "qkv_proj", "gate_up_proj", "down_proj"],
    )

    tokenizer = AutoTokenizer.from_pretrained(feature_extractor)
    tokenizer.pad_token = tokenizer.eos_token

    dpo_trainer = DPOTrainer(
        model,
        args=training_args,
        train_dataset=dataset_for_training,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )
    dpo_trainer.train()

    # Save the model and train data

    data_saved_to = os.path.join(model_saved_to, "poisoned_data_RLHF")
    cfg_saved_to = os.path.join(model_saved_to, "config.yaml")
    dpo_trainer.save_model(model_saved_to)
    dataset_for_training.save_to_disk(data_saved_to)
    OmegaConf.save(config=cfg, f=cfg_saved_to)

    print(f"< Trained model and data are saved in {model_saved_to} >")

    end_time = time.time()
    print(f"< Training DPO model is completed. ({end_time - start_time} s) >")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--poison_data_dir", type=str, required=False)
    parser.add_argument("--adaptivePhi", action="store_true", required=False)
    args = parser.parse_args()

    dpo_cfg_path = "conf/train_dpo.yaml"
    dpo_cfg = OmegaConf.load(dpo_cfg_path)

    if args.poison_data_dir is not None:
        dpo_cfg.poison_data_dir = args.poison_data_dir

    model_data_cfg_path = os.path.join(dpo_cfg.poison_data_dir, ".hydra/model_data_config.yaml")
    model_data_cfg = OmegaConf.load(model_data_cfg_path)

    poison_cfg_path = os.path.join(dpo_cfg.poison_data_dir, ".hydra/omega_config.yaml")
    poison_cfg = OmegaConf.load(poison_cfg_path)

    # overwrite
    if args.adaptivePhi:
        dpo_cfg.dpo_configs.train_only_last_layer = not args.adaptivePhi

    if not dpo_cfg.dpo_configs.train_only_last_layer:
        MODEL_NAME += "_adaptivePhi"

    cfg = OmegaConf.merge(model_data_cfg, poison_cfg, dpo_cfg)

    main(cfg)

    print("< Finished! >")
