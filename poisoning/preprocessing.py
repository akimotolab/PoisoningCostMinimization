import os
import time
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig
from utils import rename_columns


DIR = "results"


def feature_extract(data, feature_extractor, save_dir=None):
    chosen = data["chosen"]
    rejected = data["rejected"]
    x_label = "prompt" if "prompt" in data.column_names else "question"
    prompt = data[x_label]
    Phi = []

    start_time = time.time()

    for i in range(len(chosen)):
        Phi.append(
            (
                feature_extractor.extract_feature(prompt[i] + " " + chosen[i])[0][0][-1]
                - feature_extractor.extract_feature(prompt[i] + " " + rejected[i])[0][0][-1]
            )
            .cpu()
            .detach()
            .numpy()
        )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"< Computing Phi is completed ({elapsed_time} s) >")

    if save_dir is not None:
        save_dir = os.path.join(save_dir, DIR)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_path = os.path.join(save_dir, "calculated_Phi.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(Phi, f)
        print(f"< Phi is saved in {file_path} >")

    return np.array(Phi).transpose()


def train_r_o(dataset_for_rm, feature_extractor, only_last_layer, save_dir=None):
    model = AutoModelForSequenceClassification.from_pretrained(feature_extractor, device_map="auto", num_labels=1)
    for param in model.parameters():
        param.requires_grad = False if only_last_layer else True
    for param in model.score.parameters():
        param.requires_grad = True

    tokenizer = AutoTokenizer.from_pretrained(feature_extractor)
    tokenizer.pad_token = tokenizer.eos_token
    training_args = RewardConfig(
        output_dir="./outputs",
        center_rewards_coefficient=0.01,
        torch_compile=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
    )

    def preprocess_for_reward_modelling(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        question_key = "question" if "question" in examples else "prompt"
        for chosen, rejected, question in zip(examples["chosen"], examples["rejected"], examples[question_key]):
            tokenized_j = tokenizer(question + " " + chosen, truncation=True)
            tokenized_k = tokenizer(question + " " + rejected, truncation=True)

            new_examples["input_ids_chosen"].append(tokenized_j["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_j["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_k["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_k["attention_mask"])

        return new_examples

    if "chosen" not in dataset_for_rm.column_names:
        dataset_for_rm = rename_columns(dataset_for_rm)
    dataset_for_rm = dataset_for_rm.map(
        preprocess_for_reward_modelling,
        batched=True,
        num_proc=4,
    )
    trainer = RewardTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset_for_rm, max_length=512, args=training_args)
    trainer.train()

    r_o_coef = model.score.weight[0].cpu().detach().numpy()

    if save_dir is not None:
        save_path = os.path.join(save_dir, "r_o_model")
        os.makedirs(save_path)
        trainer.save_model(save_path)
        print(f"< r_o model is saved in {save_path} >")

        save_dir = os.path.join(save_dir, DIR)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, "r_o_coef.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(r_o_coef, f)
        print(f"< r_o_coef is saved in {save_path} >")

    return model, r_o_coef
