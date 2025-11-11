import json
from omegaconf import ListConfig


def select_target_prompts_by_keyword(data, key_word="target"):
    """
    Select target prompts from the data.
    """
    x_label = "prompt" if "prompt" in data.column_names else "question"
    prompts = data[x_label]

    if isinstance(key_word, str):
        key_word = [key_word]
    elif isinstance(key_word, ListConfig):
        key_word = list(key_word)
    assert isinstance(key_word, list)

    target_prompts = [False for _ in range(len(prompts))]
    for word in key_word:
        for i in range(len(prompts)):
            if word in prompts[i]:
                target_prompts[i] = True
    return target_prompts


def select_target_prompts_by_category(dataset_name="ProlificAI_social-reasoning-rlhf", category="target", split="train"):
    """
    Select target prompts from the data.
    """
    with open("dataset_labels/" + dataset_name.replace("/", "_") + "_" + split + ".json", "r") as file:
        data = json.load(file)
        eval_clses_chosen = data["eval_clses_chosen"]
        eval_clses_rejected = data["eval_clses_rejected"]

    target_prompts = []

    for cls_chosen, cls_rejected in zip(eval_clses_chosen, eval_clses_rejected):
        if cls_chosen == category or cls_rejected == category:
            target_prompts.append(True)
        else:
            target_prompts.append(False)
    return target_prompts
