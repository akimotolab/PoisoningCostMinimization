# PoisoningCostMinimization
Source code to reproduce the experiments in "Cost-Minimized Label-Flipping Poisoning Attack to LLM Alignment" published at AAAI 2026

## Getting Started
### Setting Up
Firstly, set up a virtual environment and install the required packages. We recommend using uv run between 3.9 and 3.12.

```bash
$ uv sync
$ source .venv/bin/activate
```

### Preparing Datasets
Next, prepare the training and test datasets from ProlificAI/social-reasoning-rlhf, Rexhaif/hh-rlhf-chat-template and PKU-Alignment/PKU-SafeRLHF.

```bash
$ uv run explore_datasets.py
```

## Experiments
### 1. (Preparation) Training Phi and r_o models
    - Specify the dataset and model to use in conf/compute_Phi_and_r_o.yaml.
    Example:
        - dataset name="ProlificAI/social-reasoning-rlhf"
        - model name="microsoft/Phi-3.5-mini-instruct"
            ```yaml
            feature_extractor:
              provider: microsoft
              name: Phi-3.5-mini-instruct
            dataset:
              name: social-reasoning-rlhf
            ```
    - Execute compute_Phi_and_r_o.py
        ```bash
        $ uv run compute_Phi_and_r_o.py
        ```
    - A directory will be generated under outputs/Phi_r_o (let's call it "example_Phi_r_o")

### 2. Poisoning and cost minimization
    - Specify the path of the preparation data in conf/poison_data.yaml.
    Example:
        ```yaml
        existing_r_o_path: outputs/Phi_r_o/example_Phi_r_o
        Phi_path: outputs/Phi_r_o/example_Phi_r_o
        ```
    - Specify other poisoning-related settings in poison_data.yaml.
        - Data discretization granularity is controlled by duplication_n
    - Execute poison_data.py for general data or poison_social_data.py for social-reasoning-rlhf.
        ```bash
        $ uv run poison_data.py
        ```
        or for social-reasoning-rlhf:
        ```bash
        $ uv run poison_social_data.py
        ```

    - An experiment directory will be generated under outputs/ (let's call it example_exp). This contains poisoned data (train_data), experiment configuration files (.hydra), and poisoning results (results).

### 3. DPO execution
    - Specify the previous experiment directory in conf/train_dpo.yaml.
    Example:
        ```yaml
        poison_data_dir: outputs/example_exp
        ```
    - Specify other DPO settings in train_dpo.yaml.
        - To train only the final layer of the model, set dpo_configs.train_only_last_layer to True.
        - The granularity of training data used during DPO is specified by the value in example_exp/.hydra/omega_config.yaml.
    - Execute train_dpo.py
        ```bash
        $ uv run train_dpo.py
        ```
        - When specifying the experiment directory at runtime (i.e., specifying which poisoned data to use):
            ```bash
            $ uv run train_dpo.py --poison_data_dir=outputs/example_exp2
            ```
        - When you want to set train_only_last_layer to False at runtime:
            ```bash
            $ uv run train_dpo.py --adaptivePhi
            ```
        - When train_only_last_layer=False, they are generated under example_exp/dpo_model_adaptivePhi.

### 4. Evaluation
    - To measure performance loss, pass the model path and the dataset path to evaluate.py:
        ```bash
        $ uv run evaluate.py outputs/example_exp/dpo_model processed_datasets/pku_saferlhf_test.json
        ```
        or for social-reasoning-rlhf:
        ```bash
        $ uv run evaluate_social.py outputs/example_exp/dpo_model
        ```
        - This will generate results (single_model_gaps.json) under example_exp/eval.

    - To measure the model's output length, pass the model path and the dataset path to modelreply.py:
        ```bash
        $ uv run modelreply.py outputs/example_exp/dpo_model processed_datasets/pku_saferlhf_test.json [--batch_size 100] [--max_new_tokens 512]
        ```
        - This will generate results under example_exp/eval.
        or for social-reasoning-rlhf:
        ```bash
        $ uv run modelreply_social.py outputs/example_exp/dpo_model_adaptivePhi [--batch_size 100] [--max_new_tokens 512]
        ```
        - This will generate results under example_exp/dpo_model_adaptivePhi/eval.
