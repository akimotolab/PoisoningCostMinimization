import time
import torch
from transformers import AutoModelForCausalLM


def compute_single_model_gap(model_path, ref_model_path, cost_model_path, dataset, tokenizer):
    """
    Function to load 2+1 models and compute reward gaps.
    """
    assert "prompt" in dataset.column_names or "question" in dataset.column_names
    x_label = "prompt" if "prompt" in dataset.column_names else "question"

    # Data preprocessing (tokenization)
    prompt_idx_list = []
    chosen_idx_list = []
    rejected_idx_list = []
    for d in dataset:
        p_idx = tokenizer(d[x_label], return_tensors="pt", add_special_tokens=True).input_ids
        c_idx = tokenizer(d["chosen"], return_tensors="pt", add_special_tokens=True).input_ids
        r_idx = tokenizer(d["rejected"], return_tensors="pt", add_special_tokens=True).input_ids
        prompt_idx_list.append(p_idx)
        chosen_idx_list.append(c_idx)
        rejected_idx_list.append(r_idx)

    print("< Calculating reward gaps with 2+1 model strategy... >")
    start_time = time.time()

    main_chosen_logprobs, main_rejected_logprobs = [], []
    ref_chosen_logprobs, ref_rejected_logprobs = [], []
    cost_chosen_logprobs, cost_rejected_logprobs = [], []

    # --- Step 1: Load 2 models ---
    print(f"--- Loading main and ref models: {model_path}, {ref_model_path} ---")
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to("cuda")
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_path, trust_remote_code=True).to("cuda")
    model.eval()
    ref_model.eval()

    with torch.no_grad():
        for prompt_idx, chosen_idx, rejected_idx in zip(prompt_idx_list, chosen_idx_list, rejected_idx_list):
            prompt_idx, chosen_idx, rejected_idx = prompt_idx.to("cuda"), chosen_idx.to("cuda"), rejected_idx.to("cuda")

            # --- Chosen ---
            prmpt_chosen = torch.concat([prompt_idx, chosen_idx], dim=-1).long()
            # main model
            c_logits_main = model(prmpt_chosen).logits
            c_logprobs_main = torch.gather(
                c_logits_main[:, prompt_idx.shape[-1] - 1 : -1, :].log_softmax(dim=-1), dim=-1, index=chosen_idx[:, :, None]
            ).squeeze(-1)
            main_chosen_logprobs.append(c_logprobs_main.sum(dim=-1).cpu())
            # ref model
            c_logits_ref = ref_model(prmpt_chosen).logits
            c_logprobs_ref = torch.gather(
                c_logits_ref[:, prompt_idx.shape[-1] - 1 : -1, :].log_softmax(dim=-1), dim=-1, index=chosen_idx[:, :, None]
            ).squeeze(-1)
            ref_chosen_logprobs.append(c_logprobs_ref.sum(dim=-1).cpu())

            # --- Rejected ---
            prmpt_rejected = torch.concat([prompt_idx, rejected_idx], dim=-1).long()
            # main model
            r_logits_main = model(prmpt_rejected).logits
            r_logprobs_main = torch.gather(
                r_logits_main[:, prompt_idx.shape[-1] - 1 : -1, :].log_softmax(dim=-1), dim=-1, index=rejected_idx[:, :, None]
            ).squeeze(-1)
            main_rejected_logprobs.append(r_logprobs_main.sum(dim=-1).cpu())
            # ref model
            r_logits_ref = ref_model(prmpt_rejected).logits
            r_logprobs_ref = torch.gather(
                r_logits_ref[:, prompt_idx.shape[-1] - 1 : -1, :].log_softmax(dim=-1), dim=-1, index=rejected_idx[:, :, None]
            ).squeeze(-1)
            ref_rejected_logprobs.append(r_logprobs_ref.sum(dim=-1).cpu())

    print("--- Unloading main and ref models ---")
    del model, ref_model
    torch.cuda.empty_cache()

    # --- Step 2: Load the remaining model ---
    print(f"--- Loading cost model: {cost_model_path} ---")
    cost_model = AutoModelForCausalLM.from_pretrained(cost_model_path, trust_remote_code=True).to("cuda")
    cost_model.eval()

    with torch.no_grad():
        for prompt_idx, chosen_idx, rejected_idx in zip(prompt_idx_list, chosen_idx_list, rejected_idx_list):
            prompt_idx, chosen_idx, rejected_idx = prompt_idx.to("cuda"), chosen_idx.to("cuda"), rejected_idx.to("cuda")
            # Chosen
            prmpt_chosen = torch.concat([prompt_idx, chosen_idx], dim=-1).long()
            c_logits_cost = cost_model(prmpt_chosen).logits
            c_logprobs_cost = torch.gather(
                c_logits_cost[:, prompt_idx.shape[-1] - 1 : -1, :].log_softmax(dim=-1), dim=-1, index=chosen_idx[:, :, None]
            ).squeeze(-1)
            cost_chosen_logprobs.append(c_logprobs_cost.sum(dim=-1).cpu())
            # Rejected
            prmpt_rejected = torch.concat([prompt_idx, rejected_idx], dim=-1).long()
            r_logits_cost = cost_model(prmpt_rejected).logits
            r_logprobs_cost = torch.gather(
                r_logits_cost[:, prompt_idx.shape[-1] - 1 : -1, :].log_softmax(dim=-1), dim=-1, index=rejected_idx[:, :, None]
            ).squeeze(-1)
            cost_rejected_logprobs.append(r_logprobs_cost.sum(dim=-1).cpu())

    print("--- Unloading cost model ---")
    del cost_model
    torch.cuda.empty_cache()

    # --- Step 3: Combine all calculation results ---
    reward_gaps = []
    reward_gaps_ref = []
    reward_gaps_cost = []

    for i in range(len(main_chosen_logprobs)):
        gap = main_chosen_logprobs[i] - main_rejected_logprobs[i]
        gap_ref = ref_chosen_logprobs[i] - ref_rejected_logprobs[i]
        gap_cost = cost_chosen_logprobs[i] - cost_rejected_logprobs[i]

        reward_gaps.append(torch.sigmoid(gap).item())
        reward_gaps_ref.append(torch.sigmoid(gap_ref).item())
        reward_gaps_cost.append(torch.sigmoid(gap_cost).item())

    end_time = time.time()
    print(f"< Finished all calculations ({end_time - start_time:.2f} s) >")
    return reward_gaps, reward_gaps_ref, reward_gaps_cost
