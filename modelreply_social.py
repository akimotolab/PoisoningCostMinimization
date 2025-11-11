import os
import sys
import random
import numpy as np
import torch
import argparse
import json
import logging
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM


# ===== Seed fixing for reproducibility =====
SEED = 0
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cuda.matmul.allow_tf32 = False
# ==================================

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def setup_logging(model_path):
    """Initialize logging configuration"""
    log_dir = os.path.join(model_path, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "generation_outputs.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )

    logger = logging.getLogger(__name__)
    return logger


def get_save_path(model_path, dataset_path, custom_name=None):
    """Generate save path for results"""
    eval_dir = os.path.join(model_path, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    model_name = os.path.basename(os.path.normpath(model_path))
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    file_name = f"generation_lengths_{model_name}_{dataset_name}"
    if custom_name:
        file_name += f"_{custom_name}"
    return os.path.join(eval_dir, file_name + ".json")


def analyze_generation_length(model_path, dataset_path, batch_size, max_new_tokens, save_path, test_size=0.2, seed=0):
    logger = logging.getLogger(__name__)
    logger.info("=== Generation evaluation started ===")
    logger.info(f"Model path: {model_path}")

    # Fix dataset path
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"test_size: {test_size}, seed: {seed}")
    logger.info(f"Batch size: {batch_size}, max_new_tokens: {max_new_tokens}")
    logger.info(f"Result save path: {save_path}")

    # --- 1. Load model and tokenizer ---
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", max_memory={0: "37GB", 1: "78GB"})
    logger.info(f"Model loading completed: {model.device}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Settings for decoder-only models
    tokenizer.padding_side = "left"
    logger.info("Tokenizer loading completed")

    # --- 2. Load dataset and existing results ---
    with open(dataset_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)["data"]

    all_prompts = [item["prompt"] for item in all_data]
    total_samples = len(all_prompts)

    # train/test split (same method as poison_data.py)
    logger.info(f"Total data count: {total_samples}")

    # Fix seed
    random.seed(seed)
    np.random.seed(seed)

    # Shuffle indices
    indices = list(range(total_samples))
    random.shuffle(indices)

    # Calculate split point
    split_point = int(total_samples * (1 - test_size))
    test_indices = indices[split_point:]

    # Extract only test data
    test_prompts = [all_prompts[i] for i in test_indices]
    test_indices_original = test_indices  # Keep original indices

    results = {
        "metadata": {
            "model_path": model_path,
            "dataset_path": dataset_path,
            "total_samples": len(test_prompts),
            "test_size": test_size,
            "seed": seed,
            "test_indices": test_indices_original,  # Save original indices
        },
        "generations": {},  # Manage results as dictionary with index as key
    }

    if os.path.exists(save_path):
        logger.info(f"Loading existing result file: {save_path}")
        with open(save_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        # Convert from old format (lengths, all_generations) to new format (generations)
        if "lengths" in results and "all_generations" in results and "generations" not in results:
            logger.info("Converting old format JSON file to new format")
            old_lengths = results.get("lengths", [])
            old_generations = results.get("all_generations", [])

            # Convert to new format
            results = {
                "metadata": {
                    "model_path": model_path,
                    "dataset_path": dataset_path,
                    "total_samples": len(test_prompts),
                    "test_size": test_size,
                    "seed": seed,
                    "test_indices": test_indices_original,
                    "converted_from_old_format": True,
                },
                "generations": {},
            }

            # Convert old data to new format
            for i, (length, generation) in enumerate(zip(old_lengths, old_generations)):
                # Check if it ends with punctuation
                ends_with_punctuation = generation.strip().endswith((".", "?", "!", "。", "？", "！"))

                results["generations"][str(i)] = {
                    "prompt": "",  # Old format doesn't have prompts
                    "generated_text": generation,
                    "generated_char_count": length,
                    "generated_token_count": 0,  # Set to 0 since accurate token count is unknown in old data
                    "max_new_tokens_setting": 8192,  # Default value
                    "ends_with_punctuation": ends_with_punctuation,  # Whether it ends with punctuation
                }

            logger.info(f"Conversion completed: {len(results['generations'])} samples converted")

    # --- 3. Identify unprocessed prompts and rerun targets ---
    processed_indices = {int(k) for k in results.get("generations", {}).keys()}
    logger.info(f"Existing processed sample count: {len(processed_indices)}")

    # Identify indices that need rerun
    needs_rerun_indices = set()
    for idx_str, data in results.get("generations", {}).items():
        idx = int(idx_str)
        # 1. Generated token count equals setting limit (reached token limit)
        if data.get("generated_token_count", 0) == data.get("max_new_tokens_setting", 0):
            needs_rerun_indices.add(idx)
            logger.debug(f"Rerun target (token limit): index {idx}")
        # 2. Doesn't end with punctuation (incomplete sentence)
        elif data.get("ends_with_punctuation") == False:
            needs_rerun_indices.add(idx)
            logger.debug(f"Rerun target (incomplete sentence): index {idx}")
        # 3. Old data with token count of 0 (accurate token count unknown)
        elif data.get("generated_token_count", 0) == 0:
            needs_rerun_indices.add(idx)
            logger.debug(f"Rerun target (old data): index {idx}")

    # Identify unprocessed indices
    unprocessed_indices = set(range(len(test_prompts))) - processed_indices
    logger.info(f"Unprocessed sample count: {len(unprocessed_indices)}")

    # Determine indices to process in this run
    # (unprocessed ones + ones that need rerun)
    indices_to_process = sorted(list(unprocessed_indices | needs_rerun_indices))

    if not indices_to_process:
        logger.info("All samples are processed. Ending process.")
        return

    logger.info(f"Samples to process this time: {len(indices_to_process)}")
    if needs_rerun_indices:
        token_limit_count = len(
            [
                i
                for i in needs_rerun_indices
                if results["generations"].get(str(i), {}).get("generated_token_count", 0)
                == results["generations"].get(str(i), {}).get("max_new_tokens_setting", 0)
            ]
        )
        incomplete_count = len(
            [i for i in needs_rerun_indices if results["generations"].get(str(i), {}).get("ends_with_punctuation") == False]
        )
        old_data_count = len(
            [i for i in needs_rerun_indices if results["generations"].get(str(i), {}).get("generated_token_count", 0) == 0]
        )
        logger.info(
            f"Rerun targets: token limit reached={token_limit_count}, incomplete sentences={incomplete_count}, old data={old_data_count}"
        )

    if unprocessed_indices:
        logger.info(f"Unprocessed samples: {len(unprocessed_indices)}")
        logger.info(f"First 10 unprocessed samples: {sorted(list(unprocessed_indices))[:10]}")

    prompts_to_process = [(i, test_prompts[i]) for i in indices_to_process]

    # --- 4. Generation loop (save by batch) ---
    try:
        logger.info(f"Batch processing started: {len(prompts_to_process)} samples processed in batches of {batch_size}")

        for i in tqdm(range(0, len(prompts_to_process), batch_size), desc="eval", ncols=80):
            batch_data = prompts_to_process[i : i + batch_size]
            batch_indices = [item[0] for item in batch_data]
            batch_prompts = [item[1] for item in batch_data]

            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )

            # Get generated results excluding prompt part
            input_token_lengths = inputs.input_ids.shape[1]
            generated_token_ids = output_ids[:, input_token_lengths:]

            outputs = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)

            # --- 5. Save results (by batch) ---
            batch_updated = False
            for j, original_index in enumerate(batch_indices):
                generated_part = outputs[j].lstrip()
                ends_with_punctuation = generated_part.strip().endswith((".", "?", "!", "。", "？", "！"))
                gen_ids = generated_token_ids[j]
                if tokenizer.pad_token_id is not None:
                    gen_ids = gen_ids[gen_ids != tokenizer.pad_token_id]
                generated_token_count = len(gen_ids)
                results["generations"][str(original_index)] = {
                    "prompt": batch_prompts[j],
                    "generated_text": generated_part,
                    "generated_char_count": len(generated_part),
                    "generated_token_count": generated_token_count,
                    "max_new_tokens_setting": max_new_tokens,
                    "ends_with_punctuation": ends_with_punctuation,
                }
                batch_updated = True

            if batch_updated:
                # Calculate real-time statistics
                all_lengths = [data["generated_char_count"] for data in results["generations"].values()]
                current_avg_len = float(np.mean(all_lengths)) if all_lengths else 0.0
                processed_count = len(all_lengths)

                # Update metadata
                results["metadata"]["average_length"] = current_avg_len
                results["metadata"]["processed_samples"] = processed_count
                results["metadata"]["last_updated_batch"] = i // batch_size
                results["metadata"]["last_updated_index"] = batch_indices[-1] if batch_indices else 0

                # Save to JSON file
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

                # Log progress
                logger.info(
                    f"Batch {i // batch_size + 1} completed: {processed_count}/{len(test_prompts)} samples processed, average length: {current_avg_len:.2f}"
                )

    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA Out of Memory error occurred. Please try again with a smaller batch size.")
        logger.error(f"Error details: {e}")
        logger.info(f"Results up to the error point are saved in {save_path}")
        sys.exit(1)  # Exit script
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        logger.info(f"Results up to the error point are saved in {save_path}")
        sys.exit(1)

    # --- 6. Calculate and save final statistics ---
    all_lengths = [data["generated_char_count"] for data in results["generations"].values()]
    avg_len = float(np.mean(all_lengths)) if all_lengths else 0.0
    results["metadata"]["average_length"] = avg_len
    results["metadata"]["processed_samples"] = len(all_lengths)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Final average generated character count: {avg_len:.2f}")
    logger.info(f"Final results saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to evaluate LLM generation length (for social reasoning)")
    parser.add_argument("model_path", type=str, help="Model path")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum generation tokens")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test dataset size ratio (default: 0.2)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for train/test split (default: 0)")
    parser.add_argument("--savename_suffix", type=str, default="test", help="Suffix to add to save filename (e.g., pass1)")

    args = parser.parse_args()

    assert os.path.isdir(args.model_path), "directory doesnt exist"

    # Logging setup
    logger = setup_logging(args.model_path)

    dataset_path = "processed_datasets/social_reasoning_rlhf_train.json"
    save_path = get_save_path(args.model_path, dataset_path, args.savename_suffix)

    analyze_generation_length(
        args.model_path, dataset_path, args.batch_size, args.max_new_tokens, save_path, args.test_size, args.seed
    )

    logger.info("=== Generation evaluation completed ===")
