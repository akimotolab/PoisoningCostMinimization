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

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def setup_logging(model_path):
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
    eval_dir = os.path.join(os.path.dirname(os.path.normpath(model_path)), "eval")
    os.makedirs(eval_dir, exist_ok=True)
    model_name = os.path.basename(os.path.normpath(model_path))
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    file_name = f"generation_lengths_{model_name}_{dataset_name}"
    if custom_name:
        file_name += f"_{custom_name}"
    return os.path.join(eval_dir, file_name + ".json")


def analyze_generation_length(model_path, dataset_path, batch_size, max_new_tokens, save_path):
    logger = logging.getLogger(__name__)
    logger.info(f"Model path: {model_path}")
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Batch size: {batch_size}, max_new_tokens: {max_new_tokens}")
    logger.info(f"Result save path: {save_path}")

    # --- 1. Model and tokenizer loading ---
    logger.info("Model loading starts...")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    logger.info(f"Model loading completed: {model.device}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    logger.info("Tokenizer loading completed")

    # --- 2. Dataset and existing results loading ---
    with open(dataset_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)["data"]

    all_prompts = [item["prompt"] for item in all_data]
    total_samples = len(all_prompts)

    results = {
        "metadata": {
            "model_path": model_path,
            "dataset_path": dataset_path,
            "total_samples": total_samples,
        },
        "generations": {},
    }

    if os.path.exists(save_path):
        logger.info(f"Loading existing results file: {save_path}")
        with open(save_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        if "lengths" in results and "all_generations" in results and "generations" not in results:
            logger.info("Converting old JSON file to new format")
            old_lengths = results.get("lengths", [])
            old_generations = results.get("all_generations", [])

            results = {
                "metadata": {
                    "model_path": model_path,
                    "dataset_path": dataset_path,
                    "total_samples": total_samples,
                    "converted_from_old_format": True,
                },
                "generations": {},
            }

            for i, (length, generation) in enumerate(zip(old_lengths, old_generations)):
                ends_with_punctuation = generation.strip().endswith((".", "?", "!", "。", "？", "！"))

                results["generations"][str(i)] = {
                    "prompt": "",
                    "generated_text": generation,
                    "generated_char_count": length,
                    "generated_token_count": 0,
                    "max_new_tokens_setting": 128,
                    "ends_with_punctuation": ends_with_punctuation,
                }

            logger.info(f"Conversion completed: {len(results['generations'])} samples converted")

    processed_indices = {int(k) for k in results.get("generations", {}).keys()}
    logger.info(f"Number of processed samples: {len(processed_indices)}")

    indices_to_process = set()
    for idx_str, data in results.get("generations", {}).items():
        idx = int(idx_str)
        gen_count = data.get("generated_token_count", 0)

        if gen_count == 0:
            indices_to_process.add(idx)

    indices_to_process = sorted(list(indices_to_process))

    if not indices_to_process:
        logger.info("No samples with generated token count 0 or unprocessed samples. Processing terminated.")
        return

    logger.info(f"Number of samples to process this time: {len(indices_to_process)}")

    zero_token_count = len(indices_to_process)

    if zero_token_count > 0:
        logger.info(f"Re-execution target (generated token count 0): {zero_token_count} samples")

    prompts_to_process = [(i, all_prompts[i]) for i in indices_to_process]

    try:
        logger.info(f"Batch processing starts: {len(prompts_to_process)} samples in batches of {batch_size}")

        for i in tqdm(range(0, len(prompts_to_process), batch_size), desc="eval", ncols=80):
            batch_data = prompts_to_process[i : i + batch_size]
            batch_indices = [item[0] for item in batch_data]
            batch_prompts = [item[1] for item in batch_data]

            if i == 0:
                logger.info(f"First batch indices: {batch_indices[:5]}")

            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )

            input_token_lengths = inputs.input_ids.shape[1]
            generated_token_ids = output_ids[:, input_token_lengths:]

            outputs = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)

            batch_updated = False
            for j, original_index in enumerate(batch_indices):
                generated_part = outputs[j].lstrip()
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
                }
                batch_updated = True

            if batch_updated:
                all_lengths = [data["generated_char_count"] for data in results["generations"].values()]
                current_avg_len = float(np.mean(all_lengths)) if all_lengths else 0.0
                processed_count = len(all_lengths)

                results["metadata"]["average_length"] = current_avg_len
                results["metadata"]["processed_samples"] = processed_count
                results["metadata"]["last_updated_batch"] = i // batch_size
                results["metadata"]["last_updated_index"] = batch_indices[-1] if batch_indices else 0

                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

                logger.info(
                    f"Batch {i // batch_size + 1} completed: {processed_count}/{total_samples} samples processed, average length: {current_avg_len:.2f}"
                )

    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA Out of Memory error occurred. Please reduce the batch size and try again.")
        logger.error(f"Error details: {e}")
        logger.info(f"Results up to the error point are saved in {save_path}.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        logger.info(f"Results up to the error point are saved in {save_path}.")
        sys.exit(1)

    all_lengths = [data["generated_char_count"] for data in results["generations"].values()]
    avg_len = float(np.mean(all_lengths)) if all_lengths else 0.0
    results["metadata"]["average_length"] = avg_len
    results["metadata"]["processed_samples"] = len(all_lengths)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Final average generation length: {avg_len:.2f}")
    logger.info(f"Final results are saved in {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to evaluate the generation length of LLM")
    parser.add_argument("model_path", type=str, help="Model path")
    parser.add_argument("dataset_path", type=str, help="Dataset path")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of generated tokens")
    parser.add_argument("--savename_suffix", type=str, default="", help="Suffix to add to the save file name (e.g. pass1)")

    args = parser.parse_args()

    logger = setup_logging(args.model_path)

    save_path = get_save_path(args.model_path, args.dataset_path, args.savename_suffix)

    analyze_generation_length(args.model_path, args.dataset_path, args.batch_size, args.max_new_tokens, save_path)

    logger.info("=== Generation evaluation completed ===")
