#!/usr/bin/env python3
"""
Script to explore RLHF dataset structure
"""

from datasets import load_dataset
import json
import os


def explore_hh_rlhf_chat_template():
    """Explore Rexhaif/hh-rlhf-chat-template dataset"""
    print("=" * 60)
    print("📊 Rexhaif/hh-rlhf-chat-template Dataset")
    print("=" * 60)

    # Load dataset
    dataset = load_dataset("Rexhaif/hh-rlhf-chat-template")

    print(f"Dataset structure: {dataset}")
    print(f"Training data size: {len(dataset['train'])}")
    print(f"Test data size: {len(dataset['test'])}")

    # Check first example
    first_example = dataset["train"][0]
    print("\n📝 First example:")
    print(f"  Keys: {list(first_example.keys())}")
    print(f"  Conversation length: {len(first_example['conversation'])}")
    print(f"  Chosen keys: {list(first_example['chosen'].keys())}")
    print(f"  Rejected keys: {list(first_example['rejected'].keys())}")

    # Show conversation example
    print("\n💬 Conversation example:")
    for i, turn in enumerate(first_example["conversation"][:2]):
        print(f"  Turn {i}: {turn}")

    print(f"\n✅ Chosen: {first_example['chosen']}")
    print(f"❌ Rejected: {first_example['rejected']}")

    return dataset


def explore_social_reasoning_rlhf():
    """Explore ProlificAI/social-reasoning-rlhf dataset"""
    print("\n" + "=" * 60)
    print("📊 ProlificAI/social-reasoning-rlhf Dataset")
    print("=" * 60)

    # Load dataset
    dataset = load_dataset("ProlificAI/social-reasoning-rlhf")

    print(f"Dataset structure: {dataset}")
    print(f"Data size: {len(dataset['train'])}")

    # Check first example
    first_example = dataset["train"][0]
    print("\n📝 First example:")
    print(f"  Keys: {list(first_example.keys())}")

    # Display content
    for key, value in first_example.items():
        if isinstance(value, str) and len(value) > 200:
            print(f"  {key}: {value[:200]}...")
        else:
            print(f"  {key}: {value}")

    return dataset


def explore_pku_saferlhf():
    """Explore PKU-Alignment/PKU-SafeRLHF dataset"""
    print("\n" + "=" * 60)
    print("📊 PKU-Alignment/PKU-SafeRLHF Dataset")
    print("=" * 60)

    # Load dataset
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF")

    print(f"Dataset structure: {dataset}")
    print(f"Default data size: {len(dataset['train'])}")

    # Check first example
    first_example = dataset["train"][0]
    print("\n📝 First example:")
    print(f"  Keys: {list(first_example.keys())}")

    # Display content
    for key, value in first_example.items():
        if isinstance(value, str) and len(value) > 200:
            print(f"  {key}: {value[:200]}...")
        elif isinstance(value, dict):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")

    # Check subsets
    print("\n📊 Available subsets:")
    subsets = ["default", "alpaca-7b", "alpaca2-7b", "alpaca3-8b"]
    for subset in subsets:
        try:
            subset_data = load_dataset("PKU-Alignment/PKU-SafeRLHF", name=subset)
            print(f"  {subset}: train={len(subset_data['train'])}, test={len(subset_data['test'])}")
        except Exception as e:
            print(f"  {subset}: Error - {e}")

    return dataset


def convert_to_social_reasoning_format(item, source_dataset, item_id):
    """Convert data to unified format (minimal structure)"""

    if source_dataset == "PKU-SafeRLHF":
        # For PKU-SafeRLHF
        return {
            "prompt": item["prompt"],
            "chosen": item["response_0"] if item["better_response_id"] == 0 else item["response_1"],
            "rejected": item["response_1"] if item["better_response_id"] == 0 else item["response_0"],
        }
    elif source_dataset == "social-reasoning-rlhf":
        # For social-reasoning format
        return {"prompt": item["question"], "chosen": item["chosen"], "rejected": item["rejected"]}
    elif source_dataset == "hh-rlhf-chat-template":
        # For hh-rlhf format
        # Use the last user message in conversation as prompt
        user_messages = [turn for turn in item["conversation"] if turn["role"] == "user"]
        prompt = user_messages[-1]["content"] if user_messages else ""

        return {"prompt": prompt, "chosen": item["chosen"]["content"], "rejected": item["rejected"]["content"]}

    return None


def save_unified_format(datasets_info, output_dir="processed_datasets"):
    """Save datasets in unified format"""
    print("\n" + "=" * 60)
    print("💾 Saving data in unified format")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    for dataset_name, dataset_info in datasets_info.items():
        print(f"\n🔄 Processing: {dataset_name}")

        dataset = dataset_info["dataset"]
        source_name = dataset_info["source"]

        # Process training and test data
        for split in ["train", "test"]:
            if split not in dataset:
                continue

            converted_data = []
            split_data = dataset[split]

            print(f"  Converting {split} data... ({len(split_data)} items)")

            for i, item in enumerate(split_data):
                converted_item = convert_to_social_reasoning_format(item, source_name, f"{dataset_name}_{split}_{i}")
                if converted_item:
                    converted_data.append(converted_item)

            # Save as JSON file
            output_file = os.path.join(output_dir, f"{dataset_name}_{split}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "dataset_name": dataset_name,
                        "split": split,
                        "source": source_name,
                        "total_items": len(converted_data),
                        "data": converted_data,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            print(f"  ✅ Save completed: {output_file} ({len(converted_data)} items)")


def compare_formats():
    """Compare formats of two datasets"""
    print("\n" + "=" * 60)
    print("🔄 Dataset format comparison")
    print("=" * 60)

    # Unified format proposal
    print("""
    💡 Unified format (prompt, chosen, rejected):

    {
        "dataset_name": "dataset_name",
        "split": "train/test",
        "source": "original_dataset_name",
        "total_items": count,
        "data": [
            {
                "prompt": "prompt_content",
                "chosen": "chosen_answer",
                "rejected": "rejected_answer"
            }
        ]
    }
    """)


if __name__ == "__main__":
    try:
        # Explore datasets
        dataset1 = explore_hh_rlhf_chat_template()
        dataset2 = explore_social_reasoning_rlhf()
        dataset3 = explore_pku_saferlhf()

        # Compare
        compare_formats()

        # Save in unified format
        datasets_info = {
            "social_reasoning_rlhf": {"dataset": dataset2, "source": "social-reasoning-rlhf"},
            "pku_saferlhf": {"dataset": dataset3, "source": "PKU-SafeRLHF"},
            "hh_rlhf": {"dataset": dataset1, "source": "hh-rlhf-chat-template"},
        }

        save_unified_format(datasets_info)

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Please check your internet connection or dataset access permissions.")
