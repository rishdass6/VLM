# prepare_data.py
from datasets import load_dataset, load_from_disk
from transformers import LlavaProcessor

processor = LlavaProcessor.from_pretrained("llava-hf/llava-interleave-qwen-0.5b-hf")

def preprocess_example(example):
    image = example["image"].convert("RGB")
    prompt = f"<image>\nQuestion: {example['question']} Answer:"
    full_text = prompt + " " + example["answer"]

    encoding = processor(
        text=full_text,
        images=image,
        return_tensors="pt",
        padding="max_length",
        max_length=1024,
        truncation=True,
    )

    input_ids = encoding["input_ids"].squeeze()
    labels = input_ids.clone()

    prompt_encoding = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    )

    prompt_len = prompt_encoding["input_ids"].shape[1]
    labels[:prompt_len] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": encoding["attention_mask"].squeeze(),
        "pixel_values": encoding["pixel_values"].squeeze(),
        "labels": labels,
    }

if __name__ == "__main__":
    dataset = load_dataset("flaviagiammarino/vqa-rad", split="train[:100]")
    print(dataset.column_names)
    print(dataset[0])

    dataset = dataset.map(
        preprocess_example,
        batched=False,
        remove_columns=dataset.column_names,
        num_proc=1,          # keep at 1 on Windows to avoid multiprocess issues
    )

    split = dataset.train_test_split(test_size=0.1, seed=42)
    split["train"].save_to_disk("data/train")
    split["test"].save_to_disk("data/val")
    print("Done!")