import torch
from PIL import Image
from datasets import load_from_disk
from transformers import LlavaForConditionalGeneration, AutoProcessor
from peft import PeftModel

BASE_MODEL = "llava-hf/llava-interleave-qwen-0.5b-hf"
ADAPTER_PATH = "checkpoints/epoch_5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(BASE_MODEL)
base_model = LlavaForConditionalGeneration.from_pretrained(
    BASE_MODEL, 
    torch_dtype = torch.float32,
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model = model.merge_and_unload()
model.eval()

def generate_answer(image, question):
    prompt = f"<image>Question: {question} Answer:"

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors = "pt",
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens = 32,
            temperature = 0.1,
            do_sample = False
        )
    
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    answer = processor.tokenizer.decode(generated, skip_special_tokens = True)
    return answer.strip()

def exact_match(pred, gt):
    return int(pred.lower().strip() == gt.lower().strip())

val_dataset = load_from_disk("data/val")

correct = 0
total = len(val_dataset)

for i, example in enumerate(val_dataset):
    image = example["image"] if isinstance(example["image"], Image.Image) else Image.open(example["image"]).convert("RGB")
    question = example["question"]
    gt = example["answer"]

    pred = generate_answer(image, question)
    match = exact_match(pred, gt)

    correct += match

    print(f"[{i+1}/{total}] Q: {question}")
    print(f"  GT: {gt} | Pred: {pred} | Match: {match}")

acc = correct / total
print(f"\nExact Match Accuracy: {acc:.4f} ({correct}/{total})")