# Medical Visual Question Answering with LLaVA + LoRA

Fine-tuned a LLaVA multimodal model on the VQA-RAD medical imaging dataset using LoRA for parameter-efficient training.

## What it does
Given a medical image (X-ray, MRI, CT scan) and a question, the model generates a text answer.

## Model & Architecture
- **Base model**: `llava-hf/llava-interleave-qwen-0.5b-hf`
- **Vision encoder**: SigLIP ViT — encodes the image into patch embeddings
- **Language model**: Qwen2 — generates the answer tokens
- **Connector**: Linear projector that maps vision embeddings into the language model's token space
- **Fine-tuning**: LoRA (r=16, alpha=32) applied to all attention and MLP layers in the language model

## Dataset
- **Source**: [VQA-RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad)
- **Size**: ~1800 train / 451 test examples
- **Format**: Medical image + question → short answer
- **Preprocessing**: Images resized via processor, text formatted as `<image>\nQuestion: {q} Answer: {a}`, prompt tokens masked with -100 in labels

## Training
| Setting | Value |
|---|---|
| Model | LLaVA Interleave Qwen 0.5B |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Trainable params | 1.19% (10.4M / 874M) |
| Quantization | None (float16) |
| Batch size | 2 |
| Epochs | 5 |
| Learning rate | 2e-4 |
| Hardware | NVIDIA RTX A4500 (20GB) |

## Results
| Model | Exact Match Accuracy |
|---|---|
| Base (no fine-tuning) | ~0% |
| Fine-tuned (epoch 5) | TBD |

## Reproduce
```bash
git clone https://github.com/your-username/your-repo
pip install -r requirements.txt

python scripts/prepare_data.py
python scripts/train.py
python scripts/evaluate.py
```

## Project Structure
```
scripts/
  prepare_data.py   # download and preprocess dataset
  load_model.py     # load base model
  apply_lora.py     # apply LoRA config
  train.py          # training loop
  evaluate.py       # evaluation + metrics
checkpoints/        # saved LoRA adapter weights
data/               # preprocessed train/val splits
```