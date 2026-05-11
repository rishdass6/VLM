import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import get_linear_schedule_with_warmup
from apply_lora import peft_model
from load_model import processor

EPOCHS = 3
LR = 2e-4
BATCH_SIZE = 2
OUTPUT_DIR = "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = load_from_disk("data/train")
val_dataset = load_from_disk("data/val")

def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(b['input_ids']) for b in batch])
    attention_mask = torch.stack([torch.tensor(b['attention_mask']) for b in batch])
    pixel_values = torch.stack([torch.tensor(b['pixel_values']) for b in batch])
    labels = torch.stack([torch.tensor(b['labels']) for b in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values,
        'labels': labels    
    }

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

optimizer = torch.optim.AdamW(
    [p for p in peft_model.parameters() if p.requires_grad],
    lr = LR,
    weight_decay = 0.01
)

total_steps = len(train_loader) * EPOCHS
warmup_steps = int(0.03 * total_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

peft_model.to(DEVICE)
peft_model.train()

for epoch in range(EPOCHS):
    total_loss = 0

    for step, batch in enumerate(train_loader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        outputs = peft_model(**batch)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_norm = 1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if step % 10 == 0:
            print(f"Epoch {epoch + 1} | Step {step} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    avg_train_loss = total_loss / len(train_loader)

    peft_model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            output = peft_model(**batch)
            val_loss += output.loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    peft_model.save_pretrained(f"{OUTPUT_DIR}/epoch_{epoch + 1}")
    processor.save_pretrained(f"{OUTPUT_DIR}/epoch_{epoch + 1}")
    print(f"Saved checkpoint to {OUTPUT_DIR}/epoch_{epoch + 1}")

    peft_model.train()

print("Training DONE")

