from peft import LoraConfig, get_peft_model
from load_model import model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type = "CAUSAL_LM",
    bias = "none"
)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()