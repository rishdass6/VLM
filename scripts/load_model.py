import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_name = "llava-hf/llava-interleave-qwen-0.5b-hf"

processor = AutoProcessor.from_pretrained(model_name)
processor.tokenizer.padding_side = "right"

model = LlavaForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype = torch.float32,
    device_map = "auto"
)

model.enable_input_require_grads()

print(model)
print("Done")