from datasets import load_dataset
from transformers import PaliGemmaProcessor
from transformers import PaliGemmaForConditionalGeneration, Trainer
import torch


ds = load_dataset('unsloth/Radiology_mini', split="train")

model_id ="google/paligemma2-3b-mix-448"

device = "cuda"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

for param in model.vision_tower.parameters():
    param.requires_grad = False

for param in model.multi_modal_projector.parameters():
    param.requires_grad = False# or your favorite PaliGemma


from transformers import BitsAndBytesConfig, PaliGemmaForConditionalGeneration
from peft import get_peft_model, LoraConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, device_map="auto")#, quantization_config=bnb_config)
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, device_map="auto", quantization_config=bnb_config)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

DTYPE = model.dtype

processor = PaliGemmaProcessor.from_pretrained(model_id)

import torch

image_token = processor.tokenizer.convert_tokens_to_ids("<image>")
instruction = "You are an expert radiographer. Describe accurately what you see in this image."

def collate_fn(examples):
  texts = [f"<image>" + instruction for example in examples]
  labels= [example['caption'] for example in examples]
  images = [example["image"].convert("RGB") for example in examples]
  tokens = processor(text=texts, images=images, suffix=labels,
                    return_tensors="pt", padding="longest")

  tokens = tokens.to(DTYPE).to(device)
  return tokens


from transformers import TrainingArguments
args=TrainingArguments(
            num_train_epochs=1,
            remove_unused_columns=False,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            learning_rate=2e-5,
            weight_decay=1e-6,
            adam_beta2=0.999,
            logging_steps=100,
            # optim="adamw_hf", # you can use paged optimizers like paged_adamw_8bit for QLoRA
            optim="paged_adamw_8bit",
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=1,
            output_dir="paligemma_vqav2",
            bf16=True,
            report_to=["tensorboard"],
            dataloader_pin_memory=False
        )

trainer = Trainer(
        model=model,
        train_dataset=ds ,
        data_collator=collate_fn,
        args=args
        )

trainer.train()
trainer.push_to_hub()