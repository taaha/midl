# %% [markdown] {"id":"IqM-T1RTzY6C"}
# To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
# <div class="align-center">
#   <a href="https://github.com/unslothai/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
#   <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
#   <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
# </div>
# 
# To install Unsloth on your own computer, follow the installation instructions on our Github page [here](https://github.com/unslothai/unsloth?tab=readme-ov-file#-installation-instructions).
# 
# **[NEW] As of Novemeber 2024, Unsloth now supports vision finetuning!**
# 
# You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & [how to save it](#Save)
# 
# **This notebook finetunes Llama 3.2 11B to assist medical professionals in analyzing xrays, CT scans & ultrasounds.**

# %% [code] {"id":"2eSvM9zX_2d3","execution":{"iopub.status.busy":"2025-04-01T07:18:49.145936Z","iopub.execute_input":"2025-04-01T07:18:49.146260Z","iopub.status.idle":"2025-04-01T07:22:50.830678Z","shell.execute_reply.started":"2025-04-01T07:18:49.146224Z","shell.execute_reply":"2025-04-01T07:22:50.829434Z"}}
# pip install pip3-autoremove
# pip-autoremove torch torchvision torchaudio -y
# pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
# pip install unsloth

# %% [markdown] {"id":"r2v_X2fA0Df5"}
# * We support Llama 3.2 Vision 11B, 90B; Pixtral; Qwen2VL 2B, 7B, 72B; and any Llava variant like Llava NeXT!
# * We support 16bit LoRA via `load_in_4bit=False` or 4bit QLoRA. Both are accelerated and use much less memory!

# %% [code] {"id":"QmUBVEnvCDJv","outputId":"f63afa9e-e320-4da2-bb91-b7f4f9163379","execution":{"iopub.status.busy":"2025-04-01T07:22:50.832050Z","iopub.execute_input":"2025-04-01T07:22:50.832373Z","iopub.status.idle":"2025-04-01T07:24:06.490904Z","shell.execute_reply.started":"2025-04-01T07:22:50.832336Z","shell.execute_reply":"2025-04-01T07:24:06.490188Z"}}
from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", # Llama 3.2 vision support
    "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
    "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit", # Can fit in a 80GB card!
    "unsloth/Llama-3.2-90B-Vision-bnb-4bit",

    "unsloth/Pixtral-12B-2409-bnb-4bit",              # Pixtral fits in 16GB!
    "unsloth/Pixtral-12B-Base-2409-bnb-4bit",         # Pixtral base model

    "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",          # Qwen2 VL support
    "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",

    "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",      # Any Llava variant works!
    "unsloth/llava-1.5-7b-hf-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

# %% [markdown] {"id":"SXd9bTZd1aaL"}
# We now add LoRA adapters for parameter efficient finetuning - this allows us to only efficiently train 1% of all parameters.
# 
# **[NEW]** We also support finetuning ONLY the vision part of the model, or ONLY the language part. Or you can select both! You can also select to finetune the attention or the MLP layers!

# %% [code] {"id":"6bZsfBuZDeCL","execution":{"iopub.status.busy":"2025-04-01T07:24:06.492503Z","iopub.execute_input":"2025-04-01T07:24:06.492754Z","iopub.status.idle":"2025-04-01T07:24:12.388514Z","shell.execute_reply.started":"2025-04-01T07:24:06.492732Z","shell.execute_reply":"2025-04-01T07:24:12.387809Z"}}
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)

# %% [markdown] {"id":"vITh0KVJ10qX"}
# <a name="Data"></a>
# ### Data Prep
# We'll be using a sampled version of the ROCO radiography dataset. You can access the dataset [here](https://huggingface.co/datasets/unsloth/Radiology_mini). The full dataset is [here](https://huggingface.co/datasets/eltorio/ROCOv2-radiology).
# 
# The dataset includes X-rays, CT scans and ultrasounds showcasing medical conditions and diseases. Each image has a caption written by experts describing it. The goal is to finetune a VLM to make it a useful analysis tool for medical professionals.

# %% [code] {"id":"LjY75GoYUCB8","outputId":"ac18735b-4865-4634-cdac-0672be9dc0e6","execution":{"iopub.status.busy":"2025-04-01T07:24:12.389675Z","iopub.execute_input":"2025-04-01T07:24:12.389923Z","iopub.status.idle":"2025-04-01T07:24:17.667734Z","shell.execute_reply.started":"2025-04-01T07:24:12.389901Z","shell.execute_reply":"2025-04-01T07:24:17.666898Z"}}
from datasets import load_dataset
dataset = load_dataset("unsloth/Radiology_mini", split = "train")

# %% [markdown] {"id":"W1W2Qhsz6rUT"}
# Let's take a look at the dataset, and check what the 1st example shows:

# %% [code] {"id":"bfcSGwIb6p_R","outputId":"111b6e00-b335-431c-c3a6-7e0d6702c056","execution":{"iopub.status.busy":"2025-04-01T07:24:17.668743Z","iopub.execute_input":"2025-04-01T07:24:17.669079Z","iopub.status.idle":"2025-04-01T07:24:17.673977Z","shell.execute_reply.started":"2025-04-01T07:24:17.669055Z","shell.execute_reply":"2025-04-01T07:24:17.673127Z"}}
dataset

# %% [code] {"id":"uOLWY2936t1n","outputId":"f0a03924-d626-441d-f8d4-746427e23b0c","execution":{"iopub.status.busy":"2025-04-01T07:24:17.674809Z","iopub.execute_input":"2025-04-01T07:24:17.675102Z","iopub.status.idle":"2025-04-01T07:24:17.764957Z","shell.execute_reply.started":"2025-04-01T07:24:17.675079Z","shell.execute_reply":"2025-04-01T07:24:17.764064Z"}}
dataset[0]["image"]

# %% [code] {"id":"lXjfJr4W6z8P","outputId":"05f346ff-3e4e-455b-f273-71429a91de26","execution":{"iopub.status.busy":"2025-04-01T07:24:17.765772Z","iopub.execute_input":"2025-04-01T07:24:17.766130Z","iopub.status.idle":"2025-04-01T07:24:17.774673Z","shell.execute_reply.started":"2025-04-01T07:24:17.766100Z","shell.execute_reply":"2025-04-01T07:24:17.774058Z"}}
dataset[0]["caption"]

# %% [markdown] {"id":"K9CBpiISFa6C"}
# To format the dataset, all vision finetuning tasks should be formatted as follows:
# 
# ```python
# [
# { "role": "user",
#   "content": [{"type": "text",  "text": instruction}, {"type": "image", "image": image} ]
# },
# { "role": "assistant",
#   "content": [{"type": "text",  "text": answer} ]
# },
# ]
# ```
# 
# We will craft an custom instruction asking the VLM to be an expert radiographer. Notice also instead of just 1 instruction, you can add multiple turns to make it a dynamic conversation.

# %% [code] {"id":"oPXzJZzHEgXe","execution":{"iopub.status.busy":"2025-04-01T07:24:17.777199Z","iopub.execute_input":"2025-04-01T07:24:17.777393Z","iopub.status.idle":"2025-04-01T07:24:17.789465Z","shell.execute_reply.started":"2025-04-01T07:24:17.777376Z","shell.execute_reply":"2025-04-01T07:24:17.788892Z"}}
instruction = "You are an expert radiographer. Describe accurately what you see in this image."

def convert_to_conversation(sample):
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : instruction},
            {"type" : "image", "image" : sample["image"]} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["caption"]} ]
        },
    ]
    return { "messages" : conversation }
pass

# %% [markdown] {"id":"FY-9u-OD6_gE"}
# Let's convert the dataset into the "correct" format for finetuning:

# %% [code] {"id":"gFW2qXIr7Ezy","execution":{"iopub.status.busy":"2025-04-01T07:24:17.791066Z","iopub.execute_input":"2025-04-01T07:24:17.791254Z","iopub.status.idle":"2025-04-01T07:24:37.552700Z","shell.execute_reply.started":"2025-04-01T07:24:17.791236Z","shell.execute_reply":"2025-04-01T07:24:37.551670Z"}}
converted_dataset = [convert_to_conversation(sample) for sample in dataset]

# %% [markdown] {"id":"ndDUB23CGAC5"}
# The first example is now structured like below:

# %% [code] {"id":"gGFzmplrEy9I","outputId":"a56fd110-056d-45d8-fbed-3bb694fb0d44","execution":{"iopub.status.busy":"2025-04-01T07:24:37.553824Z","iopub.execute_input":"2025-04-01T07:24:37.554194Z","iopub.status.idle":"2025-04-01T07:24:37.559459Z","shell.execute_reply.started":"2025-04-01T07:24:37.554167Z","shell.execute_reply":"2025-04-01T07:24:37.558779Z"}}
converted_dataset[0]

# %% [markdown] {"id":"FecKS-dA82f5"}
# Before we do any finetuning, maybe the vision model already knows how to analyse the images? Let's check if this is the case!

# %% [code] {"id":"vcat4UxA81vr","outputId":"4d32760d-9094-45d1-d014-012525958314","execution":{"iopub.status.busy":"2025-04-01T07:24:37.560373Z","iopub.execute_input":"2025-04-01T07:24:37.560648Z","iopub.status.idle":"2025-04-01T07:25:34.441227Z","shell.execute_reply.started":"2025-04-01T07:24:37.560612Z","shell.execute_reply":"2025-04-01T07:25:34.440593Z"}}
FastVisionModel.for_inference(model) # Enable for inference!

image = dataset[0]["image"]
instruction = "You are an expert radiographer. Describe accurately what you see in this image."

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction}
    ]}
]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                   use_cache = True, temperature = 1.5, min_p = 0.1)

# %% [markdown] {"id":"idAEIeSQ3xdS"}
# <a name="Train"></a>
# ### Train the model
# Now let's use Huggingface TRL's `SFTTrainer`! More docs here: [TRL SFT docs](https://huggingface.co/docs/trl/sft_trainer). We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`. We also support TRL's `DPOTrainer`!
# 
# We use our new `UnslothVisionDataCollator` which will help in our vision finetuning setup.

# %% [code] {"id":"95_Nn-89DhsL","outputId":"27b03b47-2fab-4905-9408-e8fa30e8c7fd","execution":{"iopub.status.busy":"2025-04-01T07:25:34.442180Z","iopub.execute_input":"2025-04-01T07:25:34.442503Z","iopub.status.idle":"2025-04-01T07:25:34.581516Z","shell.execute_reply.started":"2025-04-01T07:25:34.442471Z","shell.execute_reply":"2025-04-01T07:25:34.580896Z"}}
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model) # Enable for training!

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = converted_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # max_steps = 30,
        num_train_epochs = 1, # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",     # For Weights and Biases

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 4,
        max_seq_length = 2048,
    ),
)

# %% [code] {"cellView":"form","id":"2ejIt2xSNKKp","outputId":"e5294c5d-7420-4c1e-b4d8-c93caa5e384e","execution":{"iopub.status.busy":"2025-04-01T07:25:34.582301Z","iopub.execute_input":"2025-04-01T07:25:34.582576Z","iopub.status.idle":"2025-04-01T07:25:34.588239Z","shell.execute_reply.started":"2025-04-01T07:25:34.582546Z","shell.execute_reply":"2025-04-01T07:25:34.587560Z"}}
#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# %% [code] {"id":"yqxqAZ7KJ4oL","outputId":"108465c4-f25c-4c6a-e308-2937f9eb7fcd","execution":{"iopub.status.busy":"2025-04-01T07:25:34.589060Z","iopub.execute_input":"2025-04-01T07:25:34.589247Z","execution_failed":"2025-04-01T08:35:03.091Z"}}
trainer_stats = trainer.train()

# %% [code] {"cellView":"form","id":"pCqnaKmlO1U9","outputId":"888d5a4b-7c0d-485a-d347-efaaed70abe5","execution":{"execution_failed":"2025-04-01T08:35:03.091Z"}}
#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# %% [markdown] {"id":"ekOmTR1hSNcr"}
# <a name="Inference"></a>
# ### Inference
# Let's run the model! You can change the instruction and input - leave the output blank!
# 
# We use `min_p = 0.1` and `temperature = 1.5`. Read this [Tweet](https://x.com/menhguin/status/1826132708508213629) for more information on why.

# %% [code] {"id":"kR3gIAX-SM2q","outputId":"ac901efd-de05-40ab-ac45-45c0f35ad9ff","execution":{"execution_failed":"2025-04-01T08:35:03.092Z"}}
FastVisionModel.for_inference(model) # Enable for inference!

image = dataset[0]["image"]
instruction = "You are an expert radiographer. Describe accurately what you see in this image."

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction}
    ]}
]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                   use_cache = True, temperature = 1.5, min_p = 0.1)

# %% [markdown] {"id":"uMuVrWbjAzhc"}
# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Huggingface's `push_to_hub` for an online save or `save_pretrained` for a local save.
# 
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!

# %% [code] {"id":"upcOlWe7A1vc","outputId":"f70e56fe-3cf1-4980-b259-ae2de09e824a","execution":{"execution_failed":"2025-04-01T08:35:03.092Z"}}
model.save_pretrained("lora_model") # Local saving
tokenizer.save_pretrained("lora_model")
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving

# %% [markdown] {"id":"AEEcJ4qfC7Lp"}
# Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:

# %% [code] {"id":"MKX_XKs_BNZR","outputId":"d9c5f139-6084-431a-acc1-62f4d5d248ad","execution":{"execution_failed":"2025-04-01T08:35:03.092Z"}}
if False:
    from unsloth import FastVisionModel
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        load_in_4bit = True, # Set to False for 16bit LoRA
    )
    FastVisionModel.for_inference(model) # Enable for inference!

image = dataset[0]["image"]
instruction = "You are an expert radiographer. Describe accurately what you see in this image."

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction}
    ]}
]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                   use_cache = True, temperature = 1.5, min_p = 0.1)

# %% [markdown] {"id":"f422JgM9sdVT"}
# ### Saving to float16 for VLLM
# 
# We also support saving to `float16` directly. Select `merged_16bit` for float16. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens.

# %% [code] {"execution":{"execution_failed":"2025-04-01T08:35:03.092Z"}}
model.push_to_hub("darthPanda/llama3.2_11b_vl_radiology", token = "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") # Online saving
# tokenizer.push_to_hub("darthPanda/qwen_2b_vl_radiology", token = "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") # Online saving



# %% [markdown] {"id":"Zt9CHJqO6p30"}
# And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!
# 
# Some other links:
# 1. Llama 3.2 Conversational notebook. [Free Colab](https://colab.research.google.com/drive/1T5-zKWM_5OD21QHwXHiV9ixTRR7k3iB9?usp=sharing)
# 2. Saving finetunes to Ollama. [Free notebook](https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing)
# 3. Llama 3.2 Vision finetuning - Radiography use case. [Free Colab](https://colab.research.google.com/drive/1j0N4XTY1zXXy7mPAhOC1_gMYZ2F2EBlk?usp=sharing)
# 4. Qwen 2 VL Vision finetuning - Maths OCR to LaTeX. [Free Colab](https://colab.research.google.com/drive/1whHb54GNZMrNxIsi2wm2EY_-Pvo2QyKh?usp=sharing)
# 5. Pixtral 12B Vision finetuning - General QA datasets. [Free Colab](https://colab.research.google.com/drive/1K9ZrdwvZRE96qGkCq_e88FgV3MLnymQq?usp=sharing)
# 6. More notebooks for DPO, ORPO, Continued pretraining, conversational finetuning and more on our [Github](https://github.com/unslothai/unsloth)!
# 
# <div class="align-center">
#   <a href="https://github.com/unslothai/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
#   <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
#   <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
# </div>