# from unsloth import FastVisionModel # FastLanguageModel for LLMs
# import torch

# # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
# fourbit_models = [
#     "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", # Llama 3.2 vision support
#     "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
#     "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit", # Can fit in a 80GB card!
#     "unsloth/Llama-3.2-90B-Vision-bnb-4bit",

#     "unsloth/Pixtral-12B-2409-bnb-4bit",              # Pixtral fits in 16GB!
#     "unsloth/Pixtral-12B-Base-2409-bnb-4bit",         # Pixtral base model

#     "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",          # Qwen2 VL support
#     "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
#     "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",

#     "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",      # Any Llava variant works!
#     "unsloth/llava-1.5-7b-hf-bnb-4bit",
# ] # More models at https://huggingface.co/unsloth

# model, tokenizer = FastVisionModel.from_pretrained(
#     "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",
#     load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
#     use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
# )

# from datasets import load_dataset
# dataset = load_dataset("unsloth/Radiology_mini", split = "train")

# FastVisionModel.for_inference(model) # Enable for inference!

# image = dataset[0]["image"]
# # image = image.resize((512, 512))
# instruction = "You are an expert radiographer. Describe accurately what you see in this image."

# messages = [
#     {"role": "user", "content": [
#         {"type": "image"},
#         {"type": "text", "text": instruction}
#     ]}
# ]
# input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
# inputs = tokenizer(
#     image,
#     input_text,
#     add_special_tokens = False,
#     return_tensors = "pt",
# ).to("cuda")

# from transformers import TextStreamer
# text_streamer = TextStreamer(tokenizer, skip_prompt = True)
# _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
#                    use_cache = True, temperature = 1.5, min_p = 0.1)



from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-2B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
