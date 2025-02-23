import requests
from PIL import Image
from io import BytesIO
import torch

from transformers.generation.logits_process import TopKLogitsWarper
from transformers.generation.logits_process import LogitsProcessorList

from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)
from transformers.image_utils import load_image
import torch

from tqdm import tqdm

import matplotlib.pyplot as plt

def get_logit_lens(model, processor, text, image, layer_ids=None):
    """
    Analyze logit lens for Gemma model across different layers.
    """
    # Tokenize input
    inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)
    
    # Get all hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict_in_generate=True)
    
    hidden_states = outputs.hidden_states
    if layer_ids is None:
        layer_ids = range(len(hidden_states))
    
    # Get logits for each layer
    logits_per_layer = []
    for layer_id in tqdm(layer_ids):
        layer_hidden = hidden_states[layer_id]
        layer_logits = model.language_model.lm_head(layer_hidden)
        logits_per_layer.append(layer_logits)
    
    return logits_per_layer

def plot_logit_lens(logits_per_layer, tokenizer, target_token="cat", top_k=5):
    """
    Plot top-k token probabilities across layers for a specific token.
    """
    plt.figure(figsize=(12, 6))
    
    # Get probabilities for last position
    probs_per_layer = [torch.softmax(logits[0, -1], dim=-1) for logits in logits_per_layer]
    
    # Get the index of the target token
    target_token_id = tokenizer.encode(target_token)[0]
    
    # Get probabilities for the target token across layers
    target_probs = [probs[target_token_id].item() for probs in probs_per_layer]
    
    # Plot probabilities for the target token across layers
    plt.plot(target_probs, label=f"'{target_token}'")
    
    plt.xlabel("Layer")
    plt.ylabel("Probability")
    plt.title(f"Logit Lens Analysis for '{target_token}'")
    plt.legend()
    plt.grid(True)
    return plt

def analyze_text(text, model_id="google/paligemma2-3b-mix-448", image=None):
    """
    Perform logit lens analysis on given text using Gemma model.
    """
    # Load model and tokenizer
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto").eval()
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    tokenizer = processor.tokenizer
    
    # Get logits across layers
    logits = get_logit_lens(model, processor, text, image)
    
    # Plot results
    plt = plot_logit_lens(logits, tokenizer)
    plt.savefig("logit_lens.png")

# Example usage
if __name__ == "__main__":
    text = "caption the picture"
    img_path = "images/COCO_val2014_000000562150.jpg"
    image = Image.open(img_path)
    analyze_text(text, image=image)
