import requests
from PIL import Image
from io import BytesIO
import torch
import numpy as np

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
    input_len = inputs["input_ids"].shape[-1]
    
    # Get all hidden states
    with torch.no_grad():
        outputs = model.generate(**inputs, output_hidden_states=True, output_logits=True, return_dict_in_generate=True)

    generation = outputs.sequences[0][input_len:]
    final_generation = processor.decode(generation, skip_special_tokens=True)

    breakpoint()
    # len(outputs.sequences[0][input_len:]) -> 20 (number of generated tokens)
    # output.keys() -> odict_keys(['sequences', 'logits', 'hidden_states', 'past_key_values'])
    # len(outputs.hidden_states) -> 20 (number of generated tokens)
    # len(outputs.hidden_states[0]) -> 27 (number of layers)
    # outputs.hidden_states[0][0].shape -> torch.Size([1, 1029, 2304])    (1029 is number of input tokens)
    # outputs.hidden_states[1][0].shape -> torch.Size([1, 1, 2304])
    # outputs.hidden_states[19][0].shape -> torch.Size([1, 1, 2304])   
    
    hidden_states = outputs.hidden_states
    if layer_ids is None:
        layer_ids = range(len(hidden_states))
    
    # Get logits for each layer
    logits_per_layer = []
    # for layer_id in tqdm(layer_ids):
    for layer_id in tqdm(layer_ids[10:]):     # 10:layer_ids
        layer_hidden = hidden_states[layer_id]
        layer_logits = model.language_model.lm_head(layer_hidden)
        logits_per_layer.append(layer_logits)
    
    return logits_per_layer, final_generation

def plot_logit_lens(logits_per_layer, tokenizer, target_token="cat", top_k=5, num_patches=24):
    """
    Plot top-k token probabilities across layers for a specific token.
    Returns both the line plot and a list of segmentation maps.
    """
    line_plot = plt.figure(figsize=(12, 6))
    
    # Get probabilities for all positions
    probs_per_layer = [torch.softmax(logits[0], dim=-1) for logits in tqdm(logits_per_layer[10:])]
    
    # Get the index of the target token
    target_token_ids = tokenizer.encode(target_token)  # Don't skip any tokens
    print(f"Target token ids: {target_token_ids}")
    
    # Get probabilities for the target token across layers
    target_probs = []
    segmentation_maps = []
    
    for probs in probs_per_layer:
        # Get probability for target token at the last position
        target_prob = probs[-1][target_token_ids].max().item()
        target_probs.append(target_prob)
        
        # Generate segmentation map using patch positions
        if probs.shape[0] >= num_patches * num_patches:
            patch_probs = probs[:num_patches * num_patches]  # Take only patch positions
            patch_probs = patch_probs[:, target_token_ids].max(dim=1)[0]  # Max over target tokens
            # breakpoint()
            segmap = patch_probs.view(num_patches, num_patches).detach().cpu().float().numpy()
        else:
            segmap = np.zeros((num_patches, num_patches))
        segmentation_maps.append(segmap)
    
    # Plot probabilities for the target token across layers
    plt.plot(target_probs, label=f"'{target_token}'")
    
    plt.xlabel("Layer")
    plt.ylabel("Probability")
    plt.title(f"Logit Lens Analysis for '{target_token}'")
    plt.legend()
    plt.grid(True)
    return line_plot, segmentation_maps

def analyze_text(text, model_id="google/paligemma2-3b-mix-448", image=None, target_token="cat"):
    """
    Perform logit lens analysis on given text using Gemma model.
    """
    # Load model and tokenizer
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto").eval()
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    tokenizer = processor.tokenizer
    
    # Get logits across layers
    logits, final_generation = get_logit_lens(model, processor, text, image)

    breakpoint()
    
    # Plot results
    line_plot, segmentation_maps = plot_logit_lens(logits, tokenizer, target_token=target_token)
    line_plot.savefig("logit_lens_line.png")
    
    # Plot segmentation maps for selected layers
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    # layers_to_plot = [0, 7, 15, 23, 31, 39, 47, -1]  # Adjust based on your model's layer count
    layers_to_plot = [-1]

    for idx, layer_idx in enumerate(layers_to_plot):
        ax = axes[idx // 4, idx % 4]
        im = ax.imshow(segmentation_maps[layer_idx], cmap='viridis')
        ax.set_title(f'Layer {layer_idx}')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig("logit_lens_segmentation.png")

# Example usage
if __name__ == "__main__":
    text = "caption the picture"
    img_path = "images/COCO_val2014_000000562150.jpg"
    image = Image.open(img_path)
    analyze_text(text, image=image)
