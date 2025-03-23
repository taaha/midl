import requests
from PIL import Image
from io import BytesIO
import torch
import numpy as np
import argparse
import os
import gc

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
import seaborn as sns
import pandas as pd

import plotly.graph_objects as go

def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


def get_confidence_for_class(model, processor, text, image, class_="cat"):
    """
    Analyze logit lens for Gemma model across different layers.
    """
    # Tokenize input
    inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    
    # Get all hidden states
    with torch.no_grad():
        torch.compiler.cudagraph_mark_step_begin()
        outputs = model.generate(**inputs, output_hidden_states=True, output_logits=True, return_dict_in_generate=True)

    generation = outputs.sequences[0]
    final_generation = processor.decode(generation, skip_special_tokens=True)

    # counting number of image token (257152)
    number_of_image_tokens = torch.sum(torch.eq(outputs.sequences[0],257152)).item()
    assert number_of_image_tokens == 1024

    image_tokens_generation = outputs.sequences[0][0:number_of_image_tokens]

    class_index = processor.tokenizer.encode(class_)[0]

    number_of_image_tokens = len(outputs.sequences[0][0:number_of_image_tokens])
    number_of_layers = len(outputs.hidden_states[0])
    print(f"Number of image tokens: {number_of_image_tokens}")
    print(f"Number of layers: {number_of_layers}")
    
    # Create a numpy array to store probabilities
    temp_dir = 'temp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    probs_file = f'temp//confidence_scores_{class_}.npy'
    if os.path.exists(probs_file):
        os.remove(probs_file)
    
    layer_probs = np.zeros((number_of_layers, number_of_image_tokens))  # (num_layers, num_image_tokens)

    # going thru 27 layers
    for layer_idx, hidden_state in tqdm(enumerate(outputs.hidden_states[0])):
        # going through 1024 image tokens
        for token_idx, token_logits in tqdm(enumerate(hidden_state[0][0:number_of_image_tokens])):
            token_logits = model.language_model.lm_head(token_logits)
            probs = torch.nn.functional.softmax(token_logits, dim=-1)
            class_prob = probs[class_index]
            layer_probs[layer_idx, token_idx] = class_prob.float().detach().cpu().numpy()
            # for debugging heatmap bug
            # if layer_idx == 0 and token_idx == 0:
            #     print(f"Layer {layer_idx}, Token {token_idx}, Class Prob: {class_prob}")
            # if layer_idx == 20 and token_idx == 1023:
            #     print(f"Layer {layer_idx}, Token {token_idx}, Class Prob: {class_prob}")
            # if layer_idx == 42 and token_idx == 1023:
            #     print(f"Layer {layer_idx}, Token {token_idx}, Class Prob: {class_prob}")
        print_gpu_memory()
        
        # Save the current layer's probabilities to file
        np.save(probs_file, layer_probs)

    del layer_probs, model
    gc.collect(); torch.cuda.empty_cache()

    print_gpu_memory()

    return final_generation

def analyze_text(text, model_id="google/paligemma2-10b-mix-448", image=None, target_token="cat"):
    """
    Perform logit lens analysis on given text using Gemma model.
    """
    # Load model and tokenizer
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto").eval()
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    tokenizer = processor.tokenizer
    
    final_generation = get_confidence_for_class(model, processor, text, image, class_=target_token)


if __name__ == "__main__":
    # Add argument parsing
    parser = argparse.ArgumentParser(description='Analyze text with Gemma model.')
    parser.add_argument('--text', type=str, default="caption the picture",
                      help='Text to analyze')
    parser.add_argument('--image', type=str, default="images/COCO_val2014_000000562150.jpg",
                      help='Path to the image file')
    parser.add_argument('--target_token', type=str, default="cat",
                      help='Target token for analysis')
    args = parser.parse_args()  # Parse the arguments

    # Use the parsed arguments
    text = args.text
    img_path = args.image
    image = Image.open(img_path)
    
    final_generation = analyze_text(text, image=image, target_token=args.target_token)
    print(final_generation)

