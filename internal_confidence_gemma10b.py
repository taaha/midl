import requests
from PIL import Image
from io import BytesIO
import torch
import numpy as np
import argparse
import os

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

        print_gpu_memory()
        
        # Save the current layer's probabilities to file
        np.save(probs_file, layer_probs)

    del layer_probs, model

    # Clear CUDA cache
    if torch.cuda.is_available():
        print("Clearing CUDA cache")
        torch.cuda.empty_cache()

    # Force garbage collection
    import gc
    gc.collect()

    print_gpu_memory()

    raise Exception("Stop here")

    return final_generation


def plot_conf_scores(conf_scores, class_="cat"):
    """
    Plot confidence scores as a heatmap where rows are image tokens (1024) and columns are layers (27).
    """
    # Convert list of lists of tensors to numpy array
    conf_matrix = np.array([[float(tensor.cpu()) for tensor in layer] for layer in conf_scores])
    
    # Transpose to match the desired visualization (1024 rows x 27 columns)
    conf_matrix = conf_matrix.T
    
    # Create the figure and axis
    plt.figure(figsize=(12, 8))
    
    # Create heatmap using seaborn
    sns.heatmap(conf_matrix, cmap='Blues', 
                xticklabels=list(range(27)),  # Layer indices
                yticklabels=list(range(0, 1024, 100)),  # Show every 100th token index
                cbar_kws={'label': 'Confidence Score'})
    
    plt.xlabel('LM Layer')
    plt.ylabel('Image Embedding Index')
    plt.title(f'Confidence Scores Across Layers and Image Tokens for {class_}')
    
    # Save the plot
    plt.savefig('confidence_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_conf_scores_log(conf_scores):
    """
    Plot confidence scores as a heatmap where rows are image tokens (1024) and columns are layers (27).
    Using log scale to better visualize the range of values.
    """
    # Convert list of lists of tensors to numpy array
    conf_matrix = np.array([[float(tensor.cpu()) for tensor in layer] for layer in conf_scores])
    
    # Transpose to match the desired visualization (1024 rows x 27 columns)
    conf_matrix = conf_matrix.T
    
    # Apply log transformation (adding small epsilon to avoid log(0))
    epsilon = 1e-10
    conf_matrix_log = np.log10(conf_matrix + epsilon)
    
    # Create the figure and axis
    plt.figure(figsize=(12, 8))
    
    # Create heatmap using seaborn with a logarithmic colormap
    sns.heatmap(conf_matrix_log, 
                cmap='Blues',
                xticklabels=list(range(27)),  # Layer indices
                yticklabels=list(range(0, 1024, 100)),  # Show every 100th token index
                cbar_kws={'label': 'Log10(Confidence Score)'})
    
    plt.xlabel('LM Layer')
    plt.ylabel('Image Embedding Index')
    plt.title('Log-scale Confidence Scores Across Layers and Image Tokens')
    
    # Save the plot
    plt.savefig('confidence_heatmap_log.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_conf_segmentation(conf_scores, image, class_="cat"):
    ## max value logits
    stacked_tensors = torch.stack([torch.stack(tensor_list) for tensor_list in conf_scores])
    max_values = torch.max(stacked_tensors, dim=0).values
    max_values_list = max_values.tolist()
    max_values_list_reshaped = np.array(max_values_list).reshape(32, 32)
    segmentation_resized = (np.array(Image.fromarray(max_values_list_reshaped).resize((image.width, image.height), Image.BILINEAR)))

    ## last layer logits
    # last_layer_logits = [float(tensor.cpu()) for tensor in conf_scores[-1]]
    # last_layer_logits_reshaped = np.array(last_layer_logits).reshape(32, 32)
    # image_width, image_height = image.size
    # segmentation_resized = (np.array(Image.fromarray(last_layer_logits_reshaped).resize((image_width, image_height), Image.BILINEAR)))


    plt.imshow(image)
    plt.imshow(segmentation_resized, cmap='jet', interpolation='bilinear', alpha=0.5)
    plt.axis('off')
    plt.title(f"'{class_}' localization")
    plt.tight_layout()
    plt.savefig(f'data/segmented_images/segmentation_resized_{class_}.png')


def analyze_text(text, model_id="google/paligemma2-10b-mix-448", image=None, target_token="cat"):
    """
    Perform logit lens analysis on given text using Gemma model.
    """
    # Load model and tokenizer
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto").eval()
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    tokenizer = processor.tokenizer
    
    conf_scores, final_generation = get_confidence_for_class(model, processor, text, image, class_=target_token)

    # TODO: take max vlaue insteadof final layer
    plot_conf_segmentation(conf_scores, image, class_=target_token)
    return final_generation

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

