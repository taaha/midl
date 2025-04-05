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

from datasets import load_dataset

from unsloth import FastVisionModel # FastLanguageModel for LLMs

ds = load_dataset('unsloth/Radiology_mini', split="train")

def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


def plot_conf_scores(conf_matrix, class_="cat", is_trained=False):
    """
    Plot confidence scores as a heatmap where rows are image tokens (1024) and columns are layers (27).
    """
    number_of_layers = len(conf_matrix)
    number_of_image_tokens = len(conf_matrix.T)

    # Transpose to match the desired visualization (1024 rows x 27 columns)
    conf_matrix = conf_matrix.T
    
    # Create the figure and axis
    plt.figure(figsize=(12, 8))
   
    # Create heatmap using seaborn
    sns.heatmap(conf_matrix, cmap='Blues', 
                xticklabels=list(range(number_of_layers)),  # Layer indices
                yticklabels=list(range(0, number_of_image_tokens, 100)),  # Show every 100th token index
                cbar_kws={'label': 'Confidence Score'})
    
    plt.xlabel('LM Layer')
    plt.ylabel('Image Embedding Index')
    plt.title(f'Confidence Scores Across Layers and Image Tokens for {class_}')
    
    if not os.path.exists(f'data/pixtral/heatmaps'):
        os.makedirs(f'data/pixtral/heatmaps')

    # Save the plot
    plt.savefig(f'data/pixtral/heatmaps/confidence_heatmap_{class_}_{is_trained}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_conf_segmentation(conf_scores, image, class_="cat", is_trained=False):
    ## max value logits
    # Get max values across layers (axis 0) for each token position
    max_values = np.max(conf_scores, axis=0)
    max_values_reshaped = max_values.reshape(32, 32)
    segmentation_resized = (np.array(Image.fromarray(max_values_reshaped).resize((image.width, image.height), Image.BILINEAR)))

    ## last layer logits
    # last_layer_logits = [float(tensor.cpu()) for tensor in conf_scores[-1]]
    # last_layer_logits_reshaped = np.array(last_layer_logits).reshape(32, 32)
    # image_width, image_height = image.size
    # segmentation_resized = (np.array(Image.fromarray(last_layer_logits_reshaped).resize((image_width, image_height), Image.BILINEAR)))

    if not os.path.exists(f'data/pixtral/segmented_images'):
        os.makedirs(f'data/pixtral/segmented_images')

    plt.imshow(image)
    plt.imshow(segmentation_resized, cmap='jet', interpolation='bilinear', alpha=0.5)
    plt.axis('off')
    plt.title(f"'{class_}' localization")
    plt.tight_layout()
    plt.savefig(f'data/pixtral/segmented_images/segmentation_resized_{class_}_{is_trained}.png')

def analyze_text(text, model_id, image=None, image_index=0, target_token="cat", is_trained=False):
    """
    Perform logit lens analysis on given text using Gemma model.
    """
    conf_scores = np.load(f'temp/confidence_scores_{target_token}_{is_trained}.npy')
    
    plot_conf_scores(conf_scores, class_=target_token, is_trained=is_trained)
    plot_conf_segmentation(conf_scores, image, class_=target_token, is_trained=is_trained)


if __name__ == "__main__":
    # Add argument parsing
    parser = argparse.ArgumentParser(description='Analyze text with Gemma model.')
    parser.add_argument('--text', type=str, 
                        default="You are an expert radiographer. Describe accurately what you see in this image.", 
                        help='Text to analyze')
    parser.add_argument('--image_index', type=int, default=0, help='dataset index of image')
    parser.add_argument('--target_token', type=str, default="lesion", help='Target token for analysis')
    parser.add_argument('--model_id', type=str, default="unsloth/Pixtral-12B-2409-bnb-4bit", help='Model id')
    parser.add_argument("--is_trained", type=bool, default=False, help="Whether the model is trained")
    args = parser.parse_args()  # Parse the arguments

    # Use the parsed arguments
    text = args.text
    img_index = args.image_index
    image = ds[img_index]["image"].convert("RGB")
    image = image.resize((512, 512))
    
    final_generation = analyze_text(text, image=image, model_id=args.model_id, image_index=img_index, target_token=args.target_token, is_trained=args.is_trained)
    print(final_generation)

