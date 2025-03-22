import requests
from PIL import Image
from io import BytesIO
import torch
import numpy as np
import argparse

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
    # len(image_tokens_generation) -> 1024 (list of 1024 elements, every element is 257152)


    # len(outputs.sequences[0][input_len:]) -> 20 (number of generated tokens)
    # output.keys() -> odict_keys(['sequences', 'logits', 'hidden_states', 'past_key_values'])
    # len(outputs.hidden_states) -> 20 (number of generated tokens)
    # len(outputs.hidden_states[0]) -> 27 (number of layers)
    # outputs.hidden_states[0][0].shape -> torch.Size([1, 1029, 2304])    (1029 is number of input tokens)
    # outputs.hidden_states[1][0].shape -> torch.Size([1, 1, 2304])
    # outputs.hidden_states[19][0].shape -> torch.Size([1, 1, 2304])
    # outputs.hidden_states[19][26][0][0].shape -> torch.Size([2304])       (19 is token number in sentence, 26 is final layer)

    # model.language_model.lm_head. -> (lm_head): Linear(in_features=2304, out_features=257216, bias=False)
    # model.language_model.lm_head(outputs.hidden_states[19][26][0][0]).shape -> torch.Size([257216])
    # model.language_model.lm_head(outputs.hidden_states[19][26][0][0]) ->
    #                               tensor([ -1.5703,   6.2188, -11.3750,  ...,  -1.5547,  -1.5703,  -1.5703],
    #                               device='cuda:0', dtype=torch.bfloat16, grad_fn=<SqueezeBackward4>)

    # outputs.hidden_states[0][0][0][0].shape -> torch.Size([2304])
    # outputs.hidden_states[0][0][0][1024].shape -> torch.Size([2304])

    # breakpoint()

    class_index = processor.tokenizer.encode(class_)[0]

    class_probs_across_layers_for_each_image_token = []
    # going thru 27 layers
    for hidden_state in tqdm(outputs.hidden_states[0]):
        # going through 1024 image tokens
        class_probs_across_each_image_token = []
        for token_logits in tqdm(hidden_state[0][0:number_of_image_tokens]):
            token_logits = model.language_model.lm_head(token_logits)
            probs = torch.nn.functional.softmax(token_logits, dim=-1)
            class_prob = probs[class_index]
            class_probs_across_each_image_token.append(class_prob)
        class_probs_across_layers_for_each_image_token.append(class_probs_across_each_image_token)

    # len(class_probs_across_each_image_token) -> 1024 (number of image tokens)
    # len(class_probs_across_layers_for_each_image_token) -> 27 (number of layers)
    
    return class_probs_across_layers_for_each_image_token, final_generation


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

