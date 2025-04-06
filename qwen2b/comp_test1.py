import requests
from PIL import Image
from io import BytesIO
import torch
import numpy as np
import argparse
import os  # Add this import at the top of your file

from transformers.generation.logits_process import TopKLogitsWarper
from transformers.generation.logits_process import LogitsProcessorList

from unsloth import FastVisionModel # FastLanguageModel for LLMs
from transformers.image_utils import load_image
import torch

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import plotly.graph_objects as go

from datasets import load_dataset

ds = load_dataset('unsloth/Radiology_mini', split="train")

def get_confidence_for_class(model, tokenizer, text, image, class_="lesion"):
    """
    Analyze logit lens for Gemma model across different layers.
    """
    # Tokenize input
    FastVisionModel.for_inference(model) # Enable for inference!

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

    # text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    # r = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
    #                 use_cache = True, temperature = 1.5, min_p = 0.1)
    outputs = model.generate(**inputs, max_new_tokens = 128,
                    use_cache = True, temperature = 1.5, min_p = 0.1,
                    output_hidden_states = True, output_logits = True, return_dict_in_generate = True)

    generation = outputs.sequences[0]
    final_generation = tokenizer.decode(generation, skip_special_tokens=True)
    # Extract only the text after '/nassistant'
    final_generation = final_generation.split('\nassistant\n')[-1]

    # counting number of image token (151644)
    number_of_image_tokens = torch.sum(torch.eq(outputs.sequences[0],151655)).item()
    assert number_of_image_tokens == 324

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

    class_index = tokenizer.tokenizer.encode(class_)[0]

    class_probs_across_layers_for_each_image_token = []
    # going thru 27 layers
    for hidden_state in tqdm(outputs.hidden_states[0]):
        # going through 1024 image tokens
        class_probs_across_each_image_token = []
        for token_logits in tqdm(hidden_state[0][0:number_of_image_tokens]):
            token_logits = model.lm_head(token_logits)
            probs = torch.nn.functional.softmax(token_logits, dim=-1)
            class_prob = probs[class_index]
            class_probs_across_each_image_token.append(class_prob)
        class_probs_across_layers_for_each_image_token.append(class_probs_across_each_image_token)

    # len(class_probs_across_each_image_token) -> 1024 (number of image tokens)
    # len(class_probs_across_layers_for_each_image_token) -> 27 (number of layers)
    
    return class_probs_across_layers_for_each_image_token, final_generation


def plot_conf_scores(conf_scores, class_="lesion", is_trained=False, image_index=0, final_generation=""):
    """
    Plot confidence scores as a heatmap where rows are image tokens (1024) and columns are layers (27).
    """
    # Convert list of lists of tensors to numpy array
    conf_matrix = np.array([[float(tensor.cpu()) for tensor in layer] for layer in conf_scores])
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
    
    # Add the first 20 characters of final_generation at the bottom
    plt.text(0.5, -0.1, final_generation, 
             horizontalalignment='center',
             transform=plt.gca().transAxes,
             fontsize=10)
    
    if not os.path.exists(f'data/heatmaps/{image_index}'):
        os.makedirs(f'data/heatmaps/{image_index}')

    # Save the plot with adjusted margins to show text
    plt.savefig(f'data/heatmaps/{image_index}/confidence_heatmap_{class_}_{is_trained}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()



def plot_conf_segmentation(conf_scores, image, class_="lesion", is_trained=False, image_index=0, final_generation=""):
    ## max value logits
    stacked_tensors = torch.stack([torch.stack(tensor_list) for tensor_list in conf_scores])
    max_values = torch.max(stacked_tensors, dim=0).values
    max_values_list = max_values.tolist()
    max_values_list_reshaped = np.array(max_values_list).reshape(18, 18)
    segmentation_resized = (np.array(Image.fromarray(max_values_list_reshaped).resize((image.width, image.height), Image.BILINEAR)))

    # Create directory if it doesn't exist
    os.makedirs(f'data/segmented_images/{image_index}', exist_ok=True)

    plt.figure(figsize=(10, 8))  # Make figure slightly larger to accommodate colorbar
    plt.imshow(image)
    im = plt.imshow(segmentation_resized, cmap='jet', interpolation='bilinear', alpha=0.5)
    plt.colorbar(im, label='Confidence Score')  # Add colorbar with label
    plt.axis('off')
    plt.title(f"'{class_}' localization")
    
    # Add the first 20 characters of final_generation at the bottom
    plt.text(0.5, -0.1, final_generation, 
             horizontalalignment='center',
             transform=plt.gca().transAxes,
             fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'data/segmented_images/{image_index}/segmentation_resized_{class_}_{is_trained}.png',
                bbox_inches='tight')  # Added bbox_inches='tight' to ensure text is not cut off
    plt.close()  # Close the figure to free memory


def analyze_text(text, model_id="unsloth/Qwen2-VL-2B-Instruct-bnb-4bit", image=None, target_token="lesion", is_trained=False, image_index=0):
    """
    Perform logit lens analysis on given text using Gemma model.
    """
    
    # Load model and tokenizer
    model, processor = FastVisionModel.from_pretrained(
        model_id,
        load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
    )
    
    conf_scores, final_generation = get_confidence_for_class(model, processor, text, image, class_=target_token)

    # TODO: take max vlaue insteadof final layer
    plot_conf_segmentation(conf_scores, image, class_=target_token, is_trained=is_trained, image_index=image_index, final_generation=final_generation)
    plot_conf_scores(conf_scores, class_=target_token, is_trained=is_trained, image_index=image_index, final_generation=final_generation)
    return final_generation

if __name__ == "__main__":
    # Add argument parsing
    parser = argparse.ArgumentParser(description='Analyze text with Gemma model.')
    parser.add_argument('--text', type=str, 
                        default="You are an expert radiographer. Describe accurately what you see in this image.", 
                        help='Text to analyze')
    parser.add_argument('--image_index', type=int, default=0, help='dataset index of image')
    parser.add_argument('--target_token', type=str, default="lesion", help='Target token for analysis')
    parser.add_argument('--model_id', type=str, default="unsloth/Qwen2-VL-2B-Instruct-bnb-4bit", help='Model id')
    parser.add_argument("--is_trained", type=bool, default=False, help="Whether the model is trained")
    args = parser.parse_args()  # Parse the arguments

    # Use the parsed arguments
    text = args.text
    img_index = args.image_index
    image = ds[img_index]["image"].convert("RGB")
    image = image.resize((512, 512))
    
    final_generation = analyze_text(text, model_id=args.model_id, image=image, target_token=args.target_token, is_trained=args.is_trained, image_index=img_index)
    print(final_generation)

