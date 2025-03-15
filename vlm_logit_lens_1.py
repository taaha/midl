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
import seaborn as sns
import pandas as pd

import plotly.graph_objects as go

def get_logit_lens(model, processor, text, image, layer_ids=None):
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

    generation = outputs.sequences[0][input_len:]
    final_generation = processor.decode(generation, skip_special_tokens=True)

    # tokens_across_layers[26] matching final_generation for first token
    tokens_across_layers_for_each_generated_token = []
    for hidden_state in tqdm(outputs.hidden_states): # going through 20 generated tokens
        tokens_across_layers = []
        for layer in tqdm(hidden_state): # going through 27 layers
            last_token_logits = model.language_model.lm_head(layer[-1][-1])
            probs = torch.nn.functional.softmax(last_token_logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=1)
            tokens_across_layers.append(processor.tokenizer.decode(top_indices))
        tokens_across_layers_for_each_generated_token.append(tokens_across_layers)
    
    return tokens_across_layers_for_each_generated_token, final_generation


def plot_logit_lens(logits):
    """
    Plot logit lens as a heatmap where each row is a generated token and columns are layers.
    """
    logits = [[x[i] for x in logits] for i in range(len(logits[0]))]

    ones_matrix = [[1 for _ in row] for row in logits]

    fig = go.Figure(data=go.Heatmap(
                        z=ones_matrix,
                        text=logits,
                        texttemplate="%{text}",
                        textfont={"size":10}))

    # Set figure size (width and height in pixels)
    fig.update_layout(width=1200, height=800)
    fig.write_image("logit_lens.png")


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
    fig = plot_logit_lens(logits)


if __name__ == "__main__":
    text = "caption the picture"
    img_path = "images/COCO_val2014_000000562150.jpg"
    image = Image.open(img_path)
    analyze_text(text, image=image)
