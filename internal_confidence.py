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

    """
    working code:

    # Get logits for the last token from the last layer
    last_token_logits = model.language_model.lm_head(outputs.hidden_states[19][26][0][0])
    
    # Convert to probabilities using softmax
    probs = torch.nn.functional.softmax(last_token_logits, dim=-1)
    
    # Get top 5 probabilities and their indices
    top_probs, top_indices = torch.topk(probs, k=5)
    
    # Convert to CPU and regular Python types for easier printing
    top_probs = top_probs.cpu().tolist()
    top_indices = top_indices.cpu().tolist()

    processor.tokenizer.decode(top_indices)
    """

    """
    working code
    tokens_across_layers_for_each_generated_token = []
    # going thru 27 layers
    for hidden_state in tqdm(outputs.hidden_states[0]):
        # going through 1024 image tokens
        tokens_across_each_image_token = []
        for token_logits in tqdm(hidden_state[0][0:number_of_image_tokens]):
            token_logits = model.language_model.lm_head(token_logits)
            probs = torch.nn.functional.softmax(token_logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=1)
            tokens_across_each_image_token.append(processor.tokenizer.decode(top_indices))
        tokens_across_layers_for_each_generated_token.append(tokens_across_each_image_token)

    # len(tokens_across_layers_for_each_generated_token) -> 27 (number of layers)
    # len(tokens_across_layers_for_each_generated_token[0]) -> 1024 (number of image tokens)
    """

    class_index = processor.tokenizer.encode(class_)[0]

    class_probs_across_layers_for_each_generated_token = []
    # going thru 27 layers
    for hidden_state in tqdm(outputs.hidden_states[0]):
        # going through 1024 image tokens
        class_probs_across_each_image_token = []
        for token_logits in tqdm(hidden_state[0][0:number_of_image_tokens]):
            token_logits = model.language_model.lm_head(token_logits)
            probs = torch.nn.functional.softmax(token_logits, dim=-1)
            class_prob = probs[class_index]
            class_probs_across_each_image_token.append(class_prob)
        class_probs_across_layers_for_each_generated_token.append(class_probs_across_each_image_token)

    breakpoint() # TODO: check if this is correct tomorrow



        # tokens_across_layers = []
        # for layer in tqdm(hidden_state): # going through 27 layers
        #     last_token_logits = model.language_model.lm_head(layer[-1][-1])
        #     probs = torch.nn.functional.softmax(last_token_logits, dim=-1)
        #     top_probs, top_indices = torch.topk(probs, k=1)
        #     tokens_across_layers.append(processor.tokenizer.decode(top_indices))
        # tokens_across_layers_for_each_generated_token.append(tokens_across_layers)

    
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
    logits, final_generation = get_confidence_for_class(model, processor, text, image)
    fig = plot_logit_lens(logits)


if __name__ == "__main__":
    text = "caption the picture"
    img_path = "images/COCO_val2014_000000562150.jpg"
    image = Image.open(img_path)
    analyze_text(text, image=image)

