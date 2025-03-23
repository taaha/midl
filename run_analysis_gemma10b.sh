#!/bin/bash

# Define the image path and text prompt
IMAGE_PATH="images/COCO_val2014_000000562150.jpg"
TEXT_PROMPT="caption the picture"

# Array of target tokens to analyze
TARGET_TOKENS=(
    "cat",
    "bicycle",
    "grass",
    "yellow",
    "girl",
    "hand",
)

# Loop through each target token and run the analysis
for token in "${TARGET_TOKENS[@]}"; do
    echo "Analyzing for target token: $token"
    python internal_confidence_gemma10b_a.py \
        --text "$TEXT_PROMPT" \
        --image "$IMAGE_PATH" \
        --target_token "$token"
    echo "----------------------------------------"
done 


# Loop through each target token and run the analysis
for token in "${TARGET_TOKENS[@]}"; do
    echo "Analyzing for target token: $token"
    python internal_confidence_gemma10b_b.py \
        --text "$TEXT_PROMPT" \
        --image "$IMAGE_PATH" \
        --target_token "$token"
    echo "----------------------------------------"
done 