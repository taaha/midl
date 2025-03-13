#!/bin/bash

echo "Cloning the private GitHub repository..."


git clone https://taaha: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@github.com/taaha/midl.git

# Navigate inside the cloned repository
cd midl || { echo "Failed to enter directory"; exit 1; }

# Install the dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found!"
fi

huggingface-cli login --token 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

git config --global user.email "taaha.s.bajwa@gmail.com"
git config --global user.name "taaha"

echo "Setup complete!"