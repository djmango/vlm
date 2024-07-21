#!/bin/bash

# Initialize and update all submodules
git submodule update --init --recursive

# Read .gitmodules file and process each submodule
while IFS= read -r line
do
    if [[ $line == *"path = "* ]]; then
        # Extract submodule path
        submodule_path=$(echo $line | sed 's/.*path = //')
        
        echo "Processing submodule: $submodule_path"
        
        # Change to submodule directory
        cd "$submodule_path"
        
        # Install the package in editable mode
        pip install -e .
        
        # Return to the root directory
        cd - > /dev/null
    fi
done < .gitmodules

echo "Sobumodule setup complete."

pip install -U Pillow

pip install scipy wandb datasets nvitop deepspeed matplotlib python-dotenv

# Set the environment variable for Hugging Face datasets cache
export HF_DATASETS_CACHE="/workspace/cache"

echo "HF_DATASETS_CACHE set to /workspace/cache"

deepspeed --num_gpus=4 dist.py --deepspeed --deepspeed_config ds_config.json
