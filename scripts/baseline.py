import requests
import json
import wandb
import sys
import os

from detr.datasets import build_dataset

# log baseline

def read_log_file(url):
    response = requests.get(url)
    return response.text.split('\n')

def parse_log_row(row):
    try:
        return json.loads(row)
    except json.JSONDecodeError:
        return None

def main():
    # Initialize wandb
    wandb.init(project="NaViT_COCO", name="DETR101 150 epoch")

    # URL of the log file
    log_url = "https://gist.githubusercontent.com/szagoruyko/b4c3b2c3627294fc369b899987385a3f/raw/f019ce4568becddc0e0594e87903eddbefc60c7b/log_25935349.txt"

    # Read the log file
    log_rows = read_log_file(log_url)
    # Create a fake args object with the required argument

    class Args:
        def __init__(self):
            self.dataset_file = 'coco'
            self.masks = False
            self.coco_path = '/home/minjune/coco'

    args = Args()

    BS = 2
    world_size = 3

    dataset_train = build_dataset(image_set='train', args=args)
    # Process each row
    for i, row in enumerate(log_rows):
        data = parse_log_row(row)
        print(f"logging epoch {i}...")
        #for s in range(len(dataset_train)//BS*world_size):
        if data:
            wandb.log(data)

    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    main()