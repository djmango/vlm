import csv
from PIL import Image, ImageDraw
import io
import base64
import requests
import os
import matplotlib.pyplot as plt

from tqdm import tqdm
from datasets import load_dataset
from matplotlib.patches import Rectangle
from dotenv import load_dotenv
from datasets import load_dataset
from multiprocessing import Pool, freeze_support
from tqdm import tqdm

load_dotenv()


def display_image_with_bboxes(split='train', index=33):
    ds = load_dataset("agentsea/wave-ui-25k")
    # Get the image and its metadata
    image = ds[split][index]['image']
    bbox = ds[split][index]['bbox']

    print(bbox)
    # Apply bounding box to the image
    image_with_bbox = apply_bbox_to_image(image.copy(), bbox)

    # Create a figure and axis
    fig, ax = plt.subplots(1)
    
    # Display the image with bounding box
    ax.imshow(image_with_bbox)

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Show the plot
    plt.show()

def apply_bbox_to_image(image, bbox):
    # The image is already a PIL Image object, no need to convert
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = tuple(bbox)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    return image

def get_description_from_claude(image):
    # Convert image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Claude API endpoint and headers
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": os.environ["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01"
    }

    # Prepare the request payload
    payload = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Write a description for the UI element indicated by the Red Bounding Box. It is ..."
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_str
                        }
                    }
                ]
            }
        ]
    }

    # Send request to Claude API
    response = requests.post(url, json=payload, headers=headers)
    response_json = response.json()

    # Extract and return the description
    print(response_json)
    ret = response_json['content'][0]['text']
    in_toks = response_json['usage']['input_tokens']
    out_toks = response_json['usage']['output_tokens'] 
    print(ret)
    return ret, in_toks * 0.15 / 1e6 + out_toks *  0.6 / 1e6
    
def generate_descriptions_and_save_csv(split='train', num_samples=None):
    csv_filename = f"{split}_descriptions.csv"
    image_dir = "data/wave-ui-25k"
    os.makedirs(image_dir, exist_ok=True)

    # logs
    total_cost = 0 
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['index', 'image_path', 'description']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in tqdm(enumerate(ds[split]), total=num_samples or len(ds[split])):
            if num_samples and idx >= num_samples:
                break

            image = row['image']
            bbox = row['bbox']

            img_with_bbox = apply_bbox_to_image(image, bbox)
            image_filename = f"{split}_{idx}.png"
            image_path = os.path.join(image_dir, image_filename)
            
            # Convert RGBA to RGB before saving as JPEG
            rgb_image = img_with_bbox.convert('RGB')
            rgb_image.save(image_path, "JPEG")

            description, cost = get_description_from_claude(img_with_bbox)
            print(f'cost: ${cost}')
            total_cost += cost

            
            writer.writerow({
                'index': idx,
                'image_path': image_path,
                'description': description
            })

    print(f"Descriptions saved to {csv_filename}")
    print(f"Images saved in {image_dir}")

# Example usage:
# costs on avg 0.00373 per sample
# cost is $373 for generating 1 mil samples (haiku)
# cost is $217 for generating 1 mil (4o mini)
# TODO substitute 25k with 350k dataset and treat
# each UI element as a sample
#generate_descriptions_and_save_csv('train', num_samples=10)

def process_sample(sample):
    labels = set()
    for label_list in sample['labels']:
        if label_list:  # Check if the list is not empty
            labels.add(label_list[0])  # Add the single label text
    return labels

def process_dataset(dataset):
    all_labels = set()
    for sample in tqdm(dataset['train'], desc="Processing samples", unit="sample"):
        all_labels.update(process_sample(sample))
    return all_labels

def get_cls_labels():
    dt_350k = load_dataset('biglab/webui-350k-elements')
    dt_70k = load_dataset('biglab/webui-70k-elements')
    datasets = [dt_350k, dt_70k]
    
    with Pool() as pool:
        results = pool.map(process_dataset, datasets)
    
    all_labels = set().union(*results)
    return all_labels

if __name__ == '__main__':
    freeze_support()
    labels = get_cls_labels()
    print(len(labels))
    print(labels)

