import os
import requests
import zipfile
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            size = file.write(data)
            progress_bar.update(size)
    progress_bar.close()

def setup_coco_dataset(base_path):
    # Create the base directory
    os.makedirs(base_path, exist_ok=True)
    
    # URLs for the COCO dataset files
    urls = {
        'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
        'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    }
    
    # Download and extract each file
    for name, url in urls.items():
        zip_path = os.path.join(base_path, f'{name}.zip')
        print(f"Downloading {name}...")
        download_file(url, zip_path)
        
        print(f"Extracting {name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(base_path)
        
        # Remove the zip file after extraction
        os.remove(zip_path)
    
    print("COCO dataset setup complete!")

# Usage
coco_path = os.getenv('COCO_PATH')  # Get COCO directory from environment variable
if coco_path is None:
    raise ValueError("COCO_DIR environment variable is not set")
setup_coco_dataset(coco_path)