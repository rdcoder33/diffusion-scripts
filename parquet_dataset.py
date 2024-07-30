import dask.dataframe as dd
from PIL import Image
from io import BytesIO
from tqdm.asyncio import tqdm
import os
import torch
from transformers import CLIPProcessor
from aesthetics_predictor import AestheticsPredictorV1
import asyncio
import aiohttp
import aiofiles
import glob

# Load the aesthetics predictor
model_id = "shunk031/aesthetics-predictor-v1-vit-large-patch14"
predictor = AestheticsPredictorV1.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
device = "cuda" if torch.cuda.is_available() else "cpu"
predictor = predictor.to(device)

# Define the dataset paths
base_path = "hf://datasets/UCSC-VLAA/Recap-DataComp-1B/data/train_data/"
file_pattern = "train-{:05d}-of-03550.parquet"
file_paths = [base_path + file_pattern.format(i) for i in range(101)]  # 0 to 100

# Load the dataset using Dask
df = dd.read_parquet(file_paths)

# Define score ranges and corresponding folder names
score_folders = {
    (2,5.5): "scoreLow",
    (5.5, 6): "Score6",
    (6, 7): "Score7",
    (7, 8): "Score8",
    (8, 9): "Score9",
    (9, float('inf')): "ScoreFull"
}

# Keywords to search for in the re_caption field
keywords = ["flat", "vector", 'svg', 'cartoon', 'clipart', '3d', 'isometric']

# Create folders for each score range
for folder in score_folders.values():
    os.makedirs(os.path.join('images', folder), exist_ok=True)
    os.makedirs(os.path.join('captions', folder), exist_ok=True)

def is_image_large_enough(image_bytes):
    with Image.open(BytesIO(image_bytes)) as img:
        width, height = img.size
        return width > 512 and height > 512

def get_aesthetics_score(image_bytes):
    try:
        image = Image.open(BytesIO(image_bytes))
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = predictor(**inputs)
        return outputs.logits.item()
    except Exception as e:
        print(f"Error in aesthetics scoring: {e}")
        return 0

def get_score_folder(score):
    for (min_score, max_score), folder in score_folders.items():
        if min_score <= score < max_score:
            return folder
    return None

def caption_contains_keyword(caption):
    return any(keyword.lower() in caption.lower() for keyword in keywords)

async def download_and_process_image(session, index, row):
    image_url = row['url']
    caption = row['re_caption']
    
    if not caption_contains_keyword(caption):
        return False, None
    
    try:
        async with session.get(image_url) as response:
            if response.status == 200:
                image_content = await response.read()
                
                if is_image_large_enough(image_content):
                    score = await asyncio.get_event_loop().run_in_executor(None, get_aesthetics_score, image_content)
                    
                    score_folder = get_score_folder(score)
                    if score_folder:
                        image_filename = os.path.join('images', score_folder, f"image_{index}.jpg")
                        caption_filename = os.path.join('captions', score_folder, f"image_{index}.txt")
                        
                        async with aiofiles.open(image_filename, 'wb') as image_file:
                            await image_file.write(image_content)
                        
                        async with aiofiles.open(caption_filename, 'w') as caption_file:
                            await caption_file.write(caption)
                        
                        return True, score_folder
    except Exception as e:
        print(f"Error processing row {index}: {e}")
    
    return False, None

async def process_partition(partition):
    async with aiohttp.ClientSession() as session:
        tasks = [download_and_process_image(session, index, row) for index, row in partition.iterrows()]
        results = await tqdm.gather(*tasks)
    return results

async def main():
    # Process the DataFrame in chunks
    chunk_size = 1000  # Adjust this value based on your memory constraints
    results = []
    
    for partition in tqdm(df.partitions, desc="Processing partitions"):
        df_pandas = partition.compute()
        for i in range(0, len(df_pandas), chunk_size):
            chunk = df_pandas.iloc[i:i+chunk_size]
            chunk_results = await process_partition(chunk)
            results.extend(chunk_results)
    
    successful_downloads = sum(result[0] for result in results)
    folder_counts = {folder: sum(1 for result in results if result[1] == folder) for folder in score_folders.values()}
    
    print(f"Download and caption extraction complete. Successfully processed {successful_downloads} images.")
    print("Images and captions sorted into folders:")
    for folder, count in folder_counts.items():
        print(f"{folder}: {count}")

if __name__ == "__main__":
    asyncio.run(main())