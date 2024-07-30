import os
import shutil
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# Load the pre-trained Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to find the corresponding image file
def find_image_file(input_folder, base_name):
    for ext in ['.jpg', '.jpeg', '.webp', '.png']:
        image_name = base_name + ext
        image_path = os.path.join(input_folder, image_name)
        if os.path.exists(image_path):
            return image_path
    return None

# Read captions from text files
captions = []
file_names = []
input_folder = '/home/azureuser/dalle-4'  # Replace with the path to your input folder

print(f"Searching for caption files in: {input_folder}")

if not os.path.exists(input_folder):
    print(f"Error: Input folder '{input_folder}' does not exist.")
    exit(1)

for file_name in os.listdir(input_folder):
    if file_name.endswith('.txt'):
        file_path = os.path.join(input_folder, file_name)
        try:
            with open(file_path, 'r') as file:
                caption = file.read().strip()
                if caption:  # Only add non-empty captions
                    captions.append(caption)
                    file_names.append(file_name)
                else:
                    print(f"Warning: Empty caption in file {file_name}")
        except Exception as e:
            print(f"Error reading file {file_name}: {str(e)}")

print(f"Found {len(captions)} caption files.")

if not captions:
    print("Error: No captions found. Please check the input folder and file contents.")
    exit(1)

# Encode captions to get embeddings
print("Encoding captions...")
embeddings = model.encode(captions)

# Cluster the embeddings
print("Clustering embeddings...")
num_clusters = min(100, len(captions))  # Adjust the number of clusters, but ensure it's not larger than the number of captions
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(embeddings)
cluster_assignment = clustering_model.labels_

# Organize captions into clusters
clusters = {}
for caption, file_name, cluster_id in zip(captions, file_names, cluster_assignment):
    if cluster_id not in clusters:
        clusters[cluster_id] = []
    clusters[cluster_id].append((caption, file_name))

# Create subfolders for each category and move files
output_folder = '/home/azureuser/categories'  # Replace with the path to your output folder
print(f"Creating category folders and moving files to: {output_folder}")

os.makedirs(output_folder, exist_ok=True)

for cluster_id, cluster_items in clusters.items():
    category_folder = os.path.join(output_folder, f'CATEGORY_{cluster_id + 1}')
    os.makedirs(category_folder, exist_ok=True)
    
    for caption, file_name in cluster_items:
        # Move caption text file
        src_txt = os.path.join(input_folder, file_name)
        dst_txt = os.path.join(category_folder, file_name)
        shutil.move(src_txt, dst_txt)
        
        # Find and move corresponding image file
        base_name = os.path.splitext(file_name)[0]
        src_img = find_image_file(input_folder, base_name)
        if src_img:
            dst_img = os.path.join(category_folder, os.path.basename(src_img))
            shutil.move(src_img, dst_img)
        else:
            print(f"Warning: No image found for caption file {file_name}")
        
    # Create a summary file for the category
    summary_file = os.path.join(category_folder, f'CATEGORY_{cluster_id + 1}_summary.txt')
    with open(summary_file, 'w') as file:
        for caption, _ in cluster_items:
            file.write(f'{caption}\n')

print("Categories have been created and files have been moved to their respective subfolders.")