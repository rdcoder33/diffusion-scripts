import os
import base64
import time
from openai import AzureOpenAI

# Configuration
GPT4V_KEY = "<API_KEY>"
FOLDER_PATH = "/home/azureuser/cogvlm-image-caption/concepts/subfolder_2"  # Folder containing the images

# Configure the Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint="https://30-may-vision-8.openai.azure.com/",
    api_key=GPT4V_KEY,
    api_version="2024-02-01"
)

# Function to save caption to a file
def save_caption_to_file(image_name, caption):
    base_name = os.path.splitext(image_name)[0]  # Remove the file extension
    file_path = os.path.join(FOLDER_PATH, f"{base_name}.txt")
    with open(file_path, 'w') as file:
        file.write(caption)


i = 0
# Iterate over each image in the folder
for image_name in os.listdir(FOLDER_PATH):
    image_path = os.path.join(FOLDER_PATH, image_name)
    base_name = os.path.splitext(image_name)[0]
    caption_file_path = os.path.join(FOLDER_PATH, f"{base_name}.txt")
    print(i)
    i=i+1
    # # Skip processing if the caption file already exists
    # if os.path.exists(caption_file_path):
    #     print(f"Skipping {image_name} as caption file already exists.")
    #     continue
    
    if os.path.isfile(image_path):
        # Encode image in base64
        encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode('ascii')
        
        # Payload for the request
        user_textprompt = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}",
                
            }
        }
        # system_prompt = {
        #     "type": "text",
        #     "text": "You are master image caption creator which creates a caption describing what is happening in the image. Your foucs is on the activity in the image, people in the image, their skin color, race, You describe their gender with words like [man, woman, men, women,boy, girl, male kid, female kid]. You will use following words to describe position: [left,right,top,bottom,top left, top right, top center, bottom left, bottom right, bottom center, center left, center right, center]. You will also make sure to describe the color of each item. Describe the style of the image in detail like flat, strokes, gradients, shadows, vibe only in the end of the caption"
        # }
        system_prompt = {
            "type": "text",
            "text": "You are a expert image caption assistant, which describe each subject in the image with thier position and role in the image. describe the style of the image and the end of the caption"
        }

        # Create the message for the API
        messages = [
            {"role": "system", "content": [system_prompt]},
            {"role": "user", "content": [user_textprompt, {"type": "text", "text": "Image Caption Agency"}]},
        ]

        # Request a completion from the API
        try:
            # time.sleep(2)  # Corrected from time.wait(5)
            completion = client.chat.completions.create(
                model="gpt-4o",  # Use a suitable model
                messages=messages,
                temperature=0.7,
                top_p=0.95,
                max_tokens=32,
            )

            # Extract the caption from the response
            caption = completion.choices[0].message.content
            print(f"Caption for {image_name.replace('.jpg', '')}: {caption}")
            # Save the caption to a file
            save_caption_to_file(image_name, caption)
        except Exception as e:
    
            print(f"Failed to generate caption for {image_name}. Error: {e}")
