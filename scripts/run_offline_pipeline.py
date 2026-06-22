import os
import torch
from datasets import load_dataset
from tqdm import tqdm
# import the implemented github preprocessing tool cleanly
from preprocessing.offline_preprocess import preprocess_image

def process_data_split(split_name, num_samples=1000):
    print(f"Streaming {split_name} split from Hugging Face...")
    dataset = load_dataset("to-be/OpenHand-Synth", split=split_name, streaming=True)
    
    output_dir = f"data/processed/offline/{split_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, item in enumerate(tqdm(dataset, total=num_samples)):
        if i >= num_samples: 
            break
        
        # Save a quick temporary image file for Kanita's function to read
        temp_path = f"temp_{split_name}_{i}.png"
        item['image'].save(temp_path)
        
        try:
            # RUN KANITA'S CODE HERE
            processed_np = preprocess_image(temp_path, image_size=(128, 512))
            
            # Convert to PyTorch Tensor
            tensor_data = torch.tensor(processed_np, dtype=torch.float32).unsqueeze(0)
            
            # Save final tensor package
            torch.save({
                'image': tensor_data, 
                'text': item['text']
            }, f"{output_dir}/sample_{i}.pt")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    # Let's generate 1000 training images and 200 testing images to start!
    process_data_split("train", num_samples=1000)
    process_data_split("test", num_samples=200)
    print(" Data processing complete for both splits!")