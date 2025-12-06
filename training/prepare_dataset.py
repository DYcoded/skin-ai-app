import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

SOURCE_IMAGES_DIR = "HAM10000_all_images" 
CSV_PATH = "HAM10000_metadata.csv"
BASE_OUTPUT_DIR = "processed_dataset"


def create_dataset_structure():
    df = pd.read_csv(CSV_PATH)
    
    classes = df['dx'].unique()
    
    for split in ['train', 'test']:
        for label in classes:
            path = os.path.join(BASE_OUTPUT_DIR, split, label)
            os.makedirs(path, exist_ok=True)
            
    print(f"Folder structure created inside '{BASE_OUTPUT_DIR}'.")
    
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['dx'], random_state=42)
    
    print(f"Total Images: {len(df)}")
    print(f"Training (Train): {len(train_df)}")
    print(f"Testing (Test): {len(test_df)}")
    
    
    def copy_images(dataframe, split_name):
        print(f"Copying {split_name} images...")
        not_found_count = 0
        
        for index, row in dataframe.iterrows():
            image_name = row['image_id'] + ".jpg" 
            label = row['dx'] 
            
            src_path = os.path.join(SOURCE_IMAGES_DIR, image_name)
            dst_path = os.path.join(BASE_OUTPUT_DIR, split_name, label, image_name)
            
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
            else:
                not_found_count += 1
                
        print(f"{split_name} completed. Images not found: {not_found_count}")

    
    copy_images(train_df, 'train')
    copy_images(test_df, 'test')
    
    print("\nCompleted! Dataset is ready.")

if __name__ == "__main__":
    create_dataset_structure()