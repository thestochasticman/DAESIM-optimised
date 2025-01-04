import gdown
import os
from os import makedirs

def download_google_drive_folder(folder_url, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Use gdown to download the folder
    gdown.download_folder(folder_url, quiet=False, use_cookies=False, output=output_path)
    print(f"Folder downloaded successfully to {output_path}")

if __name__ == "__main__":
    from os import rename
    # Replace with your folder's shareable link
    folder_url = 'https://drive.google.com/drive/u/0/folders/1NnQ1_8bRx3BOgycdsPKavbrmcWK3zpdb'
    output_path = '/home/y/DAESIM2-ANALYSIS-DATA'
    makedirs(output_path, exist_ok=True)
    download_google_drive_folder(folder_url, output_path)
