import os
import shutil

def move_images_to_main_directory(main_directory):
    # Supported image file extensions
    image_extensions = (".jpg", ".jpeg", ".png", ".gif")
    
    # Get the list of image files already in the main directory
    existing_files = [file for file in os.listdir(main_directory) if file.lower().endswith(image_extensions)]
    
    # Initialize a list to store the image files found in subdirectories
    image_files = []
    
    # Iterate over all subdirectories and files in the main directory
    for root, dirs, files in os.walk(main_directory):
        for file in files:
            # Check if the file has a supported image extension
            if file.lower().endswith(image_extensions):
                image_files.append(file)
    
    # Check if there are any image files in subdirectories that are not in the main directory
    new_files = set(image_files) - set(existing_files)
    
    if new_files:
        print(f"Found {len(new_files)} new image files to move.")
        
        # Move the new image files to the main directory
        for file in new_files:
            file_path = None
            for root, dirs, files in os.walk(main_directory):
                if file in files:
                    file_path = os.path.join(root, file)
                    break
            
            if file_path:
                shutil.move(file_path, main_directory)
        
        print("Files restructured - image files have been moved to the main directory.")
    else:
        pass
