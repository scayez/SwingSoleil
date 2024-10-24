import os
import shutil
from PIL import Image
import glob
from moviepy.editor import ImageSequenceClip



def copy_png_files(source_directory):
    # Define the target folder where the PNG files will be copied
    target_directory = os.path.join(source_directory, 'copied_png_files')
    
    # Create the target directory if it doesn't exist
    os.makedirs(target_directory, exist_ok=True)
    
    # Specify the folders to look for
    target_folders = [ 'Eiger_image','basler_image', 'integration_corr', 'positions']
    
    # Loop through each subfolder in the source directory
    for subfolder in os.listdir(source_directory):
        subfolder_path = os.path.join(source_directory, subfolder)
        
        # Check if it's a directory
        if os.path.isdir(subfolder_path):
            # Loop through each specified folder
            for folder in target_folders:
                folder_path = os.path.join(subfolder_path, folder)
                
                # Check if the specified folder exists
                if os.path.isdir(folder_path):
                    # Loop through each file in the folder
                    for file in os.listdir(folder_path):
                        if file.endswith('.png'):
                            # Construct full file path
                            file_path = os.path.join(folder_path, file)
                            # Copy the file to the target directory
                            shutil.copy(file_path, target_directory)


def group_filenames(folder):
    folder_path = os.path.join(folder, 'copied_png_files')  # Corrected path joining

    # Initialize a dictionary to hold lists of file paths by reference
    file_paths_dict = {}

    # Loop through all .png files in the specified folder
    for file_path in glob.glob(os.path.join(folder_path, '*.png')):
        # Extract the reference (S_C_X_P_X) part and category
        reference = file_path.split('@')[-2].split('.png')[0]
        category = file_path.split('@')[-3]  # Extract the category

        # Initialize the reference in the dictionary if not already present
        if reference not in file_paths_dict:
            file_paths_dict[reference] = {
                'basler': [],
                'eiger': [],
                'corr_integration_': [],
                'pos_': []
            }

        # Append the file path to the appropriate category list
        if category in file_paths_dict[reference]:
            file_paths_dict[reference][category].append(file_path)

    return file_paths_dict

def create_image_grid(images, grid_size=(2, 2)):
    """Create a new image from a list of images arranged in a grid."""
    # Calculate the size of the new image
    img_width, img_height = images[0].size
    new_image = Image.new('RGB', (img_width * grid_size[0], img_height * grid_size[1]))

    # Paste images into the new image
    for index, img in enumerate(images):
        x = (index % grid_size[0]) * img_width
        y = (index // grid_size[0]) * img_height
        new_image.paste(img, (x, y))
    
    return new_image

# def create_grids_from_paths(folder):
#     file_paths_dict = group_filenames(folder)
#     grid_folder = os.path.join(folder, 'grid_images')
#     if not os.path.exists(grid_folder):
#         os.makedirs(grid_folder)
#     # Iterate over each reference and create grids
#     print(file_paths_dict)
#     for reference, categories in file_paths_dict.items():
#         # Collect images for the grid
#         images = []
#         for category in ['eiger', 'basler', 'corr_integration_', 'pos_']:
#             if category in categories and categories[category]:
#                 # Load the first image from each category
#                 img_path = categories[category][0]
#                 img = Image.open(img_path)
#                 images.append(img)

#         # Create a grid if we have enough images
#         if len(images) == 4:  # We need exactly 4 images for a 2x2 grid
#             grid_image = create_image_grid(images, grid_size=(2, 2))
#             #grid_image.show()  # Display the grid image
#             # Optionally save the grid image
#             grid_image.save(os.path.join(grid_folder, f"{reference}_grid.png"))


def create_grids_from_paths(folder):
    file_paths_dict = group_filenames(folder)
    grid_folder = os.path.join(folder, 'grid_images')
    if not os.path.exists(grid_folder):
        os.makedirs(grid_folder)
    # Iterate over each reference and create grids
    print(file_paths_dict)
    for reference, categories in file_paths_dict.items():
        # Collect images for the grid
        images = []
        file_nums = [] 
        for category in ['eiger', 'basler', 'corr_integration_', 'pos_']:
            if category in categories and categories[category]:
                # Load the first image from each category
                img_path = categories[category][0]
                img = Image.open(img_path)
                images.append(img)

        # # Create a grid if we have enough images
        # if len(images) == 4:  # We need exactly 4 images for a 2x2 grid
        #     grid_image = create_image_grid(images, grid_size=(2, 2))
        #     # Extract the last 10 characters of the reference filename
        #     reference_filename = os.path.basename(categories['eiger'][0])  # Assuming 'eiger' category has the reference filename
        #     file_num = reference_filename[-9:-5]
        #     # Optionally save the grid image with the last 10 characters of the filename
        #     #grid_image.save(os.path.join(grid_folder, f"{file_num}.png"))
        #     grid_image.save(os.path.join(grid_folder, f"{reference}_{file_num}.png"))


        # Extract the file number from the filename
                reference_filename = os.path.basename(img_path)
                file_num = reference_filename[-9:-5]
                file_nums.append(file_num)  # Store the extracted file number
        # Check if we have enough images to create a grid
        if len(images) >= 4:  # Adjust this condition based on your requirements
            # Create grids for the first four images (or more if desired)
            for i in range(0, len(images), 4):  # Create grids for every set of 4 images
                grid_images = images[i:i+4]  # Get the next set of 4 images
                if len(grid_images) == 4:  # Ensure we have exactly 4 images
                    grid_image = create_image_grid(grid_images, grid_size=(2, 2))
                    # Use the corresponding file numbers for naming
                    grid_image.save(os.path.join(grid_folder, f"{reference}_{file_nums[i]}_{file_nums[i+1]}_{file_nums[i+2]}_{file_nums[i+3]}.png"))
        else:
            print(f"Skipping {reference}: Not enough images for a grid.")  # Debug: Notify if skipping





# def convert_video(image_folder):
#      #Specify the path to your images

#     for file in glob.glob(folder+'/video/'):
#         # Create a video clip from the images
#         clip = ImageSequenceClip(file, fps=2)  # Set the desired frames per second (fps)

#         # Write the video file
#         clip.write_videofile(folder+'/video.mp4', codec='libx264')  # Output vi

def convert_video(image_folder):
    # Specify the path to your images
    video_folder = os.path.join(image_folder, 'grid_images')
    print(video_folder)
    # Check if the video folder exists
    if not os.path.exists(video_folder):
        print("The video folder does not exist.")
        return
    # Create a list of image files in the video folder
    image_files = sorted(glob.glob(video_folder + '/*.png'))  # Change the pattern as needed
    
    if not image_files:
        print("No images found in the specified folder.")
        return
    # Create a video clip from the images
    clip = ImageSequenceClip(image_files, fps=2)  # Set the desired frames per second (fps)
    # Write the video file
    clip.write_videofile(os.path.join(image_folder, 'video.mp4'), codec='libx264')  # Output video
    # Delete the video folder
    #shutil.rmtree(video_folder)
    print("The video folder has been deleted.")