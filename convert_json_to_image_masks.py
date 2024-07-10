import json
import os
import shutil
from PIL import Image, ImageDraw

def create_masks(image_dir, mask_json, mask_output_dir, image_output_dir):
    """
    Generate and save binary mask images from the json mask geometries file.

    Args:
        image_dir (str): Directory containing the original images.
        mask_json (str): Path to the JSON file with masking information.
        output_dir (str): Directory to save the generated mask images.
    """
    with open(mask_json) as f:
        mask_data = json.load(f)
    
    if not os.path.exists(mask_output_dir):
        os.makedirs(mask_output_dir)
        
    if not os.path.exists(image_output_dir):
        os.makedirs(image_output_dir)

    for img_name, mask_info in mask_data.items():
        # something a bit weired has happened in the json export from VAI - https://www.robots.ox.ac.uk/~vgg/software/via/
        # splitting on the g in image_file_name.png and taking the first piece to remove the number on the end of the file, 
        # should be fine as a hack for now as file names are date_plot.png
        img_name = img_name.split('g')[0]+'g'
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            continue
        
        image = Image.open(img_path)
        mask = Image.new('L', image.size, 0)
        
        regions = mask_info.get('regions', {})
        for region in regions.values():
            shape_attr = region['shape_attributes']
            if shape_attr['name'] == 'polygon':
                all_points_x = shape_attr['all_points_x']
                all_points_y = shape_attr['all_points_y']
                polygon = list(zip(all_points_x, all_points_y))
                mask_value = region['region_attributes'].get('beet')
                if mask_value is None:
                    continue
                draw = ImageDraw.Draw(mask)
                draw.polygon(polygon, outline=int(mask_value), fill=int(mask_value)) # note you wont see this in a normal image as the values are so close to black, *100 to get viewable ones

        mask_output_path = os.path.join(mask_output_dir, img_name)
        mask.save(mask_output_path)
        shutil.copy(f"{image_dir}{img_name}", f"{image_output_dir}{img_name}")



image_dir = '/workspaces/beet_analysis/plot_images_resized/'
mask_json = '/workspaces/beet_analysis/via_region_data.json'
mask_output_dir = '/workspaces/beet_analysis/images_masks/'
image_output_dir = '/workspaces/beet_analysis/images_with_masks/'
create_masks(image_dir, mask_json, mask_output_dir, image_output_dir)