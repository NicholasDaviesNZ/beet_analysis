# a first look using K-means with three classes to see if we can split out soil, beet plants and other - seams unlikely but worth a go as it will save having to manually segment some of the images for fine tunning of something else. 

# first we will take the aproach of falttening each image out to a l*w x 3 vector, join them togeather into a nxmx4 array, and run a KNN over them, and reconsturct the origonal output form the flattened results

import numpy as np
from PIL import Image
import os
from sklearn.cluster import KMeans
raw_imgs_directory = '/workspaces/beet_analysis/plot_images/'
imgs_directory = '/workspaces/beet_analysis/plot_images_resized/'
save_dir = "/workspaces/beet_analysis/plot_masks/"

# the image size to resize the images to
img_size = (168,123, 3)
# the number of images, we could generate this by counign the number of images in the raw_imgs dir. 
img_num = 512

# the drone was flown at different heights on different occasions resulting in differing array sizes, for now, lets just resize them to a standard of 112x82 to keep the res down
# the first thing we will do is resize the images so they are all the same, and convert them to RGB (as they have an alpha channel) 


def resize_andr_remove_alpha(imgs_directory, save_dir):
    """
    function to take in the raw plot images - created by the ortho to plot images script using the grid with plots json and makes them all a standard size - img_size 
    strips the alpha channel and saves the standardized images to save_dir
    """
    for filename in os.listdir(imgs_directory):
            file_path = os.path.join(imgs_directory, filename)

            with Image.open(file_path) as img:
                img_resized = img.resize((img_size[1], img_size[0])).convert("RGB") 
                img_resized.save(os.path.join(save_dir, f'{filename.split(".")[0]}.png'))
                
def load_images_from_directory(directory):
    """
    Function to load images from the provided directory, flatten them and join them togeather into a numpy array
    """
    reshaped_arrays = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        with Image.open(file_path) as img:
            img_array = np.array(img)
            img_reshape = img_array.reshape(img_array.shape[0] * img_array.shape[1] * img_array.shape[2])
            reshaped_arrays.append(img_reshape)

    combined_array = np.stack(reshaped_arrays, axis=0)
    return(combined_array)


def create_seg(in_dir, save_dir, combined_array):
    """
    runs the kmeans clasifier on 3 cats (giveing it 4 seems to do a bit better job). it takes in the array built by the load function, along with the direcorty the
    images came from - this is just for naming, and the dir the masks will be saved to, and it saves the k-means groups as masks to that dir
    """
    kmeans = KMeans(n_clusters=4, random_state=0) 
    kmeans.fit(combined_array)
    cluster_labels = kmeans.labels_
    clustered_images =  np.zeros((img_num, *img_size))

    for ii in range(img_num):
        clustered_images[ii] = kmeans.cluster_centers_[cluster_labels[ii]].reshape(img_size)

    os.makedirs(save_dir, exist_ok=True)

    for index, filename in  enumerate(os.listdir(in_dir)):
        image = Image.fromarray(clustered_images[index].astype(np.uint8))
        image.save(os.path.join(save_dir, filename))

# take the raw plot images and resize them so they are all the same and strip the alpha channel out
resize_andr_remove_alpha(raw_imgs_directory, imgs_directory)
# get a numpy array of num_samplesximage_len*image_widthx3
combined_array = load_images_from_directory(imgs_directory)
# run the k-means clasifier
create_seg(imgs_directory, save_dir, combined_array)
