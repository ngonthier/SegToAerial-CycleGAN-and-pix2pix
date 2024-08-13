import glob
import pathlib
import os
from PIL import Image
import numpy as np
from osgeo import gdal
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool
import ntpath

LUT = [
    {"color": "#db0e9a", "class": "building"},
    {"color": "#938e7b", "class": "pervious surface"},
    {"color": "#f80c00", "class": "impervious surface"},
    {"color": "#a97101", "class": "bare soil"},
    {"color": "#1553ae", "class": "water"},
    {"color": "#194a26", "class": "coniferous"},
    {"color": "#46e483", "class": "deciduous"},
    {"color": "#f3a60d", "class": "brushwood"},
    {"color": "#660082", "class": "vineyard"},
    {"color": "#55ff00", "class": "herbaceous vegetation"},
    {"color": "#fff30d", "class": "agricultural land"},
    {"color": "#e4df7c", "class": "plowed land"},
    {"color": "#3de6eb", "class": "swimming pool"},
    {"color": "#ffffff", "class": "snow"},
    {"color": "#8ab3a0", "class": "clear cut"},
    {"color": "#6b714f", "class": "mixed"},
    {"color": "#c5dc42", "class": "ligneous"},
    {"color": "#9999ff", "class": "greenhouse"},
    {"color": "#000000", "class": "other"}
]

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def convert_to_seg(image_array, lut):
    def hex_to_tuple(hex):
        h = hex.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            class_id = image_array[i, j][0]
            
            color = hex_to_tuple(lut[class_id-1]["color"]) if class_id < len(lut) else (0,0,0)

            image_array[i, j] = color

    # Convert the NumPy array back to an image and save it to a file
    return image_array


def convert_seg(input_path, output_path, LUT):
    input_tif = gdal.Open(input_path)
    rgb_data = np.zeros((input_tif.RasterYSize, input_tif.RasterXSize, 3), dtype=np.uint8)

    rgb_data[..., 0] = input_tif.GetRasterBand(1).ReadAsArray()
    rgb_data = convert_to_seg(rgb_data, LUT)

    # Close input TIFF file
    input_tif = None

    img = Image.fromarray(rgb_data)
    img.save(output_path)

def convert_image(input_path, output_path):
    input_tif = gdal.Open(input_path)
    rgb_data = np.zeros((input_tif.RasterYSize, input_tif.RasterXSize, 3), dtype=np.uint8)

    for i in range(3): # Keep only RGB
        rgb_data[..., i] = input_tif.GetRasterBand(i+1).ReadAsArray()

    # Close input TIFF file
    input_tif = None

    img = Image.fromarray(rgb_data)
    img.save(output_path)

def convert_msk(input_path, output_path):
    im = Image.open(msk)
    imarray = np.array(im)
    # Need to remove the other classes and to convert to color 256 RGB 
    print(np.shape(imarray),np.unique(imarray))
    nb_mask += 1

def convert_dataset_pix2pix_format(path_dataset='\\\\store\\store-DAI\\projets\\ocs\\dataset\\dp014_V1-2_FLAIR19_RVBIE',
                                   output_path='D:\data\PixtoPix_FLAIR',max_number_img=20,
                                   path_csv_files='D:\data\FLAIR-INC',no_multiprocessing=False,
                                   not_redo=True):
    """
    Create folder /path/to/data with subfolders A and B. A and B should each have their own subfolders train, val, test, etc. 
    In /path/to/data/A/train, put training images in style A. In /path/to/data/B/train, put the corresponding images in style B.
    Repeat same for other data splits (val, test, etc)."""

    if not no_multiprocessing:
        pool=Pool()

    path_train_A = os.path.join('A','train')
    path_train_B = os.path.join('B','train')
    path_test_A = os.path.join('A','test')
    path_test_B = os.path.join('B','test')

    train_set = os.path.join(path_csv_files,'TRAIN_FLAIR-INC 1.csv')
    test_set_path = os.path.join(path_csv_files,'TEST_FLAIR-INC 1.csv')

    test_set = pd.read_csv(test_set_path,names=['img','msk'])
    test_set['img'] = test_set['img'].apply(lambda x: x.split('/')[-1])
    test_set['img'] = test_set['img'].apply(lambda x: x.split('.')[0])
    list_test_img = test_set['img'].to_list()
    print('list_test_img',list_test_img)

    pathlib.Path(os.path.join(output_path,path_train_A)).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(output_path,path_train_B)).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(output_path,path_test_A)).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(output_path,path_test_B)).mkdir(parents=True, exist_ok=True)
    aerial_tif = glob.glob(os.path.join(path_dataset,"*","*","img","*.tif"))
    mask_tif = glob.glob(os.path.join(path_dataset,"*","*","msk","*.tif"))

    number_image = 0
    number_test_img = 0
    print('Number of images :',len(aerial_tif))

    for img, msk in tqdm(zip(aerial_tif, mask_tif)):
        print(img)
        if "D032_2016" in img: # Zone a ignorer 
            continue

        if number_image > max_number_img:
            break
        filename_img = os.path.basename(img).split('.')[0]
        img_short = path_leaf(img).split('.')[0]

        #number = filename_img.split('_')[1]
        if img_short in list_test_img: 
            print('test image : ',img)
            number_test_img += 1 
            folder_A = path_test_A
            folder_B = path_test_B
        else:
            folder_A = path_train_A
            folder_B = path_train_B


        base_dir_A = os.path.join(output_path,folder_A)
        base_dir_B = os.path.join(output_path,folder_B)

        mask_out = f"{base_dir_A}/{filename_img}.png"
        image_out = f"{base_dir_B}/{filename_img}.png"
        #json_out = f"{base_dir}/{number}.json" # can be used for the prompt of pix2pixTurbo non ? 
        #percentages_out = f"{base_dir}/PCT_{number}.json"

        #if not_redo: 
        #    if os.path.isfile(mask_out) and os.path.isfile(image_out):
        #        number_image += 1
        #        continue

        #meta = metadata[filename_img]
        #if not no_multiprocessing:
        #    pool.apply_async(convert_seg, args=(msk, mask_out, LUT))
        #    pool.apply_async(convert_image, args=(img, image_out))
        #else:
        #    convert_seg(msk, mask_out, LUT)
        #    convert_image(img, image_out)
        number_image += 1

    if not no_multiprocessing:
        pool.close()
        pool.join()

    print('Number of images in test set',number_test_img)
    print('Number of images in total',number_image)


def copy_files_format_for_pix2pix(path_dataset='\\\\store\\store-DAI\\projets\\ocs\\dataset\\dp014_V1-2_FLAIR19_RVBIE',output_path='D:\data\PixtoPix_FLAIR',max_number_img=2):
    """
    Create folder /path/to/data with subfolders A and B. A and B should each have their own subfolders train, val, test, etc. 
    In /path/to/data/A/train, put training images in style A. In /path/to/data/B/train, put the corresponding images in style B.
    Repeat same for other data splits (val, test, etc)."""
    
    
    # Besoin de charger les images dans le test set / celle dans le train set et celle à ignorer 

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    domains = [f for f in os.listdir(path_dataset)]
    
    nb_img = 0
    nb_mask = 0

    for domain in domains:
        print(domain) 
        if domain=='': # Domain à laisser de côté 
            continue
        path_domain= os.path.join(path_dataset, domain)
        zones = [f for f in os.listdir(path_domain)]
        for zone in zones:
            path_zone = os.path.join(path_domain, zone,'msk')
            list_msks = glob.glob(os.path.join(path_zone,'*.tif'))
            print(list_msks)
            for msk in list_msks: 
                if nb_mask < max_number_img:
                    print(msk)
                    im = Image.open(msk)
                    imarray = np.array(im)
                    # Need to remove the other classes and to convert to color 256 RGB 
                    print(np.shape(imarray),np.unique(imarray))
                    nb_mask += 1
            path_aerial = os.path.join(path_domain, zone,'img')
            list_imgs = glob.glob(os.path.join(path_aerial,'*.tif'))
            for img in list_imgs: 
                if nb_img < max_number_img:
                    print(img)
                    im = Image.open(img)
                    imarray = np.array(im)
                    print(np.shape(imarray),np.unique(imarray))
                    nb_img += 1




if __name__ == '__main__':  

    #copy_files_format_for_pix2pix()
    convert_dataset_pix2pix_format(no_multiprocessing=True,max_number_img=2000000)
    #convert_dataset_pix2pix_format(path_dataset='/lustre/fsn1/projects/rech/abj/ujq24es/dataset/dp014_V1-2_FLAIR19_RVBIE',
    #                               output_path='/lustre/fsn1/projects/rech/abj/ujq24es/dataset/PixtoPix_FLAIR',max_number_img=20000000000,
    #                               path_csv_files='/lustre/fsn1/projects/rech/abj/ujq24es/dataset/FLAIR-INC')