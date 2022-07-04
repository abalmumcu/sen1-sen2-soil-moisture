import os
import numpy as np
from osgeo import gdal 
import matplotlib.pyplot as plt


def image_load(img_path,sentinel_type=1):
    try:
        images = [f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))]
    except:
        print("[ERROR_PATH] Path is wrong!")
    image_list = []
    feature_list = []
    for img in images:
        image_list = []
        data = gdal.Open(img_path+'/'+ img, gdal.GA_ReadOnly) 
        rgb = np.stack([(data.GetRasterBand(b).ReadAsArray()) for b in (1,2,3)] ,axis=2)
        if sentinel_type == 2:
            features = np.stack([(data.GetRasterBand(b).ReadAsArray()) for b in (10,11,12)] ,axis=2)
            feature_list.append(features)

        image_list.append(rgb)
    dataset = np.array(image_list, dtype="float")
    if sentinel_type == 2: 
        feature_dataset = np.array(feature_list, dtype="float")
        return dataset,feature_dataset
    return dataset 

def dataset(dataset_path,sentinel_type=1):
    if sentinel_type == 2:
        bands_dataset = np.zeros(shape=(13,256,256,3))
        feature_dataset = np.zeros(shape=(13,256,256,3))
        for station in range(1,14):
            tmp_dataset,tmp_feature_dataset = image_load(dataset_path + str(station),sentinel_type=2)
            bands_dataset[station-1] = tmp_dataset
            feature_dataset[station-1] = tmp_feature_dataset
        return bands_dataset,feature_dataset
    elif sentinel_type ==1:
        bands_dataset = np.zeros(shape=(13,256,256,3))
        for station in range(1,14):
            tmp_dataset = image_load(dataset_path + str(station),sentinel_type=1)
            bands_dataset[station-1] = tmp_dataset
        return bands_dataset
    else:
        print("[TYPE_ERROR] Given wrong sentinel type number! Please give '1' or '2' for progress.")
    





sentinel_1_path= "./extracted_images/Sentinel1/Sentinel1_crop_"
sentinel_2_path= "./extracted_images/Sentinel2/Sentinel2_crop_"
