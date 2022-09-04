import glob
import numpy as np
from osgeo import gdal 
from sklearn.preprocessing import MinMaxScaler

def normalization(data):
    scaler = MinMaxScaler() 
    data_scaled = scaler.fit_transform(data)
    return data_scaled

def auto_clear_sen2(image):
    new_img = image ** 3
    img_mean= new_img.mean()


def image_load(img_path,sentinel_type=1):
    try:
        images = [f for f in glob.glob(img_path+'/*.tif')] 
    except:
        print("[ERROR_PATH] Path is wrong!")
    image_list = []
    feature_list = []

    for img in images:
        print(img)
        data = gdal.Open(img, gdal.GA_ReadOnly) 
        if sentinel_type == 1:
            rgb = np.stack([normalization(data.GetRasterBand(b).ReadAsArray()) for b in range(1,4)] ,axis=2)
        if sentinel_type == 2:
            rgb = np.stack([(data.GetRasterBand(b).ReadAsArray()) for b in range(1,4)] ,axis=2)
            features = np.stack([normalization(data.GetRasterBand(b).ReadAsArray()) for b in range(10,13)] ,axis=2)
            feature_list.append(features)
        image_list.append(rgb)
    dataset = np.array(image_list, dtype="float")
    if sentinel_type == 2: 
        feature_dataset = np.array(feature_list, dtype="float")
        return dataset,feature_dataset
    return dataset  

def normalize(image):
    norm = (image - np.min(image)) / (np.max(image) - np.min(image))
    return (norm - 0.5)*2

def filter_night_images(images):
    filtered_imgs = [] 
    images_pow = normalize(images) ** 3 
    for imgs in images_pow:
        if imgs.mean() > -0.9:
            filtered_imgs.append(imgs)


def dataset(dataset_path,sentinel_type=1):
    if sentinel_type == 2:
        bands_dataset = np.empty(13,dtype=object)
        feature_dataset = np.empty(13,dtype=object)
        for station in range(1,14):
            tmp_dataset,tmp_feature_dataset = image_load(f'{dataset_path}{str(station)}',sentinel_type=2)
            bands_dataset[station-1] = tmp_dataset
            feature_dataset[station-1] = tmp_feature_dataset
            print(np.shape(feature_dataset))
        return np.concatenate(bands_dataset.tolist()),np.concatenate(feature_dataset.tolist())
    elif sentinel_type ==1:
        bands_dataset = np.empty(13,dtype=object)
        for station in range(1,14):
            tmp_dataset = image_load(f'{dataset_path}{str(station)}',sentinel_type=1)
            print(np.shape(tmp_dataset))
            bands_dataset[station-1] = tmp_dataset

            print(np.shape(bands_dataset))
        return np.concatenate(bands_dataset.tolist())
    else:
        print("[TYPE_ERROR] Given wrong sentinel type number! Please give '1' or '2' for progress.")
    





# sentinel_1_path= "./extracted_images/Sentinel1/Sentinel1_crop_"
# sentinel_2_path= "./extracted_images/Sentinel2/Sentinel2_crop_"
