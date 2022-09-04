from GAN_model import gan_model
from utils import dataset,image_load,filter_night_images
import numpy as np

sentinel_1_path= "/Users/alperbalmumcu/Github/sen1-sen2-soil-moisture/extracted_images/Sentinel1/Sentinel1_crop_"
sentinel_2_path= "/Users/alperbalmumcu/Github/sen1-sen2-soil-moisture/extracted_images/Sentinel2/Sentinel2_crop_"

# Sen1 = image_load(sentinel_1_path,sentinel_type= 1)
Sen1 = dataset(sentinel_1_path,sentinel_type=1)
print("[SENTINEL 1] dataset created")
Sen2 = dataset(sentinel_2_path,sentinel_type=2)
print("[SENTINEL 2] dataset created")

image_shape = (256,256,3)

g_model_AtoB = gan_model.define_generator(image_shape)
g_model_BtoA = gan_model.define_generator(image_shape)
d_model_A = gan_model.define_discriminator(image_shape)
d_model_B = gan_model.define_discriminator(image_shape)
c_model_AtoB = gan_model.define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
c_model_BtoA = gan_model.define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)


if __name__ == "__main__":
    print(Sen1.shape)
    print(Sen2.shape)
    print("Mean S1: ",Sen1.mean())
    print("Mean S2: ",Sen2.mean())
    print("Min S1: ",Sen1.min())
    print("Min S2: ", Sen2.min())
    gan_model.train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA,Sen2,Sen1)

    # g_model_AtoB.predict(Sen2[0])
