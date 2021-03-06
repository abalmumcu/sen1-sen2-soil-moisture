from GAN.GAN_model import gan_model
from GAN.utils import dataset

sentinel_1_path= "./extracted_images/Sentinel1/Sentinel1_crop_"
sentinel_2_path= "./extracted_images/Sentinel2/Sentinel2_crop_"

Sen1 = dataset(sentinel_1_path,sentinel_type= 1)
Sen2,S2_Features = dataset(sentinel_2_path,sentinel_type= 2)

image_shape = (256,256,3)

g_model_AtoB = gan_model.define_generator(image_shape)
g_model_BtoA = gan_model.define_generator(image_shape)
d_model_A = gan_model.define_discriminator(image_shape)
d_model_B = gan_model.define_discriminator(image_shape)
c_model_AtoB = gan_model.define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
c_model_BtoA = gan_model.define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)

gan_model.train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA,Sen2,Sen1)
