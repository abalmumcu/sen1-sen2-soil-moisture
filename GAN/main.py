from GAN_model import gan_model

image_shape = (256,256,3)

g_model_AtoB = gan_model.define_generator(image_shape)
g_model_BtoA = gan_model.define_generator(image_shape)
d_model_A = gan_model.define_discriminator(image_shape)
d_model_B = gan_model.define_discriminator(image_shape)
c_model_AtoB = gan_model.define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
c_model_BtoA = gan_model.define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)

gan_model.train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA,S2,S1)
