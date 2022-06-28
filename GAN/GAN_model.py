import tensorflow as tf
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.layers import Input
from keras.models import Model
from keras.layers import Conv2D, LeakyReLU, Activation, Concatenate, Conv2DTranspose
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
# from keras.preprocessing.image import img_to_array, load_img
import keras.backend as K
from matplotlib import pyplot
import random
import numpy as np

class gan_model():
    def preprocessing_generator_encoder(in_image):
        init = tf.keras.initializers.RandomNormal(stddev=0.02)

        g = Conv2D(16, (7,7), padding='same', kernel_initializer=init)(in_image)
        g = InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)

        g = Conv2D(32, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)

        g = Conv2D(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)
        return g


    def preprocessing_generator_decoder(g):
        init = RandomNormal(stddev=0.02)
        g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)
        g = Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)
        g = Conv2D(3, (3,3), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization(axis=-1)(g)
        g = Activation('tanh')(g)
        return g


    def define_discriminator(image_shape):
        init_disc = RandomNormal(stddev=0.02) #weight init
        input_image = Input(shape=image_shape)  #input
        d = Conv2D(16, (4,4), strides=(2,2), padding='same', kernel_initializer=init_disc)(input_image)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(32, (4,4), strides=(2,2), padding='same',  kernel_initializer=init_disc)(d)
        d = InstanceNormalization(axis=-1)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init_disc)(d)
        d = InstanceNormalization(axis=-1)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init_disc)(d)
        d = InstanceNormalization(axis=-1)(d)
        d = LeakyReLU(alpha=0.2)(d)
        # d = Conv2D(512, (4,4), padding='same', kernel_initializer=init_disc)(d)
        # d = InstanceNormalization(axis=-1)(d)
        # d = LeakyReLU(alpha=0.2)(d)
        output_image = Conv2D(1, (4,4), padding='same', kernel_initializer=init_disc)(d)
        model = Model(input_image, output_image)
        model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
        return model


    def resnet_block(n_filters, input_layer):
        init = RandomNormal(stddev=0.02)
        g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
        g = InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)
        g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization(axis=-1)(g)
        g = Concatenate()([g, input_layer])
        return g



    def define_generator(image_shape=(256,256,3), n_resnet=6):
        init = RandomNormal(stddev=0.02)
        in_image = Input(shape=image_shape)

        #g = preprocessing_generator_encoder(in_image)
        #g = preprocessing_generator_decoder(g)

        g = gan_model.preprocessing_generator_encoder(in_image)

        for _ in range(n_resnet):
            g = gan_model.resnet_block(128, g)

        out_image = gan_model.preprocessing_generator_decoder(g)

        model = Model(in_image, out_image)
        return model

    def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
        g_model_1.trainable = True
        d_model.trainable = False
        g_model_2.trainable = False
        # Adverisal Model
        input_gen = Input(shape=image_shape)
        gen1_out = g_model_1(input_gen)
        output_d = d_model(gen1_out)
        # Identity Model
        input_id = Input(shape=image_shape)
        output_id = g_model_1(input_id)
        # Forward Model
        output_f = g_model_2(gen1_out)
        # Backward cycle
        gen2_out = g_model_2(input_id)
        output_b = g_model_1(gen2_out)
        model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
        model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=Adam(lr=0.0002, beta_1=0.5))
        return model

    def generate_real_samples(dataset, n_samples, patch_shape):
        ix = np.random.randint(0, dataset.shape[0], n_samples)
        # retrieve selected images
        X = dataset[ix]
        y = np.ones((n_samples, patch_shape, patch_shape, 1))
        return X, y


    def generate_fake_samples(g_model, dataset, patch_shape=256):
        X = g_model.predict(dataset)
        y = np.zeros((len(X), patch_shape, patch_shape, 1))
        return X, y

    def update_image_pool(pool, images, max_size=50):
        selected = list()
        for image in images:
            if len(pool) < max_size:
                pool.append(image)
                selected.append(image)
            elif random() < 0.5:
                selected.append(image)
            else:
                ix = random.randint(0, len(pool))
                selected.append(pool[ix])
                pool[ix] = image
        return np.asarray(selected)


    def save_models(step, g_model_AtoB, g_model_BtoA , d_model_A, d_model_B):
    # save the first generator model
        filename1 = './Risma/g_model_AtoB_%06d.h5' % (step+1)
        g_model_AtoB.save(filename1)

        # save the second generator model
        filename2 = './Risma/g_model_BtoA_%06d.h5' % (step+1)
        g_model_BtoA.save(filename2)

        filename3 = './Risma/d_model_A_%06d.h5' % (step+1)
        d_model_A.save(filename3)
        filename4 = './Risma/d_model_B_%06d.h5' % (step+1)
        d_model_B.save(filename4)
        print('>Saved: %s and %s and %s and %s' % (filename1, filename2, filename3, filename4))

        
        
    def summarize_performance(step, g_model, trainX, name, n_samples=5):
        #n_samples = len(trainX)
        X_in, _ = gan_model.generate_real_samples(trainX, n_samples, 0)
        X_out, _ = gan_model.generate_fake_samples(g_model, X_in, 0)
        X_in = (X_in + 1) / 2.0
        X_out = (X_out + 1) / 2.0

        n_samples = 5
        for i in range(n_samples):
            pyplot.subplot(2, n_samples, 1 + i)
            pyplot.axis('off')
            pyplot.imshow(X_in[i])
        for i in range(n_samples):
            pyplot.subplot(2, n_samples, 1 + n_samples + i)
            pyplot.axis('off')
            pyplot.imshow(X_out[i])
        filename1 = './Risma/%s_generated_plot_%06d.png' % (name, (step+1))
        pyplot.savefig(filename1)
        pyplot.close()

    def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, trainA, trainB):
        n_epochs, n_batch, = 20, 1
        n_patch = d_model_A.output_shape[1]
        poolA, poolB = list(), list()
        n_steps = int(len(trainA)) * n_epochs
        learning_rate = 0.0002

        for i in range(n_steps):
            X_realA, y_realA = gan_model.generate_real_samples(trainA, n_batch, n_patch)
            X_realB, y_realB = gan_model.generate_real_samples(trainB, n_batch, n_patch)
            X_fakeA, y_fakeA = gan_model.generate_fake_samples(g_model_BtoA, X_realB, n_patch)
            X_fakeB, y_fakeB = gan_model.generate_fake_samples(g_model_AtoB, X_realA, n_patch)
            #X_fakeA = update_image_pool(poolA, X_fakeA)
            #X_fakeB = update_image_pool(poolB, X_fakeB)
            g_loss2, _, _, _, _  = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
            dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
            dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
            g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
            dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
            dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
            # summarize_performance(i, g_model_AtoB, trainA, 'S2toS1')
            # summarize_performance(i, g_model_BtoA, trainB, 'S1toS2')
            print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dA_loss1, dA_loss2, dB_loss1, dB_loss2, g_loss1, g_loss2))
            if (i+1) % int(len(trainA)) == 0:
                gan_model.summarize_performance(i, g_model_AtoB, trainA, 'S2toS1')
                gan_model.summarize_performance(i, g_model_BtoA, trainB, 'S1toS2')
                gan_model.save_models(i, g_model_AtoB, g_model_BtoA , d_model_A, d_model_B)

                learning_rate *= 0.90
                K.set_value(c_model_AtoB.optimizer.lr , learning_rate)
                K.set_value(c_model_BtoA.optimizer.lr , learning_rate)