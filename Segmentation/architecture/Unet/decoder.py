#Doniyor Tropmann


import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
import logging
from keras import losses

from Segmentation.architecture.Unet.reuse import ReuseElements



class Decoder(object):

    def __init__(self,**kwargs):
        self._decoder_layers = []
        self._loss = kwargs.pop('loss', 'bce_dice_loss')
        #self.arch  = None

    def build_decoder(self, encoder):

        mask_out, inputs = self.decoder(encoder)
        model = models.Model(inputs=[inputs], outputs=[mask_out])
        # model.compile(optimizer=optimizers, loss=bce_dice_loss, metrics=[dice_loss, precision, recall, F1])

        return model

    def decoder(self, encoder):
        center, conv_pool, conv, inputs = encoder
        reuse = ReuseElements(self._decoder_layers)

        decoder4 = reuse(self._decoder_block(center, conv[4], 512))
        # 16data_aug_iter

        decoder3 = reuse(self._decoder_block(decoder4, conv[3], 256))
        # 32

        decoder2 = reuse(self._decoder_block(decoder3, conv[2], 128))
        # 64

        decoder1 = reuse(self._decoder_block(decoder2, conv[1], 64))
        # 128

        decoder0 = reuse(self._decoder_block(decoder1, conv[0], 32))
        # 256

        outputs = reuse(layers.Conv2D(1, (1, 1), activation='sigmoid'))(decoder0)

        # hidden = reuse(tfl.Dense(100, tf.nn.relu))(code)
        # hidden = reuse(tfl.Dense(64 * 64 * 3))(hidden)
        # recon = tf.reshape(recon, [-1, 64, 64, 3])
        return (outputs, inputs)

    def _decoder_block(self, input_tensor, concat_tensor, num_filters):
        decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
        decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        return decoder

    @property
    def loss(self):
        if self._loss == 'bce_dice_loss':
            print('Here Loss')

            self._loss = self.bce_dice_loss

        elif self._loss not in [self.bce_dice_loss]:
            print(type(self._loss))
            logging.error('Unrecognized loss type')
        return self._loss

    def dice_loss(self, mask_true, mask_pred):
        def dice_coeff(mask_true, mask_pred):
            smooth = 1.
            # Flatten
            mask_true_f = tf.reshape(mask_true, [-1])
            mask_pred_f = tf.reshape(mask_pred, [-1])
            intersection = tf.reduce_sum(mask_true_f * mask_pred_f)
            score = (2. * intersection + smooth) / (tf.reduce_sum(mask_true_f) + tf.reduce_sum(mask_pred_f) + smooth)
            return score

        return 1 - dice_coeff(mask_true, mask_pred)

    def bce_dice_loss(self, mask_true, mask_pred):
        # Compute balanced_dice_loss

        balanced_dice_loss = losses.binary_crossentropy(mask_true, mask_pred) + self.dice_loss(mask_true, mask_pred)

        return balanced_dice_loss