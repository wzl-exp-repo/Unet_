#Doniyor Tropmann
#It's old version of architecture. it is not used


import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models


class ReuseElements(object):

  def __init__(self, elements):
    self._elements = elements
    self._adding = (len(elements) == 0)
    self._index = 0

  def __call__(self, provided):
    if self._adding:
      self._elements.append(provided)
      return provided
    existing = self._elements[self._index]
    self._index += 1
    assert isinstance(existing, type(provided))
    return existing





class Unet(object):

  def __init__(self,  image_shape = (64, 64, 3)):
    #self._input = input
    self._encoder_layers = []
    self._decoder_layers = []
    self.img_shape = image_shape
  '''
  @property
  def weights(self):
    weights = []
    for layer in self._encoder_layers:
      weights += layer.variables
    for layer in self._decoder_layers:
      weights += layer.variables
    return weights
  '''

  def build_model(self):
      inputs = layers.Input(shape=self.img_shape)
      encoder = self.encoder(inputs)
      print(encoder)
      mask_out = self.decoder(encoder)
      model = models.Model(inputs=[inputs], outputs=[mask_out])
      #model.compile(optimizer=optimizers, loss=bce_dice_loss, metrics=[dice_loss, precision, recall, F1])

      return model



  def encoder(self, input):
    reuse = ReuseElements(self._encoder_layers)

    inputs = reuse(input)
    # 256

    #print('Test', len(reuse(self._encoder_block(inputs, 32))))
    encoders_pool = []
    encoders = []

    encoders_pool0, encoders0 = reuse(self._encoder_block(inputs, 32))
    # 128

    encoders_pool1, encoders1 = reuse(self._encoder_block(encoders_pool0, 64))
    # 64

    encoders_pool2, encoders2 = reuse(self._encoder_block(encoders_pool1, 128))
    # 32

    encoders_pool3, encoders3 = reuse(self._encoder_block(encoders_pool2, 256))
    # 16

    encoders_pool4, encoders4 = reuse(self._encoder_block(encoders_pool3, 512))
    # 8

    center = reuse(self._conv_block(encoders_pool4, 1024))
    # center

    encoders_pool.extend((encoders_pool0, encoders_pool1, encoders_pool2, encoders_pool3, encoders_pool4))
    encoders.extend((encoders0, encoders1, encoders2, encoders3, encoders4))


    #hidden = reuse(tfl.Dense(100, tf.nn.relu))(data)
    #code = reuse(tfl.Dense(self._code_size))(hidden)
    return (center,encoders_pool,encoders)

  def decoder(self, encoder):
    center, conv_pool, conv = encoder
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


    #hidden = reuse(tfl.Dense(100, tf.nn.relu))(code)
    #hidden = reuse(tfl.Dense(64 * 64 * 3))(hidden)
    #recon = tf.reshape(recon, [-1, 64, 64, 3])
    return outputs





  def _conv_block(self, input_tensor, num_filters):
      encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
      encoder = layers.BatchNormalization()(encoder)
      encoder = layers.Activation('relu')(encoder)
      encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
      encoder = layers.BatchNormalization()(encoder)
      encoder = layers.Activation('relu')(encoder)
      return encoder

  def _encoder_block(self,input_tensor, num_filters):
      encoder = self._conv_block(input_tensor, num_filters)
      encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)

      return encoder_pool, encoder

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



  '''
  def dice_coeff(mask_true, mask_pred):
      smooth = 1.
      # Flatten
      mask_true_f = tf.reshape(mask_true, [-1])
      mask_pred_f = tf.reshape(mask_pred, [-1])
      intersection = tf.reduce_sum(mask_true_f * mask_pred_f)
      score = (2. * intersection + smooth) / (tf.reduce_sum(mask_true_f) + tf.reduce_sum(mask_pred_f) + smooth)
      return score



  def bce_dice_loss(self, mask_true, mask_pred):
      # Compute balanced_dice_loss

      def dice_loss(mask_true, mask_pred):
          return (1 - self.dice_coeff(mask_true, mask_pred))

      balanced_dice_loss = losses.binary_crossentropy(mask_true, mask_pred) + dice_loss(mask_true, mask_pred)

      return balanced_dice_loss
  '''