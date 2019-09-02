#Doniyor Tropmann

from tensorflow.python.keras import layers
from tensorflow.python.keras import models

from Segmentation.architecture.Unet.reuse import ReuseElements

class Encoder(object):

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

  def build_encoder(self):
      inputs = layers.Input(shape=self.img_shape)
      encoder = self.encoder(inputs)
      #print(encoder)
      #mask_out = self.decoder(encoder)
      #model = models.Model(inputs=[inputs], outputs=[mask_out])
      #model.compile(optimizer=optimizers, loss=bce_dice_loss, metrics=[dice_loss, precision, recall, F1])

      return encoder



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
    return (center,encoders_pool,encoders, inputs)







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