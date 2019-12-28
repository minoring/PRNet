import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2

L2_WEIGHT_DECAY = 2e-4


#TODO: Check if it is correct model.
def conv_block(input_tensor, filters, kernel_size, strides=1):
  """A block that has a conv layer at shortcut.
  
  Reference:
    - https://github.com/tensorflow/models/blob/master/official/vision/image_classification/resnet_model.py
  """

  assert filters % 2 == 0  # filters must be divisible by channel_factor (2 here)
  shortcut = input_tensor
  if strides != 1 or input_tensor.get_shape()[3] != filters:
    shortcut = Conv2D(filters, (1, 1),
                      strides=strides,
                      kernel_regularizer=l2(L2_WEIGHT_DECAY),
                      padding='same')(shortcut)

  x = Conv2D(filters // 2, (1, 1),
             kernel_regularizer=l2(L2_WEIGHT_DECAY),
             padding='same')(input_tensor)
  x = tf.keras.layers.Activation('relu')(x)

  x = Conv2D(filters // 2,
             kernel_size,
             strides=strides,
             kernel_regularizer=l2(L2_WEIGHT_DECAY),
             padding='same')(x)
  x = tf.keras.layers.Activation('relu')(x)

  x = Conv2D(filters, (1, 1),
             kernel_regularizer=l2(L2_WEIGHT_DECAY),
             padding='same')(x)

  x = tf.keras.layers.add([x, shortcut])
  x = BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)

  return x


def resnet_FNC256(channel=3, is_training=True, batch_size=16, name='resfcn256'):
  """Returns ResNet model"""
  filters = 16
  # Encoder
  img_input = Input(shape=(256, 256, 3), batch_size=batch_size)
  x = Conv2D(filters, (4, 4), padding='same')(img_input)  # 256x256x16
  x = conv_block(x, filters=filters * 2, kernel_size=4, strides=2)  # 128x128x32
  x = conv_block(x, filters=filters * 2, kernel_size=4, strides=1)  # 128x128x32
  x = conv_block(x, filters=filters * 4, kernel_size=4, strides=2)  # 64x64x64
  x = conv_block(x, filters=filters * 4, kernel_size=4, strides=1)  # 64x64x64
  x = conv_block(x, filters=filters * 8, kernel_size=4, strides=2)  # 32x32x128
  x = conv_block(x, filters=filters * 8, kernel_size=4, strides=1)  # 32x32x128
  x = conv_block(x, filters=filters * 16, kernel_size=4, strides=2)  # 16x16x256
  x = conv_block(x, filters=filters * 16, kernel_size=4, strides=1)  # 16x16x256
  x = conv_block(x, filters=filters * 32, kernel_size=4, strides=2)  # 8x8x512
  x = conv_block(x, filters=filters * 32, kernel_size=4, strides=1)  # 8x8x512
  # Decoder
  x = Conv2DTranspose(filters * 32, (4, 4), padding='same')(x)  # 8x8x512
  x = Conv2DTranspose(filters * 16, (4, 4), strides=2,
                      padding='same')(x)  # 16x16x256
  x = Conv2DTranspose(filters * 16, (4, 4), padding='same')(x)  # 16x16x256
  x = Conv2DTranspose(filters * 16, (4, 4), padding='same')(x)  # 16x16x256
  x = Conv2DTranspose(filters * 8, (4, 4), strides=2,
                      padding='same')(x)  # 32x32x128
  x = Conv2DTranspose(filters * 8, (4, 4), padding='same')(x)  # 32x32x128
  x = Conv2DTranspose(filters * 8, (4, 4), padding='same')(x)  # 32x32x128
  x = Conv2DTranspose(filters * 4, (4, 4), strides=2,
                      padding='same')(x)  # 64x64x64
  x = Conv2DTranspose(filters * 4, (4, 4), padding='same')(x)  # 64x64x64
  x = Conv2DTranspose(filters * 4, (4, 4), padding='same')(x)  # 64x64x64

  x = Conv2DTranspose(filters * 2, (4, 4), strides=2,
                      padding='same')(x)  # 128x128x32
  x = Conv2DTranspose(filters * 2, (4, 4), strides=1,
                      padding='same')(x)  # 128x128x32
  x = Conv2DTranspose(filters, (4, 4), strides=2,
                      padding='same')(x)  # 256x256x16
  x = Conv2DTranspose(filters, (4, 4), strides=1,
                      padding='same')(x)  # 256x256x16

  x = Conv2DTranspose(3, (4, 4), strides=1, padding='same')(x)  # 256x256x3
  x = Conv2DTranspose(3, (4, 4), strides=1, padding='same')(x)  # 256x256x3
  pos = Conv2DTranspose(3, (4, 4),
                        strides=1,
                        padding='same',
                        activation='sigmoid')(x)

  return tf.keras.models.Model(inputs=img_input, outputs=pos, name=name)


class PosPrediction(object):

  def __init__(self, resolution_input=256, resolution_output=256):
    # Hyperparameter settings
    self.resolution_input = resolution_input
    self.resolution_output = resolution_output
    self.MaxPos = resolution_input * 1.1
    self.network = resnet_FNC256()

  def predict(self, input_image):
    pos = self.network.predict(input_image)
    return pos * self.MaxPos  # ???

  def restore(self, model_path):
    """Restore pre-trained model weights"""
    pass #TODO: It seems difficult to restore pre-trained tensorflow weights.
    # Write own training loop and load it.

if __name__ == '__main__':
  resnet = resnet_FNC256()
  resnet.summary()
  tf.keras.utils.plot_model(resnet, show_shapes=True)
