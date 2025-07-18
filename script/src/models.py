import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization

def build_simple_3dcnn(input_shape: tuple, config: dict) -> Model:
    """
    基本的な3DCNNモデルを構築する
    """

    dropout_rate = config['model']['params']['dropout_rate']

    img_input = Input(shape=input_shape, name='img_input')

    x = Conv3D()




    output = Dense(1, name='output')(x)

    model = Model(inputs=img_input, outputs=output)

    return model