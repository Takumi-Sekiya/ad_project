import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv3D, MaxPooling3D, GlobalAveragePooling3D, Dense, Dropout, BatchNormalization,
    Concatenate, Add, Activation
)
from tensorflow.keras.initializers import HeNormal

def build_simple_3dcnn(input_shape: tuple, config: dict) -> Model:
    """
    基本的な3DCNNモデルを構築する.
    """
    dropout_rate = config['model']['params']['dropout_rate']

    # 画像入力
    img_input = Input(shape=input_shape, name='img_input')

    x = Conv3D(32, (5, 5, 5), padding='same', activation='relu')(img_input)
    x = Conv3D(32, (5, 5, 5), padding='same', activation='relu')(x)
    x = MaxPooling3D((2, 2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D((2, 2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv3D(128, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(128, (3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D((2, 2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv3D(256, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(256, (3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D((2, 2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv3D(512, (1, 1, 1), padding='same', activation='relu')(x)
    x = GlobalAveragePooling3D()(x)

    x = Dense(100, activation='relu', kernel_initializer=HeNormal())(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(60, activation='relu', kernel_initializer=HeNormal())(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(20, activation='relu', kernel_initializer=HeNormal())(x)

    output = Dense(1, activation='linear', name='output')(x)

    model = Model(inputs=img_input, outputs=output)

    return model


def build_multimodal_3dcnn(input_shape: tuple, config: dict) -> Model:
    """
    3DMRI画像と数値データを入力とするマルチモーダル回帰モデルを構築する.
    """
    dropout_rate = config['model']['params']['dropout_rate']
    numerical_features_dim = config['model']['params']['numerical_features_dim']

    # 1. 画像特徴抽出ブランチ
    img_input = Input(shape=input_shape, name='img_input')

    x = Conv3D(32, (5, 5, 5), padding='same', activation='relu')(img_input)
    x = Conv3D(32, (5, 5, 5), padding='same', activation='relu')(x)
    x = MaxPooling3D((2, 2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D((2, 2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv3D(128, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(128, (3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D((2, 2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv3D(256, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(256, (3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D((2, 2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv3D(512, (1, 1, 1), padding='same', activation='relu')(x)
    x = GlobalAveragePooling3D()(x)

    x = Dense(128, activation='relu', kernel_initializer=HeNormal())(x)
    img_features = Dense(32, activation='relu', kernel_initializer=HeNormal())(x)

    # 2. 数値データ入力ブランチ
    numerical_input = Input(shape=(numerical_features_dim, ), name='numerical_input')

    # 3. 特徴量の結合, 回帰
    combined_features = Concatenate()([img_features, numerical_input])

    y = Dense(64, activation='relu', kernel_initializer=HeNormal())(combined_features)
    y = Dropout(dropout_rate)(y)
    y = Dense(32, activation='relu', kernel_initializer=HeNormal())(combined_features)
    y = Dropout(dropout_rate)(y)

    output = Dense(1, activation='linear', name='output')(y)

    model = Model(inputs=[img_input, numerical_input], outputs=output)

    return model
    

def _resnet_basic_block(input_tensor, kernel_size, filters, strides=(1, 1, 1)):
    """
    3D ResNet18/34用のBasic Block.
    """
    x = Conv3D(filters, kernel_size, strides=strides, padding='same', activation='relu')(input_tensor)
    x = Conv3D(filters, kernel_size, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    shortcut = input_tensor
    if strides != (1, 1, 1) or input_tensor.shape[-1] != filters:
        shortcut = Conv3D(filters, (1, 1, 1), strides=strides)(input_tensor)
        shortcut = BatchNormalization()(shortcut)
    
    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x

def build_3d_resnet(input_shape: tuple, config: dict) -> Model:
    """
    3DResNetモデルを構築する.
    ResNet18に近いモデル.
    """
    dropout_rate = config['model']['params']['dropout_rate']

    img_input = Input(shape=input_shape, name='img_input')

    x = Conv3D(32, (5, 5, 5), padding='same', activation='relu')(img_input)
    x = MaxPooling3D((2, 2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    x = _resnet_basic_block(x, 3, 64)
    x = _resnet_basic_block(x, 3, 64)

    x = _resnet_basic_block(x, 3, 128, strides=(2, 2, 2))
    x = _resnet_basic_block(x, 3, 128)

    x = _resnet_basic_block(x, 3, 256, strides=(2, 2, 2))
    x = _resnet_basic_block(x, 3, 256)

    x = _resnet_basic_block(x, 3, 512, strides=(2, 2, 2))
    x = _resnet_basic_block(x, 3, 512)

    x = GlobalAveragePooling3D()(x)

    x = Dense(128, activation='relu', kernel_initializer=HeNormal())(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(32, activation='relu', kernel_initializer=HeNormal())(x)
    x = Dropout(dropout_rate)(x)

    output = Dense(1, activation='linear', name='output')(x)

    model = Model(inputs=img_input, outputs=output)

    return model


MODEL_REGISTRY = {
    "Simple3DCNN": build_simple_3dcnn,
    "Multimodal3DCNN": build_multimodal_3dcnn,
    "#DResNet": build_3d_resnet,
    # 新しいモデルを追加
}

def build_model(input_shape: tuple, config: dict) -> Model:
    """
    設定ファイルの情報に基づき, 指定されたモデルを動的に構築する
    """
    model_name = config['model']['name']
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' is not registered. Available models: {list(MODEL_REGISTRY.keys())}")
    
    build_fn = MODEL_REGISTRY[model_name]
    return build_fn(input_shape=input_shape, config=config)