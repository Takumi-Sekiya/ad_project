import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, GlobalAveragePooling3D, Dense, Dropout, BatchNormalization
from tensorflow.keras.initializers import HeNormal

def build_simple_3dcnn(input_shape: tuple, config: dict) -> Model:
    """
    基本的な3DCNNモデルを構築する
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
    x = Dense(60, activation='linear', kernel_initializer=HeNormal())(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(20, activation='linear', kernel_initializer=HeNormal())(x)

    output = Dense(1, activation='linear', name='output')(x)

    model = Model(inputs=img_input, outputs=output)

    return model

MODEL_REGISTRY = {
    "Simple3DCNN": build_simple_3dcnn,
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