"""VGGFace model for Keras.

Source:
https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/
"""
from keras.layers import (
    Activation,
    Convolution2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
    ZeroPadding2D,
)
from keras.models import Model, Sequential


def load_vgg_face(input_shape=(224, 224, 3)):
    """Load the VGGFace model."""
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(4096, (7, 7), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation("softmax"))

    # pre-trained weights of vgg-face model.
    # you can find it here:
    # https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5
    # related blog post: https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/
    model.load_weights("data/vgg_face_weights.h5")
    return model


def get_base_model(input_shape):
    """Get the base model."""
    base_model = load_vgg_face(input_shape)
    base_model.trainable = False
    for layer in base_model.layers[-4:]:
        layer.trainable = True

    # TODO: move this to apply_top_layers func
    x = Sequential()
    x = Convolution2D(1024, (1, 1))(base_model.layers[-4].output)
    x = Flatten()(x)
    outputs = Dense(256, activation="relu")(x)
    age_model = Model(inputs=base_model.input, outputs=outputs)
    return age_model
