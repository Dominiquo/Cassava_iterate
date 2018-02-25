from keras.engine.topology import Input
from keras.models import Model
from keras.layers.core import Activation, Reshape, Dense
from keras.utils import plot_model
import ENet.encoder as encoder
import ENet.decoder as decoder



def autoencoder(num_classes, input_shape,
                loss='categorical_crossentropy',
                optimizer='adadelta'):
    h,w = input_shape
    data_shape = h*w
    inp = Input(shape=(h, w, 3))
    enet = encoder.build(inp)
    enet = decoder.build(enet, nc=num_classes, in_shape=input_shape)
    enet = Reshape((data_shape, num_classes))(enet)
    enet = Activation('softmax')(enet)
    model = Model(inputs=inp, outputs=enet)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'mean_squared_error'])
    name = 'enet'
    print(model.summary())
    return model, name
