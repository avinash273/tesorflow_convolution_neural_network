# Shanker, Avinash
# 1001-668-570
# 2019-11_02
# Assignment-04-03

import pytest
import numpy as np
from cnn import CNN
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, MaxPooling2D
from keras.layers.convolutional import Conv2D
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import SGD

def test_model_evaluate():
    my_cnn = CNN()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    #Comment if you want on entire dataset
    x_train = x_train[0:600,:]
    x_test = x_test[0:100]
    y_train = y_train[0:600,:]
    y_test = y_test[0:100]
    x_train=x_train.astype('float32')/255.0
    x_test=x_test.astype('float32')/255.0
    
    #Traditional Model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation = 'relu',input_shape=(32, 32,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))    
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu')) 
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    Optimizer = SGD(lr=0.01, momentum=0.0)
    model.compile(optimizer=Optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train,batch_size=32, epochs=10)
    Correct_Eval = model.evaluate(x_test, y_test) 
    print(model.summary())
    
    #My test model
    my_cnn.add_input_layer(shape=x_train.shape[1:], name="input")
    my_cnn.append_conv2d_layer(num_of_filters=32, kernel_size = 3, padding='same', strides=1, activation='relu')
    my_cnn.append_maxpooling2d_layer(pool_size=(2, 2), name='pool1')
    my_cnn.append_conv2d_layer(num_of_filters=64, kernel_size = 3, padding='same', strides=1, activation='relu')
    my_cnn.append_flatten_layer(name='flat')
    my_cnn.append_dense_layer(num_nodes=512, activation='relu', trainable=True)
    my_cnn.append_dense_layer(num_nodes=10, activation='softmax', trainable=True)
    Optimizer = SGD(lr=0.01, momentum=0.0)
    my_cnn.model.compile(optimizer = Optimizer, loss='categorical_crossentropy', metrics = ['accuracy'])
    my_cnn.model.fit(x_train, y_train,batch_size=32,epochs=10)
    Mycnn_Eval = my_cnn.model.evaluate(x_test, y_test)
    print("\nCorrect_Eval: ",Correct_Eval)
    print("\nMycnn_Eval: ",Mycnn_Eval)
    assert np.allclose(Correct_Eval, Mycnn_Eval, atol=1e-1*5, rtol=1e-1)

def test_model_train():
    my_cnn = CNN()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    #Comment if you want on entire dataset
    x_train = x_train[0:600,:]
    x_test = x_test[0:100]
    y_train = y_train[0:600,:]
    y_test = y_test[0:100]
    x_train=x_train.astype('float32')/255.0
    x_test=x_test.astype('float32')/255.0
    
    #Tradition Model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation = 'relu',input_shape=(32, 32,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))    
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu')) 
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    Optimizer = SGD(lr=0.01, momentum=0.0)
    model.compile(optimizer=Optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    Correct_Loss_History = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test,y_test))
    
    #My CNN Model
    my_cnn.add_input_layer(shape=x_train.shape[1:], name="input")
    my_cnn.append_conv2d_layer(num_of_filters=32, kernel_size = 3, padding='same', strides=1, activation='relu')
    my_cnn.append_maxpooling2d_layer(pool_size=(2, 2), name='pool1')
    my_cnn.append_conv2d_layer(num_of_filters=64, kernel_size = 3, padding='same', strides=1, activation='relu')
    my_cnn.append_flatten_layer(name='flat')
    my_cnn.append_dense_layer(num_nodes=512, activation='relu', trainable=True)
    my_cnn.append_dense_layer(num_nodes=10, activation='softmax', trainable=True)
    my_cnn.model.compile(loss='categorical_crossentropy', optimizer = Optimizer, metrics = ['accuracy'])
    my_cnn_Loss_History = my_cnn.model.fit(x_train, y_train, batch_size=32,epochs=10)

    print("\nCorrect_Loss_History.history['loss']: ",Correct_Loss_History.history['loss'])
    print("\nmy_cnn_Loss_History.history['loss']: ",my_cnn_Loss_History.history['loss'])
    
    assert np.allclose(Correct_Loss_History.history['loss'], my_cnn_Loss_History.history['loss'], atol=1e-1*6, rtol=1e-3)
    
test_model_evaluate()    
test_model_train()



#References
#https://www.programcreek.com/python/example/89658/keras.layers.Conv2D
#https://keras.io/layers/convolutional/
#http://faroit.com/keras-docs/2.0.2/getting-started/sequential-model-guide/
#https://www.tensorflow.org/guide/keras/train_and_evaluate
#https://keras.io/examples/cifar10_cnn/
#https://www.programcreek.com/python/example/101388/keras.datasets.cifar10.load_data
#https://keras.io/metrics/
#https://keras.io/utils/
#https://keras.io/getting-started/sequential-model-guide/