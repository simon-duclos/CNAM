# -*- coding: utf-8 -*-
"""ALL COMMENTS IN FRENCH
Modèle semi-supervisé d'apprentissage sur MNIST 
"""

#Import et préparation des données
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

#Import des données MNIST et séparation native apprentissage et test
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Reshape pour passer d'une réprésentation (28,28) à (28,28,1) pour imagegenerator
X_train = X_train.reshape(X_train.shape[0],28,28,1)
X_test = X_test.reshape(X_test.shape[0], 28,28,1)
#Conversion de type
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#Normalisation
X_train /= 255
X_test /= 255

# Encodage disjonctif des sorties
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

#Séparation en données supervisées et non supervisées
nb_labels = 100#nombre d'exemples supervisés
X_train_unsupervised = X_train[nb_labels:,]
X_train_supervised = X_train[:nb_labels,]
Y_train_supervised = Y_train[:nb_labels,]

#Autoencodeur simple
from keras.models import Sequential ,Model
from keras.layers import Dense, Activation, Input, Flatten, Reshape
from keras.optimizers import SGD


#Fonction de précision - accuracy
def accuracy(y_true,y_pred):
  return  np.mean(np.equal(np.argmax(y_true, axis=-1),np.argmax(y_pred, axis=-1)))*100

#Autoencodeur convolutionnel
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras.models import Model
from keras.optimizers import SGD

nb_epoch = 30
batch_size = 32
learning_rate = 0.5
sgd = SGD(learning_rate)
weights = [1., 0]#Mettre [1,0] pour une performance supervisée 'de base'

input_encoder = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format
#Encodeur
hidden_encoder = Conv2D(16, (3, 3), activation='relu', padding='same')(input_encoder)
hidden_encoder = MaxPooling2D((2, 2), padding='same')(hidden_encoder)
hidden_encoder = Conv2D(8, (3, 3), activation='relu', padding='same')(hidden_encoder)
hidden_encoder = MaxPooling2D((2, 2), padding='same')(hidden_encoder)
hidden_encoder = Conv2D(8, (3, 3), activation='relu', padding='same')(hidden_encoder)
hidden_encoder = MaxPooling2D((2, 2), padding='same')(hidden_encoder)
# Bottleneck : représentation (4, 4, 8) i.e. 128-dimensional

#Sortie régression logistique vers les 10 classes de chiffres
output_encoder = Flatten()(hidden_encoder)
output_encoder = Dense(10, activation='softmax')(output_encoder)
#Décodeur
output_decoder = Conv2D(8, (3, 3), activation='relu', padding='same')(hidden_encoder)
output_decoder = UpSampling2D((2, 2))(output_decoder)
output_decoder = Conv2D(8, (3, 3), activation='relu', padding='same')(output_decoder)
output_decoder = UpSampling2D((2, 2))(output_decoder)
output_decoder = Conv2D(16, (3, 3), activation='relu')(output_decoder)
output_decoder = UpSampling2D((2, 2))(output_decoder)
output_decoder = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(output_decoder)

conv_autoencoder_model = Model(inputs=input_encoder, outputs=[output_encoder, output_decoder])
conv_autoencoder_model.compile(loss=['categorical_crossentropy','mean_squared_error'],loss_weights=weights,optimizer='adam',metrics=['accuracy'])
conv_autoencoder_model.fit( X_train_supervised,[Y_train_supervised,  X_train_supervised],batch_size=batch_size, 
                      epochs=nb_epoch,
                      verbose=1,
                      validation_data=(X_test, [Y_test,X_test])
                      )

#Résultats sur l'autoencoder convolutionnel
XY_train_supervised_predicted = conv_autoencoder_model.predict(X_train_supervised)
XY_test_predicted = conv_autoencoder_model.predict(X_test)
print("Train Accuracy = ",accuracy(Y_train_supervised,XY_train_supervised_predicted[0]))
print("Test Accuracy = ",accuracy(Y_test,XY_test_predicted[0]))
scores = conv_autoencoder_model.evaluate(X_test, [Y_test,X_test], verbose=0)
print("Modèle de perceptron multi-couches", conv_autoencoder_model.summary())
for i in range(5):
  print("%s: %.2f%%" % (conv_autoencoder_model.metrics_names[i], scores[i]*100))#71,3 for 100 images


from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rotation_range=8, 
                               width_shift_range=0.08, 
                               shear_range=0.3, 
                               height_shift_range=0.08, 
                               zoom_range=0.08 )
# fit parameters from data
datagen.fit(X_train)

def generate_generator_multiple(generator,input, output, batch_size):
  gen = datagen.flow(input,output,batch_size)
  while True:
            Xi = gen.next()
            yield Xi[0],[Xi[1], Xi[0]]

#Supervised learning
for i in range(2):
  inputgenerator=generate_generator_multiple(generator = datagen, input = X_train_supervised, output = Y_train_supervised, batch_size =32)
  conv_autoencoder_model.fit_generator(inputgenerator,steps_per_epoch=len(X_train_supervised) / 32, epochs=10)

#Unsupervised learning
Y_train_unsupervised_conv = conv_autoencoder_model.predict(X_train_unsupervised)[0]
inputgenerator_conv=generate_generator_multiple(generator = datagen, input = X_train_unsupervised, output = Y_train_unsupervised_conv, batch_size =32)
conv_autoencoder_model.fit_generator(inputgenerator,steps_per_epoch=len(X_train_unsupervised) / 32, epochs=10)

scores = conv_autoencoder_model.evaluate(X_test, [Y_test,X_test], verbose=0)
print("Modèle de perceptron multi-couches", conv_autoencoder_model.summary())
print("%s: %.2f%%" % (conv_autoencoder_model.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (conv_autoencoder_model.metrics_names[1], scores[1]*100))#71,3 for 100 images
