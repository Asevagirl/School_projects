#!/usr/bin/env python
# coding: utf-8

# # Exercice1

# In[1]:


#Charger l'architecture du reseau ResNet50
from keras.applications.resnet50 import ResNet50
model = ResNet50(include_top=True, weights='imagenet')


# In[2]:


model.layers.pop()


# In[3]:


from keras.models import Model
model = Model(input=model.input,output=model.layers[-1].output)


# In[4]:


model.summary()


# In[5]:


from keras.optimizers import SGD
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9), metrics=['binary_accuracy'])


# In[6]:


#Chargement des données VOC avec une fonction generatrice donnée:
from data_gen import PascalVOCDataGenerator
data_dir = '/Users/macbookibtissam/Desktop/tps_rn_deep/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/' # A changer avec votre chemin
data_generator_train = PascalVOCDataGenerator('trainval', data_dir)


# In[7]:


import numpy as np
batch_size = 32
generator = data_generator_train.flow(batch_size=batch_size)
# Initilisation des matrices contenant les Deep Features et les labels
X_train = np.zeros((len(data_generator_train.images_ids_in_subset),2048))
Y_train = np.zeros((len(data_generator_train.images_ids_in_subset),20))
# Calcul du nombre e batchs
nb_batches = int(len(data_generator_train.images_ids_in_subset) / batch_size) + 1

for i in range(nb_batches):
    # Pour chaque batch, on extrait les images d'entrée X et les labels y
    X, y = next(generator)
    # On récupère les Deep Feature par appel à predict
    y_pred = model.predict(X)
    X_train[i*batch_size:(i+1)*batch_size,:] = y_pred
    Y_train[i*batch_size:(i+1)*batch_size,:] = y


# In[8]:


#Idem pour la matrice des Deep Features sur la base de test
data_dir_test = '/Users/macbookibtissam/Desktop/tps_rn_deep/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/' # A changer avec votre chemin
data_generator_test = PascalVOCDataGenerator('test', data_dir_test)


# In[9]:


import numpy as np
batch_size = 32
generator = data_generator_test.flow(batch_size=batch_size)
# Initilisation des matrices contenant les Deep Features et les labels
X_test = np.zeros((len(data_generator_test.images_ids_in_subset),2048))
Y_test = np.zeros((len(data_generator_test.images_ids_in_subset),20))
# Calcul du nombre e batchs
nb_batches = int(len(data_generator_test.images_ids_in_subset) / batch_size) + 1

for i in range(nb_batches):
    # Pour chaque batch, on extrait les images d'entrée X et les labels y
    X, y = next(generator)
    # On récupère les Deep Feature par appel à predict
    y_pred = model.predict(X)
    X_test[i*batch_size:(i+1)*batch_size,:] = y_pred
    Y_test[i*batch_size:(i+1)*batch_size,:] = y


# In[10]:


#Sauvegarde des deep features et les labels:
outfile = 'DF_ResNet50_VOC2007'
np.savez(outfile, X_train=X_train, Y_train=Y_train,X_test=X_test, Y_test=Y_test)


# # Exercice 3:

# In[11]:


outfile = 'DF_ResNet50_VOC2007.npz'
npzfile = np.load(outfile)
X_train = npzfile['X_train']
Y_train = npzfile['Y_train']
X_test = npzfile['X_test']
Y_test = npzfile['Y_test']
print ("X_train=",X_train.shape, "Y_train=",Y_train.shape, " X_test=",X_test.shape, "Y_train=",Y_test.shape)


# In[12]:


#On va maintenant considérer les Deep Features comme les données d’entrée et définir un réseau de neurones complètement connectés sans couche cachée pour prédire les labels de sortie :
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(20,  input_dim=2048, name='fc1', activation='sigmoid'))
model.summary()


# In[13]:


#Compiler le modèle
from keras.optimizers import SGD
learning_rate = 0.1
sgd = SGD(learning_rate)
model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['binary_accuracy'])


# In[14]:


batch_size = 32
nb_epoch = 20
model.fit(X_train, Y_train,batch_size=batch_size, epochs=nb_epoch,verbose=1)
scores = model.evaluate(X_test, Y_test, verbose=0)
print("%s TEST: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("%s TEST: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[ ]:


from sklearn.metrics import average_precision_score
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)
AP_train = np.zeros(20)
AP_test = np.zeros(20)
for c in range(20):
    AP_train[c] = average_precision_score(Y_train[:, c], y_pred_train[:, c])
    AP_test[c] = average_precision_score(Y_test[:, c], y_pred_test[:, c])

print ("MAP TRAIN =", AP_train.mean()*100)
print ("MAP TEST =", AP_test.mean()*100)


# # Exercice4

# In[15]:


# Load ResNet50 architecture & its weights
model = ResNet50(include_top=True, weights='imagenet')
model.layers.pop()
# Modify top layers
x = model.layers[-1].output
x = Dense(data_generator_train.nb_classes, activation='sigmoid', name='predictions')(x)
model = Model(input=model.input,output=x)


# In[16]:


nlayers=20
for i in range(nlayers):
    model.layers[i].trainable = True


# In[ ]:


lr = 0.1
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=lr), metrics=['binary_accuracy'])


# In[ ]:


batch_size=32
nb_epochs=10
data_generator_train = PascalVOCDataGenerator('trainval', data_dir)
steps_per_epoch_train = int(len(data_generator_train.id_to_label) / batch_size) + 1
model.fit_generator(data_generator_train.flow(batch_size=batch_size),
                    steps_per_epoch=steps_per_epoch_train,
                    epochs=nb_epochs,
                    verbose=1)


# In[ ]:


import numpy as np
from sklearn.metrics import average_precision_score
from data_gen import PascalVOCDataGenerator

default_batch_size = 200
default_data_dir = '/data/VOCdevkit/VOC2007/'

def evaluate(model, subset, batch_size=default_batch_size, data_dir=default_data_dir, verbose=0):
    """evaluate
    Compute the mean Average Precision metrics on a subset with a given model

    :param model: the model to evaluate
    :param subset: the data subset
    :param batch_size: the batch which will be use in the data generator
    :param data_dir: the directory where the data is stored
    :param verbose: display a progress bar or not, default is no (0)
    """
    #disable_tqdm = (verbose == 0)

    # Create the generator on the given subset
    data_generator = PascalVOCDataGenerator(subset, data_dir)
    steps_per_epoch = int(len(data_generator.id_to_label) / batch_size) + 1

    # Get the generator
    generator = data_generator.flow(batch_size=batch_size)

    y_all = []
    y_pred_all = []
    for i in range(steps_per_epoch):
        # Get the next batch
        X, y = next(generator)
        y_pred = model.predict(X)
        # We concatenate all the y and the prediction
        for y_sample, y_pred_sample in zip(y, y_pred):
            y_all.append(y_sample)
            y_pred_all.append(y_pred_sample)
    y_all = np.array(y_all)
    y_pred_all = np.array(y_pred_all)

    # Now we can compute the AP for each class
    AP = np.zeros(data_generator.nb_classes)
    for cl in range(data_generator.nb_classes):
        AP[cl] = average_precision_score(y_all[:, cl], y_pred_all[:, cl])

    return AP

