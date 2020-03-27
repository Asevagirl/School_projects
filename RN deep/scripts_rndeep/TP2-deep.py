#!/usr/bin/env python
# coding: utf-8

# #### Exercice 1: regression logistique avec keras

# In[35]:


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D


# In[36]:


# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(len(y_test))
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# In[25]:


#Creation d'un reseau de neurones vide
model = Sequential()


# In[4]:


#Ajout d'une couche cachée et d'une fonction d'activation
model.add(Dense(10,input_dim = 784,name = 'fc1'))
model.add(Activation('softmax'))


# In[5]:


#Visualisation de l'architecture
model.summary()


# In[6]:


#compilation avec une foction loss
learning_rate = 0.1
sgd = SGD(learning_rate)

model.compile(loss = 'categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])


# In[7]:


#Phase d'apprentissage (fit)
batch_size = 100
nb_epoch = 10

Y_train = np_utils.to_categorical(y_train,10)
Y_test = np_utils.to_categorical(y_test,10)

model.fit(X_train,Y_train,batch_size = batch_size,nb_epoch = nb_epoch,verbose = 1)


# In[8]:


#Evaluer les performances du modèle dur l'ensemble de test:
scores = model.evaluate(X_test, Y_test, verbose=0)

#Print de la fonction cout
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))

#print de l'accuracy
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# #### Exercice2: Perceptron avec keras 

# In[9]:


#Creation d'un reseau de neurones vide
model2 = Sequential()

#Ajout d'une couche cachée et d'une fonction d'activation
model2.add(Dense(100,  input_dim=784, name='fc1'))
model2.add(Activation('sigmoid'))
model2.add(Dense(10))
model2.add(Activation('softmax'))


# In[10]:


model2.summary()


# In[11]:


#compilation avec une foction loss
learning_rate = 1.0
sgd = SGD(learning_rate)

model2.compile(loss = 'categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])


# In[12]:


#Phase d'apprentissage (fit)
batch_size = 100
nb_epoch = 10 #faut changer en 100 pour plus d 'accuracy'

Y_train = np_utils.to_categorical(y_train,10)
Y_test = np_utils.to_categorical(y_test,10)

model2.fit(X_train,Y_train,batch_size = batch_size,nb_epoch = nb_epoch,verbose = 1)


# In[13]:


#Evaluer les performances du modèle dur l'ensemble de test:
scores = model2.evaluate(X_test, Y_test, verbose=0)

#Print de la fonction cout
print("%s: %.2f%%" % (model2.metrics_names[0], scores[0]*100))

#print de l'accuracy
print("%s: %.2f%%" % (model2.metrics_names[1], scores[1]*100))


# In[14]:


#Suvegarder le modèle
from keras.models import model_from_yaml
def saveModel(model, savename):
  # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(savename+".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    print("Yaml Model ",savename,".yaml saved to disk")
      # serialize weights to HDF5
    model.save_weights(savename+".h5")
    print("Weights ",savename,".h5 saved to disk")

saveModel(model2,'perceptron keras')


# #### Exercice3: Reseau de neurones convolutifavec keras: 

# In[26]:


#Implementation d'un ConvNet
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)


# In[27]:


model3 = Sequential()


# In[28]:


#Convolution
model3.add(Conv2D(16,kernel_size=(5, 5),activation='relu',input_shape=(28, 28, 1),padding='valid'))
#Max-pooling avec un décalage de 2(par défaut)
model3.add(MaxPooling2D(pool_size=(2, 2)))

model3.add(Conv2D(32,kernel_size=(5, 5),activation='relu',input_shape=(28, 28, 1),padding='valid'))
model3.add(MaxPooling2D(pool_size=(2, 2)))

#Flatten
model3.add(Flatten())

#Ajout de couche completement connectées:
model3.add(Dense(100,  input_dim=512, name='fc1'))
model3.add(Activation('sigmoid'))
model3.add(Dense(10,  input_dim=100, name='fc2'))
model3.add(Activation('softmax'))


# In[29]:


#compilation avec une foction loss
learning_rate = 1.0
sgd = SGD(learning_rate)

model3.compile(loss = 'categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

#Phase d'apprentissage (fit)
batch_size = 100
nb_epoch = 10

Y_train = np_utils.to_categorical(y_train,10)
Y_test = np_utils.to_categorical(y_test,10)

model3.fit(X_train,Y_train,batch_size = batch_size,nb_epoch = nb_epoch,verbose = 1)



# In[30]:



#Evaluer les performances du modèle dur l'ensemble de test:
scores = model3.evaluate(X_test, Y_test, verbose=0)

#Print de la fonction cout
print("%s: %.2f%%" % (model3.metrics_names[0], scores[0]*100))

#print de l'accuracy
print("%s: %.2f%%" % (model3.metrics_names[1], scores[1]*100))


# #### Exercice 4: Visualition avec t-SNE: 

# In[37]:


import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.mixture import GaussianMixture
from scipy import linalg
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE


# In[38]:



instance = TSNE(n_components=2,init='pca',perplexity=30,verbose = 2)


# In[40]:


x2d = instance.fit_transform(X_test.reshape(10000, 784))


# In[ ]:


#Calcul de l’enveloppe convexe des points projetés pour chacune des classe classe.
def convexHulls(points, labels):
    convex_hulls = []
    for i in range(10):
        convex_hulls.append(ConvexHull(points[labels==i,:]))
    return convex_hulls
# Function Call
#convex_hulls= convexHulls(x2d, labels)


# In[ ]:


#Calcul de l’ellipse de meilleure approximation des points.
def best_ellipses(points, labels):
    gaussians = []
    for i in range(10):
        gaussians.append(GaussianMixture(n_components=1, covariance_type='full').fit(points[labels==i, :]))
    return gaussians
# Function Call
#ellipses = best_ellipses(x2d, labels)


# In[ ]:


#Calcul du « Neighborhood Hit » (NH) 
def neighboring_hit(points, labels):
    k = 6
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)

    txs = 0.0
    txsc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    nppts = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for i in range(len(points)):
        tx = 0.0
    for j in range(1,k+1):
        if (labels[indices[i,j]]== labels[i]):
            tx += 1
    tx /= k
    txsc[labels[i]] += tx
    nppts[labels[i]] += 1
    txs += tx

    for i in range(10):
        txsc[i] /= nppts[i]

    return txs / len(points)
    return txs / len(points)


# In[ ]:


def visualization(points2D, labels, convex_hulls, ellipses ,projname, nh):
    points2D_c= []
    for i in range(10):
          points2D_c.append(points2D[labels==i, :])
    # Data Visualization
    cmap =cm.tab10

    plt.figure(figsize=(3.841, 7.195), dpi=100)
    plt.set_cmap(cmap)
    plt.subplots_adjust(hspace=0.4 )
    plt.subplot(311)
    plt.scatter(points2D[:,0], points2D[:,1], c=labels,  s=3,edgecolors='none', cmap=cmap, alpha=1.0)
    plt.colorbar(ticks=range(10))

    plt.title("2D "+projname+" - NH="+str(nh*100.0))

    vals = [ i/10.0 for i in range(10)]
    sp2 = plt.subplot(312)
    for i in range(10):
        ch = np.append(convex_hulls[i].vertices,convex_hulls[i].vertices[0])
        sp2.plot(points2D_c[i][ch, 0], points2D_c[i][ch, 1], '-',label='$%i$'%i, color=cmap(vals[i]))
    plt.colorbar(ticks=range(10))
    plt.title(projname+" Convex Hulls")
    

    def plot_results(X, Y_, means, covariances, index, title, color):
        splot = plt.subplot(3, 1, 3)
        for i, (mean, covar) in enumerate(zip(means, covariances)):
            v, w = linalg.eigh(covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])
          # as the DP will not use every component it has access to
          # unless it needs it, we shouldn't plot the redundant
          # components.
            if not np.any(Y_ == i):
                  continue
            plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color, alpha = 0.2)

          # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.6)
            splot.add_artist(ell)

        plt.title(title)
    plt.subplot(313)

    for i in range(10):
        plot_results(points2D[labels==i, :], ellipses[i].predict(points2D[labels==i, :]), ellipses[i].means_,
        ellipses[i].covariances_, 0,projname+" fitting ellipses", cmap(vals[i]))

    plt.savefig(projname+".png", dpi=100)
    plt.show()


# In[ ]:


points=x2d
labels=Y_test.argmax(1)
n=neighboring_hit(points, labels)
b=best_ellipses(points, labels)
c=convexHulls(points, labels)
visualization(points, labels, c, b ,"t-sne", 1)


# In[ ]:


from keras.models import model_from_yaml
def loadModel(savename):
    with open(savename+".yaml", "r") as yaml_file:
        model = model_from_yaml(yaml_file.read())
    print ("Yaml Model ",savename,".yaml loaded ")
    model.load_weights(savename+".h5")
    print ("Weights ",savename,".h5 loaded ")
    return model


# In[ ]:


model = loadModel('perceptron keras')
model.pop()
model.predict(X_test)


# In[ ]:


instance_bis = TSNE(n_components=2,init='pca',perplexity=30,verbose = 2)


# In[ ]:


x2d_bis = instance_bis.fit_transform(X_train[:500,:])
labels = y_train[:500]


# In[ ]:


convex_hulls= convexHulls(x2d_bis, labels)
ellipses = best_ellipses(x2d_bis, labels)
visualization(x2d,labels,convex_hulls,ellipses,'proj1',1)


# In[ ]:




