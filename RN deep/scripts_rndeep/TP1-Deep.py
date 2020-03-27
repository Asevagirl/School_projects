#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import mnist
# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# In[3]:


import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('TKAgg')
plt.figure(figsize=(7.195, 3.841), dpi=100)
for i in range(200):
    plt.subplot(10,20,i+1)
    plt.imshow(X_train[i,:].reshape([28,28]), cmap='gray')
    print(X_train[i,:].size)
    plt.axis('off')
plt.show()


# In[4]:


from keras.utils import np_utils
import numpy as np
K=10
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, K)
Y_test = np_utils.to_categorical(y_test, K)


# In[5]:


def forward(batch, W, b):
    return (batch.dot(W)+b)


# In[6]:


def softmax(X):
    E = np.exp(X)
    return (E.T / np.sum(E,axis=1)).T

def sigmoide(X):
    return (1/(1+np.exp(-X)))


# In[7]:


N = X_train.shape[0]
d = X_train.shape[1]
W = np.zeros((d,K))
b = np.zeros((1,K))
numEp = 20 # Number of epochs for gradient descent
eta = 1e-1 # Learning rate
batch_size = 100
nb_batches = int(float(N) / batch_size)
gradW = np.zeros((d,K))
gradb = np.zeros((1,K))

for epoch in range(numEp):
    for ex in range(nb_batches):
        batch = X_train[ex*batch_size:(ex+1)*batch_size,]
        Y_pred=softmax(forward(batch, W, b))
        dW=(batch.T).dot(Y_pred-(Y_train[ex*batch_size:(ex+1)*batch_size,]))/batch_size
        db=sum(Y_pred-Y_train[ex*batch_size:(ex+1)*batch_size,])/batch_size
        W= W-(eta*dW)
        b=b-eta*db


# In[8]:


def accuracy(W, b, images, labels):
    pred = forward(images, W,b )
    return np.where( pred.argmax(axis=1) != labels.argmax(axis=1) , 0.,1.).mean()*100.0


# In[9]:


accuracy(W,b,X_test,Y_test)


# In[9]:


def forward1(batch, Wh, bh, Wy, by):
    u= batch.dot(Wh)+bh
    h= sigmoide(u)
    v=h.dot(Wy)+by
    return (h,v)


def accuracy(Wh, bh, Wy, by, images, labels):
    h,pred = forward1(images, Wh, bh, Wy, by)
    return np.where( pred.argmax(axis=1) != labels.argmax(axis=1) , 0.,1.).mean()*100.0


# In[10]:


N = X_train.shape[0]
d = X_train.shape[1]
Wh = np.zeros((d,100))
Wy = np.zeros((100,K))
bh = np.zeros((1,100))
by = np.zeros((1,K))
numEp = 100 # Number of epochs for gradient descent
eta = 1# Learning rate
batch_size = 100
nb_batches = int(float(N) / batch_size)
gradW = np.zeros((d,K))
gradb = np.zeros((1,K))


for epoch in range(numEp):
    for ex in range(nb_batches):
        batch = X_train[ex*batch_size:(ex+1)*batch_size,]
        H,V=forward1(batch, Wh, bh,Wy,by)
        V= softmax(V)
        dWy=(H.T).dot(V-(Y_train[ex*batch_size:(ex+1)*batch_size,]))/batch_size
        dby=sum(V-Y_train[ex*batch_size:(ex+1)*batch_size,])/batch_size
        deltah=((V-(Y_train[ex*batch_size:(ex+1)*batch_size,])).dot(Wy.T))*(H-(H*H))
        dWh=(batch.T).dot(deltah)/batch_size
        dbh=sum(deltah)/batch_size
        Wy= Wy-(eta*dWy)
        by=by-eta*dby
        Wh= Wh-(eta*dWh)
        bh=bh-eta*dbh


# In[11]:


#accuracy(Wh, bh, Wy, by,X_test,Y_test)


# In[12]:


sigma = 1e-1
N = X_train.shape[0]
d = X_train.shape[1]
Wh = np.random.randn(d,100) * sigma
Wy = np.random.randn(100,K) * sigma
bh = np.random.randn(1,100) * sigma
by = np.random.randn(1,K) * sigma
numEp = 100 # Number of epochs for gradient descent
eta = 1# Learning rate
batch_size = 100
nb_batches = int(float(N) / batch_size)
gradW = np.zeros((d,K))
gradb = np.zeros((1,K))


for epoch in range(numEp):
    for ex in range(nb_batches):
        batch = X_train[ex*batch_size:(ex+1)*batch_size,]
        H,V=forward1(batch, Wh, bh,Wy,by)
        V= softmax(V)
        dWy=(H.T).dot(V-(Y_train[ex*batch_size:(ex+1)*batch_size,]))/batch_size
        dby=sum(V-Y_train[ex*batch_size:(ex+1)*batch_size,])/batch_size

        deltah=((V-(Y_train[ex*batch_size:(ex+1)*batch_size,])).dot(Wy.T))*(H-(H*H))
        dWh=(batch.T).dot(deltah)/batch_size
        dbh=sum(deltah)/batch_size
        Wy= Wy-(eta*dWy)
        by=by-eta*dby
        Wh= Wh-(eta*dWh)
        bh=bh-eta*dbh


# In[ ]:





# In[13]:


accuracy(Wh, bh, Wy, by,X_test,Y_test)


# In[14]:


sigma = 1e-1
N = X_train.shape[0]
d = X_train.shape[1]
Wh = np.random.randn(d,100) * sigma / np.sqrt(d)
Wy = np.random.randn(100,K) * sigma / np.sqrt(d)
bh = np.random.randn(1,100) * sigma / np.sqrt(d)
by = np.random.randn(1,K) * sigma / np.sqrt(d)
numEp = 100 # Number of epochs for gradient descent
eta = 1# Learning rate
batch_size = 100
nb_batches = int(float(N) / batch_size)
gradW = np.zeros((d,K))
gradb = np.zeros((1,K))

for epoch in range(numEp):
    for ex in range(nb_batches):
        batch = X_train[ex*batch_size:(ex+1)*batch_size,]
        H,V=forward1(batch, Wh, bh,Wy,by)
        V= softmax(V)
        dWy=(H.T).dot(V-(Y_train[ex*batch_size:(ex+1)*batch_size,]))/batch_size
        dby=sum(V-Y_train[ex*batch_size:(ex+1)*batch_size,])/batch_size

        deltah=((V-(Y_train[ex*batch_size:(ex+1)*batch_size,])).dot(Wy.T))*(H-(H*H))
        dWh=(batch.T).dot(deltah)/batch_size
        dbh=sum(deltah)/batch_size
        Wy= Wy-(eta*dWy)
        by=by-eta*dby
        Wh= Wh-(eta*dWh)
        bh=bh-eta*dbh


# In[15]:


accuracy(Wh, bh, Wy, by,X_test,Y_test)

