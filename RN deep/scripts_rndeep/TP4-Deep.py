#!/usr/bin/env python
# coding: utf-8

# On va commencer par parser le ficher d’entrée pour récupérer le texte et effectuer quelques pré-traitements simples

# In[24]:


bStart = False
fin = open("fleurs_mal.txt", 'r' , encoding = 'utf8')
lines = fin.readlines()
lines2 = []
text = []

for line in lines:
    line = line.strip().lower() # Remove blanks and capitals
    if("Charles Baudelaire avait un ami".lower() in line and bStart==False):
        print("START")
        bStart = True
    if("End of the Project Gutenberg EBook of Les Fleurs du Mal, by Charles Baudelaire".lower() in line):
        print("END")
        break
    if(bStart==False or len(line) == 0):
        continue
    lines2.append(line)

fin.close()
text = " ".join(lines2)
chars = sorted(set([c for c in text]))
nb_chars = len(chars)
print("Le text contient %d chars dont %d uniques:"%(len(text),nb_chars))


# Question :
# 
# Comment s’interprète la variable chars ? Que représente nb_chars ?
# 
# Reponse: chars liste des caractères. nb_chars: nombre des caractères uniques dans le text.
# 

# Dans la suite, on va considérer chaque caractère du texte d’entrée par un encodage one-hot sur le dictionnaire de symboles. On va appliquer un réseau de neurones récurrent qui va traiter une séquence de SEQLEN caractères, et dont l’objectif va être de prédire le caractère suivant en fonction de la séquence courante. On se situe donc dans le cas d’un problème d’apprentissage auto-supervisé, i.e. qui ne contient pas de label mais dont on va construire artificiellement une supervision.
# 
# Les données d’entraînement consisteront donc en un ensemble de séquences d’entraînement de taille SEQLEN, avec une étiquette cible correspondant au prochain caractère à prédire. 

# In[25]:


# mapping char -> index in dictionary: used for encoding (here)
char2index = dict((c, i) for i, c in enumerate(chars))
# mapping char -> index in dictionary: used for decoding, i.e. generation - part c)
index2char = dict((i, c) for i, c in enumerate(chars)) # mapping index -> char in dictionary


print(char2index)


# In[26]:


SEQLEN = 10 # Length of the sequence to predict next char
STEP = 1 # stride between two subsequent sequences
input_chars = []
label_chars = []
for i in range(0, len(text) - SEQLEN, STEP):
        input_chars.append(text[i:i + SEQLEN])
        label_chars.append(text[i + SEQLEN])
        
nbex = len(input_chars)


# In[27]:


import numpy as np
X = np.zeros((len(input_chars), SEQLEN, nb_chars), dtype=np.bool)
y = np.zeros((len(input_chars), nb_chars), dtype=np.bool)

#print(X)
#n_patterns = len(X)
#print ("Total Patterns: ", n_patterns)

for i, input_char in enumerate(input_chars):
    for j, ch in enumerate(input_char):
        X[i,j,char2index[ch]]= True
        if(i != nbex -1):
            y[i, char2index[input_chars[i+1][SEQLEN - 1]]] = True
print(X)
n_patterns = len(X)
print ("Total Patterns: ", n_patterns)

      


# In[28]:


import _pickle as pickle

ratio_train = 0.8
nb_train = int(round(len(input_chars)*ratio_train))
print("nb tot=",len(input_chars) , "nb_train=",nb_train)
X_train = X[0:nb_train,:,:]
y_train = y[0:nb_train,:]

X_test = X[nb_train:,:,:]
y_test = y[nb_train:,:]
print("X train.shape=",X_train.shape)
print("y train.shape=",y_train.shape)

print("X test.shape=",X_test.shape)
print("y test.shape=",y_test.shape)

outfile = "Baudelaire_len_"+str(SEQLEN)+".p"

with open(outfile, "wb" ) as pickle_f:
    pickle.dump( [index2char, X_train, y_train, X_test, y_test], pickle_f)


# On va maintenant entrainer le modèle sur un reseau de neurones 

# In[29]:


#Charger les données précedentes
SEQLEN = 10
outfile = "Baudelaire_len_"+str(SEQLEN)+".p"
[index2char, X_train, y_train, X_test, y_test] = pickle.load( open( outfile, "rb" ) )
index2char


# In[30]:


#Creer un modèle séquentiel
from keras.layers.recurrent import SimpleRNN
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, RMSprop

model = Sequential()


# In[31]:


#Ajouter une couche récurrente simple:ici on travaillera avec un RNN simple sans LSTM ou autre
HSIZE = 128
model.add(SimpleRNN(HSIZE, return_sequences=False, input_shape=(SEQLEN, nb_chars),unroll=True))


# Question :
# Expliquer à quoi correspond return_sequences=False. N.B. : unroll=True permettra simplement d’accélérer les calculs.
# 
# Réponse:
# return_sequences: Boolean. Whether to return the last output. in the output sequence, or the full sequence

# On ajoutera enfin une couche complètement connectée suivie d’une fonction softmax for effectuer la classification du caractère suivant la séquence.

# In[32]:


model.add(Dense(nb_chars))
model.add(Activation("softmax"))


# Pour optimiser des réseaux récurrents, on utilise préférentiellement des méthodes adaptatives comme RMSprop [TH12]. On pourra donc compiler le modèle et utiliser la méthode summary() pour visualiser le nombre de paramètres du réseaux

# In[34]:


BATCH_SIZE = 128
NUM_EPOCHS = 50
learning_rate = 0.001
optim = RMSprop(lr=learning_rate)
model.compile(loss="categorical_crossentropy", optimizer=optim,metrics=['accuracy'])
model.summary()


# In[35]:


model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)
scores_train = model.evaluate(X_train, y_train, verbose=1)
scores_test = model.evaluate(X_test, y_test, verbose=1)
print("PERFS TRAIN: %s: %.2f%%" % (model.metrics_names[1], scores_train[1]*100))
print("PERFS TEST: %s: %.2f%%" % (model.metrics_names[1], scores_test[1]*100))


# In[36]:


from keras.models import model_from_yaml
def saveModel(model, savename):
  # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(savename+".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
        print ("Yaml Model ",savename,".yaml saved to disk")
      # serialize weights to HDF5
    model.save_weights(savename+".h5")
    print("Weights ",savename,".h5 saved to disk")
    
saveModel(model,'MYMODEL')


# On va maintenant se servir du modèle précédemment entraîné pour générer du texte qui va « imiter » le style du corpus de poésie sur lequel il a été appris. On mettre en place un script exo2.py pour cette partie.
# 
# On va commencer par charger les données :

# In[37]:


SEQLEN = 10
outfile = "Baudelaire_len_"+str(SEQLEN)+".p"
[index2char, X_train, y_train, X_test, y_test] = pickle.load( open( outfile, "rb" ) )


# In[38]:


from keras.models import model_from_yaml
def loadModel(savename):
    with open(savename+".yaml", "r") as yaml_file:
        model = model_from_yaml(yaml_file.read())
    print("Yaml Model ",savename,".yaml loaded ")
    model.load_weights(savename+".h5")
    print("Weights ",savename,".h5 loaded ")
    return model
loadModel('MYMODEL')


# On va maintenant sélectionner une chaîne de caractère initiale pour notre réseau, afin de prédire le caractère suivant :

# In[48]:


seed =15499
char_init = ""
for i in range(SEQLEN):
    char = index2char[np.argmax(X_train[seed,i,:])]
    char_init += char

print("CHAR INIT: "+char_init)


# On va maintenant convertir la séquence de départ au format one-hot pour appliquer le modèle de prédiction.

# In[49]:


test = np.zeros((1, SEQLEN, nb_chars), dtype=np.bool)
test[0,:,:] = X_train[seed,:,:]


# Au lieu de prédire directement la sortie de probabilité maximale, on va échantillonner une sortie tirée selon la distribution de probabilités du soft-max. Pour commencer on va utiliser un paramètre de température pour rendre la distribution plus ou moins piquée.
# On pourra utiliser la fonction suivante pour effectuer l’échantillonage après transformation de distribution:

# In[50]:


def sampling(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    predsN = pow(preds,1.0/temperature)
    predsN /= np.sum(predsN)
    probas = np.random.multinomial(1, predsN, 1)
    return np.argmax(probas)


# On va maintenant mettre en place la génération de texte à partir d’une séquence de SEQLEN caractère initiaux.

# In[59]:


nbgen = 400 # number of characters to generate (1,nb_chars)
gen_char = char_init
temperature  = 0.3

for i in range(nbgen):
    preds = model.predict(test)[0]  # shape (1,nb_chars)
    next_ind = sampling(preds,temperature)
    next_char = index2char[next_ind]
    gen_char += next_char
    for i in range(SEQLEN-1):
        test[0,i,:] = test[0,i+1,:]
    test[0,SEQLEN-1,:] = 0
    test[0,SEQLEN-1,next_ind] = 1

print("Generated text: "+gen_char)


# Plus le nb epochs est grand plus le modele a une meilleure performance.
# Pour des valeurs de température grande, moins l'apprentissage est bon.
# 

# ### Exercice 2:
# 

# In[20]:


import pandas as pd
filename = 'flickr_8k_train_dataset.txt'
df = pd.read_csv(filename, delimiter='\t')
nb_samples = df.shape[0]
iter = df.iterrows()
allwords = []
for i in range(nb_samples):
    x = iter.__next__()
    cap_words = x[1][1].split() # split caption into words
    cap_wordsl = [w.lower() for w in cap_words] # remove capital letters
    allwords.extend(cap_wordsl)

unique = list(set(allwords)) # List of different words in captions
print(len(unique))


# In[21]:


GLOVE_MODEL = "glove.6B.100d.txt"
fglove = open(GLOVE_MODEL, "r")


# In[22]:


import numpy as np
cpt=0
listwords = []
listembeddings = []
for line in fglove:
    row = line.strip().split()
    word = row[0] #COMPLETE WITH YOUR CODE
    if(word in unique or word=='unk'): #COMPLETE WITH YOUR CODE - use a numpy array with dtype="float32"
        listwords.append(word)
        embedding = np.asarray(row[1:], "float32")
        listembeddings.append(embedding)

        cpt +=1
        print("word: "+word+" embedded "+str(cpt))

fglove.close()
nbwords = len(listembeddings)
tembedding = len(listembeddings[0])
print("Number of words="+str(len(listembeddings))+" Embedding size="+str(tembedding))


# In[23]:


embeddings = np.zeros((len(listembeddings)+2,tembedding+2))

for i in range(nbwords):
    embeddings[i,0:tembedding] = listembeddings[i]
#print(embeddings.shape)
listwords.append('<start>')
embeddings[7001,100] = 1
listwords.append('<end>')
embeddings[7002,101] = 1


# In[24]:


#Sauvegarder liste des mots et vecteurs associés:
import _pickle as pickle

outfile = 'Caption_Embeddings.p'
with open(outfile, "wb" ) as pickle_f:
    pickle.dump( [listwords, embeddings], pickle_f)


# In[25]:


#Normaliser les vecteurs pour avoir une norme euclidienne unité:

import numpy as np
import _pickle as pickle

outfile = 'Caption_Embeddings.p'
[listwords, embeddings] = pickle.load( open( outfile, "rb" ) )
print("embeddings: "+str(embeddings.shape))

for i in range(embeddings.shape[0]):
    embeddings[i,:] /= np.linalg.norm(embeddings[i,:])


# Question: Pourquoi normaliser les vecteurs?
# 
# Reponse: Normaliser équivaut à perdre la notion de longueur. Autrement dit, une fois que nous normalison les vecteurs de mots, on oublie la longueur (norme, module) qu'ils avaient juste après la phase de formation. De ce fait, un mot qui est utilisé de manière cohérente dans un contexte similaire ne sera pas représenté par un vecteur plus long qu'un mot de même fréquence utilisé dans différents contextes.

# ### Clustering dans l'espace des embeddings avec le Kmeans: 

# In[26]:


from sklearn.cluster import KMeans
kmeans =   KMeans(n_clusters=10, max_iter=1000,init='random').fit(embeddings)# COMPLETE WITH YOUR CODE - apply fit() method on embeddings
clustersID  = kmeans.labels_
clusters = kmeans.cluster_centers_


# In[44]:


from sklearn.manifold import TSNE
import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

indclusters = np.ndarray(shape=(10,7003), dtype=int)

for i in range(10):
    norm = np.linalg.norm((clusters[i] - embeddings),axis=1)
    inorms = np.argsort(norm)
    indclusters[i][:] = inorms[:]

    print("Cluster "+str(i)+" ="+listwords[indclusters[i][0]])
    for j in range(1,21):
        print(" mot: "+listwords[indclusters[i][j]])


# In[45]:


#Visualisation de la répartition des points dans l’espace d’embedding, avec la méthode t-SNE:
tsne = TSNE(n_components=2, perplexity=30, verbose=2, init='pca', early_exaggeration=24)
points2D = tsne.fit_transform(embeddings)


# In[48]:


pointsclusters= np.ndarray(shape=(10,2), dtype=int)
for i in range(10):
     pointsclusters[i,:] = points2D[int(indclusters[i][0])]

cmap =cm.tab10
plt.figure(figsize=(3.841, 7.195), dpi=100)
plt.set_cmap(cmap)
plt.subplots_adjust(hspace=0.4 )
plt.scatter(points2D[:,0], points2D[:,1], c=clustersID,  s=3,edgecolors='none', cmap=cmap, alpha=1.0)
plt.scatter(pointsclusters[:,0], pointsclusters[:,1], c=range(10),marker = '+', s=1000, edgecolors='none', cmap=cmap, alpha=1.0)

plt.colorbar(ticks=range(10))
plt.show()


# In[ ]:




