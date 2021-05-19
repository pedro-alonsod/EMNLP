#%%
import os
#1 imports & func
import nltk
#nltk.download('all')
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
#import yawlib
#from yawlib.glosswordnet import Gloss
#from yawlib import YLConfig
#from yawlib import WordNetSQL
#ywn = WordNetSQL(YLConfig.WORDNET_30_PATH)
from nltk.corpus import stopwords #this to spacy stop words

#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import faulthandler
import gc
import nltk
import re
import random
import string
from scipy.ndimage.interpolation import shift
stop_words = set(stopwords.words("english"))
resdef = wn.synset('ocean.n.01').definition()
print(resdef)
from string import punctuation
import os
import sklearn
from sklearn import datasets
import importlib
import numpy as np
import pprint as pp
# from sklearn.preprocessing import MultiLabelBinarizer
# nltk.download('reuters')
import keras
import pandas as pd
# import modin.pandas as pd

import gensim
# import modin.pandas as md
from scipy.spatial import distance
import numpy as np
# np.set_printoptions(threshold=sys.maxsize)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC


import csv
import spacy
import seaborn as sns
from math import sqrt
from random import seed
from random import randrange
# from neupy import algorithms, utils

########MySQL stuff
import mysql.connector
from mysql.connector import Error

#Copy to ditto the mutable vectors
import copy 

#Manage larg data ~20GB
faulthandler.enable()
# import vaex
import dask.dataframe as dd
print('import completed')
print(os.getcwd())
# sys.exit()
#check time used
import time

#file reader
import codecs
import sys
import time
import csv

import text_functions as tf

from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import traceback

np.seterr('raise')
from tqdm import tqdm_notebook

#  and 
from tqdm import notebook
k = 1
# np.seterr(all='raise')
#For internet connections e. g. onedrive
import requests 
import json

#for gensim imports
import gzip
import gensim 
import logging
# import help_functions as hf
import nltk
import codecs
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import text_functions as tf

# In[169]:
dictToefl = {'BEAGLEContext':[],
            'BEAGLEOrder':[],
            'BEAGLEEnv':[],
            'BEAGLEComposite':[],
            'W2V':[]
            }

# Hyperparameters for RI

test_name = "new_toefl.txt" # file with TOEFL dataset

filename = "storageWxW_t0_1k.csv" #not used

threshold = 15000 # Frequency threshold in the corpus ??
dimension = 300 # Dimensionality for high-dimensional vectors
lemmatizer = nltk.WordNetLemmatizer()  # create an instance of lemmatizer
ones_number = 3 # number of nonzero elements in randomly generated high-dimensional vectors
window_size = 2 #number of neighboring words to consider both back and forth. In other words number of words before/after current word
zero_vector = np.zeros(dimension)
test_name = "new_toefl.txt" # file with TOEFL dataset
data_file_name = "lemmatized.text" # file with the text corpus
epochsRI = 10 


for epochsRITrain in range(21):
    for run in range(3):
        print('Reading')

        sentences = []

        file = open("lemmatized.text", "r")
        for line in file: # read the file and create list which contains all sentences found in the text
            sentences.append(line.split())
        # train word2vec on the two sentences

        number_of_tests = 0
        text_file = open(test_name, "r") #open TOEFL tasks
        for line in text_file:
            number_of_tests += 1
        text_file.close()
        # First create the normal RI
        amount_dictionary = {}
        #dimension = 300 # Dimensionality for high-dimensional vectors

        # Count how many times each word appears in the corpus
        text_file = open(data_file_name, "r")
        for line in text_file:
            if line != "\n":
                words = line.split()
                for word in words:
                    if amount_dictionary.get(word) is None:
                        amount_dictionary[word] = 1
                    else:
                        amount_dictionary[word] += 1
        text_file.close()


        # In[172]:


        # I was testing thats why the double cell basically this is the same as previous
        text = []

        # Count how many times each word appears in the corpus
        text_file = open(data_file_name, "r")
        for line in text_file:
            if line != "\n":
                text.append(line.strip())
                words = line.split()
                for word in words:
                    if amount_dictionary.get(word) is None:
                        amount_dictionary[word] = 1
                    else:
                        amount_dictionary[word] += 1
        text_file.close()


        # print(len(amount_dictionary.keys()))


        # In[173]:


        # first we need the word_space thingy!!!
        dictionary = {} #vocabulary and corresponing random high-dimensional vectors
        word_space = {} #embedings
        wordXWord = {} #for negative info paper
        dfRIMidRun = pd.DataFrame()

        # For BEAGLE ####################################################
        # env = Gaussian distribution in the standard implementation
        # A word’s context vector is updated with the sum of the environmentalvectors for the other words appearing in 
        # the same sentence. A word’s order vectoris formed by binding it with all ngram chunks in the sentence 
        # with directional circular convolution (see Jones & Mewhort, 2007 for additional detail)
        envVecs_static = {}
        orderVecs_position = {}
        contextVecs_cooccurrence = {}
        compositeVecs_CoPlusOr = {}

        #################################### 
        mu, sigma = 0, 1/300 # mean and standard deviation
        # s = np.random.normal(mu, sigma, 1000)
        N = 300.0
        SD = 0.01
        phiVec = np.random.normal(mu, sigma, 300)

        ########
        # Functions borrowed from holoword.py
        ########

        def getNGrams(wordlist, n):
            ngrams = []
            # if len
            # print(len(wordlist)-(n-1))
            for i in range(len(wordlist)-(n-1)):
                ngrams.append(wordlist[i:i+n])
            return ngrams

        def normalize(a):
            '''
            Normalize a vector to length 1.
            '''
            return a / np.sum(a**2.0)**0.5


        def cconv(a, b):
            '''
            Computes the circular convolution of the (real-valued) vectors a and b.
            '''
            return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real

        def ccorr(a, b):
            '''
            Computes the circular correlation (inverse convolution) of the real-valued
            vector a with b.
            '''
            return cconv(np.roll(a[::-1], 1), b)


        def ordConv(a, b, p1, p2):
            '''
            Performs ordered (non-commutative) circular convolution on the vectors a and
            b by first permuting them according to the index vectors p1 and p2.
            '''
            return cconv(a[p1], b[p2])


        def seqOrdConv(l , p1, p2 ):
            '''
            Given a list of vectors, iteratively convolves them into a single vector
            (i.e., "binds" them together: (((1+2)+3)+4)+5 ). Used to combine characters in ngrams.
            '''
            return reduce(lambda a,b: normalize(ordConv(a, b, p1, p2)), l)


        #modified from holoword.py to use the number vectors instead of characters
        def getOpenNGrams(word, charVecList, charPlaceholder):
            ngrams = []
            sizes = range(len(word))[2:len(word)]
            sizes.append(len(word))
            for size in sizes:
                for i in xrange(len(word)):
                    if i+size > len(word): break
                    tmp = []
                    for char in word[i:(i+size)]:
                        tmp.append(charVecList[char])
                    ngrams.append(tmp)
                    if i+size == len(word): continue
                    for b in xrange(1, size):
                        for e in xrange(1, len(word)-i-size+1):
                            tmp = []
                            for char in word[i:(i+b)]:
                                tmp.append(charVecList[char])
                            tmp.append(charPlaceholder)
                            for char in word[(i+b+e):(i+e+size)]:
                                tmp.append(charVecList[char])
                            ngrams.append(tmp)
            return ngrams
        
        ##################### END ###########################

        # for ngramSize in range(8):
        #     listOfNgrams = []
        #     if ngramSize < 3:
        #         continue
        #     else:    
        #         text_file = open(data_file_name, "r")

        #         for line in text_file: #read line in the file
        #             words = line.split() # extract words from the line
        #             listOfNgrams.append(getNGrams(words, ngramSize))
        #         text_file.close()
                # addNgrams(listOfNgrams, list(amount_dictionary.keys(), ngramSize)

        #Create a dictionary with the assigned random high-dimensional vectors
        text_file = open(data_file_name, "r")
        for line in text_file: #read line in the file
            words = line.split() # extract words from the line
            # print(getNGrams(words, 3)) #works to get word ngrams
            # sys.exit()
            for word in words:  # for each word
                if dictionary.get(word) is None: # If the word was not yed added to the vocabulary
                    if amount_dictionary[word] < threshold:
                        dictionary[word] = tf.get_random_word_vector(dimension, ones_number) # assign a  
                    else:
                        dictionary[word] = np.zeros(dimension) # frequent words are assigned with empty vectors. In a way they will not contribute to the word embedding

                if word_space.get(word) is None: # If the word was not yed added to the vocabulary
                        word_space[word] = np.zeros(dimension) # initialize to all zeros
                        wordXWord[word] = np.zeros(dimension)

        text_file.close()

        dfRIBegin = pd.DataFrame.from_dict(word_space, orient='index')
        # print('word_space first \n', dfRIBegin.head())

        for word in word_space.keys():
            envVecs_static[word] = np.random.normal(mu, sigma, 300)
            contextVecs_cooccurrence[word] = np.zeros(300)
            orderVecs_position[word] = np.zeros(300)
            compositeVecs_CoPlusOr[word] = np.zeros(300)


        print('In order make')
        lineCounter = 0
        with open(data_file_name,'r') as file:
        
            # reading each line    
            for line in file:
                words = line.split()
                # reading each word        
                for i in range(len(words)):
        # These are bigrams ##############################################################################
                # print('i', i, 'len(words)', len(words))
                    # print(words[i])
                    print('starting bigrams')
                    if (i-1) < 0 and (i+1) >= len(words):
                        # pass
                        # if only one word just add it no order nedded just the word
                        orderVecs_position[words[i]] += np.roll(envVecs_static[words[i]], -1)
                    elif (i-1) < 0:
                        # print(i, 'i', 'i+1', i+1, 'words', words)
                        orderVecs_position[words[i]] += np.roll(cconv(phiVec, np.roll(envVecs_static[words[i+1]], -1)), -1)
                    elif (i+1) >= len(words):
                        orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[words[i-1]], np.roll(phiVec, -1)), -1)
                    else:
                        # Get 2grams from words
                        biGrams = list(getNGrams(words, 2))
                        # print(biGrams)
                        for tup in biGrams:
                            if words[i] == tup[0]:
                                orderVecs_position[words[i]] += np.roll(cconv(phiVec, np.roll(envVecs_static[tup[1]], -1)), -1)
                            elif words[i] == tup[1]:
                                orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[tup[0]], np.roll(phiVec, -1)), -1)
                            else:
                                continue
    # improved????????????????????????????? ^^^^^^^^ Maybe...-->
                    # if (i-1) < 0 and (i+1) >= len(words):
                    #     # pass
                    #     # if only one word just add it no order nedded just the word
                    #     orderVecs_position[words[i]] += np.roll(envVecs_static[words[i]], -1)
                    # elif (i-2) < 0 and (i+1) >= len(words):
                    #     # pass
                    #     # we have 2 words one back and the target add them
                    #     orderVecs_position[words[i]] += np.roll(cconv(np.roll(envVecs_static[words[i]], -1), phiVec), -1)
                    # elif (i-1) < 0 and (i+2) >= len(words):
                    #     # pass
                    #     # we have only 2 words target and front word add them both
                    #     orderVecs_position[words[i]] += cconv(np.roll(phiVec, -1), envVecs_static[words[i]])
                    # elif (i-1) < 0:
                    #     # Can't go left only add right one
                    #     orderVecs_position[words[i]] += cconv(np.roll(phiVec, -1), envVecs_static[words[i+1]])
                    # # else:
                    # #     # can do left add both
                    # #     print('i', i, 'i-1', i-1,'i+1', i+1, 'len(words)', len(words))
                    # #     contextVecs_cooccurrence[words[i]] = envVecs_static[words[i]] + envVecs_static[words[i+1]] + envVecs_static[words[i-1]]
                    # elif (i+1) >= len(words):
                    #     # Can't go right only add left one
                    #     orderVecs_position[words[i]] += cconv(np.roll(envVecs_static[words[i-1]], -1), phiVec)
                    # else:
                    #     # can do right add both
                    #     orderVecs_position[words[i]] += cconv(np.roll(envVecs_static[words[i-1]], -1), phiVec) + cconv(np.roll(phiVec, -1), envVecs_static[words[i+1]]) 
                    print('ending bigrams...')
        # These are bigrams ##############################################################################
        # These are trigrams ##############################################################################
        # print('i', i, 'len(words)', len(words))
                    print('starting trigrams')
                    if (i-1) < 0 and (i+1) >= len(words):
                        # pass
                        # if only one word just add it no order nedded just the word 
                        # [word]
                        orderVecs_position[words[i]] += np.roll(envVecs_static[words[i]], -1)
                    elif (i-1) < 0 and (i+2) >= len(words):
                        # print(i, 'i', 'i+1', i+1, 'words', words)
                        #  ['who', 'be']
                            # ^
                        orderVecs_position[words[i]] += np.roll(cconv(phiVec, np.roll(envVecs_static[words[i+1]], -1)), -1)
                    elif (i-2) < 0 and (i+1) >= len(words):
                        #  ['who', 'be']
                                # ^
                        orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[words[i-1]], np.roll(phiVec, -1)), -1)
                    elif (i+1) >= len(words):
                        #  ['who', 'be', 'the']
                                        # ^        
                        orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[words[i-2]], np.roll(cconv(envVecs_static[words[i-1]], np.roll(phiVec, -1)), -1)), -1)#np.roll(cconv(envVecs_static[words[i-1]], np.roll(phiVec, -1)), -1)
                    else:
                        # Get 2grams from words
                        triGrams = list(getNGrams(words, 3))
                        # print(biGrams)
                        for tup in triGrams:
                            # print('trigrams', tup, '----', triGrams, words[i], 'words[i]')
                            # sys.exit()
                            if words[i] == tup[0]:
                                orderVecs_position[words[i]] +=  np.roll(cconv(phiVec, np.roll(cconv(envVecs_static[tup[1]], np.roll(envVecs_static[tup[2]], -1)), -1)), -1) #np.roll(cconv(phiVec, np.roll(envVecs_static[words[i+1]], -1)), -1)
                            elif words[i] == tup[1]:
                                orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[tup[0]], np.roll(cconv(phiVec, np.roll(envVecs_static[tup[2]], -1)), -1)), -1) #np.roll(cconv(envVecs_static[words[i-1]], np.roll(phiVec, -1)), -1)
                            elif words[i] == tup[2]:
                                # fix this ************************** ^^^^^^^^^^^^Maybe up no needed??? Possibly too tired to be sure :'( remember there are 1s no ngram possible so maybe not
                                orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[tup[0]], np.roll(cconv(envVecs_static[tup[2]], np.roll(phiVec, -1)), -1)), -1) #np.roll(cconv(envVecs_static[words[i-1]], np.roll(phiVec, -1)), -1)
                            else:
                                continue
                    print('ending trigrams...')
                # if (i-1) < 0 and (i+2) >= len(words):
                #     # contextVecs_cooccurrence[words[i]] += envVecs_static[words[i]]
                #     pass
                # elif (i-2) < 0 and (i+1) >= len(words):
                #         pass
                # elif (i-1) < 0 and (i+2) >= len(words):
                #         pass
                # elif (i-2) < 0:
                #     # Can't go left only add right one
                #     if (i-1) < 0:
                #         orderVecs_position[words[i]] += np.roll(cconv(phiVec, np.roll(cconv(np.roll(envVecs_static[words[i+1]], -1), envVecs_static[words[i+2]]), -1)), -1)
                #     else:
                #         # orderVecs_position[words[i]] += cconv(phiVec, cconv(envVecs_static[words[i+1]], envVecs_static[words[i+2]]))
                #         # print(i, 'i', i-1, 'i+1', i+1, len(words), 'len(words)')
                #         orderVecs_position[words[i]] += np.roll(cconv(np.roll(cconv(np.roll(envVecs_static[words[i-1]], -1), phiVec), -1), envVecs_static[words[i+1]]))       
                # # else:
                # #     # can do left add both
                # #     print('i', i, 'i-1', i-1,'i+1', i+1, 'len(words)', len(words))
                # #     contextVecs_cooccurrence[words[i]] = envVecs_static[words[i]] + envVecs_static[words[i+1]] + envVecs_static[words[i-1]]
                # elif (i+2) >= len(words):
                #     if len(words) == 0:
                #         continue
                #     else:
                #         # Can't go right only add left one
                #         # print(i, 'i', i-1, 'i+1', i+1, len(words), 'len(words)')
                #         if (i+1) >= len(words):
                #             orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[words[i-1]], np.roll(cconv(np.roll(envVecs_static[words[i-2]], -1), phiVec), -1)), -1)
                #         else:
                #             orderVecs_position[words[i]] += np.roll(cconv(np.roll(cconv(np.roll(envVecs_static[words[i-1]], -1), envVecs_static[words[i-2]]), -1), phiVec), -1)
                #             orderVecs_position[words[i]] += np.roll(cconv(np.roll(cconv(np.roll(envVecs_static[words[i-1]], -1), phiVec), -1), envVecs_static[words[i+1]]), -1)
                # else:
                #     # can do right add both
                #     orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[words[i-2]], np.roll(cconv(np.roll(envVecs_static[words[i-1]], -1), phiVec), -1)), -1) + np.roll(cconv(envVecs_static[words[i-1]], np.roll(cconv(np.roll(phiVec, -1), envVecs_static[words[i+1]]), -1)), -1) + np.roll(cconv(phiVec, np.roll(cconv(np.roll(envVecs_static[words[i+1]], -1), envVecs_static[words[i+2]]), -1)), -1)
        # These are trigrams ##############################################################################
    # These are 4grams ##########################################################
                    print('starting quadGrams...')
                    # if (i-1) < 0 and (i+1) >= len(words):
                    if len(words) == 1:
                        # pass
                        # if only have one word just add it no order nedded just the word
                        orderVecs_position[words[i]] += np.roll(envVecs_static[words[i]], -1)
                    elif (i-1) < 0 and (i+2) >= len(words):
                    # because we have something like this 
                    #  ['who', 'be']
                        # ^ we add 2 focus on the second
                        orderVecs_position[words[i]] += np.roll(cconv(phiVec, np.roll(envVecs_static[words[i+1]], -1)), -1)
                    elif (i-2) < 0 and (i+1) >= len(words):
                        #  ['who', 'be']
                                # ^ again 2 but order is first
                        orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[words[i-1]], np.roll(phiVec, -1)), -1)
                    elif (i-1) < 0 and (i+3) >= len(words):
                    # if we have something like this and we are at the start
                    # ['who', 'be', 'the]
                    #    ^  perm( cconv(_, perm(cconv(b, perm(c))))  <-- abc
                        orderVecs_position[words[i]] += np.roll(cconv(phiVec, np.roll(cconv(envVecs_static[word[i+1]], np.roll(envVecs_static[words[i+2]], -1)), -1)), -1)
                    
                    elif (i-2) < 0 and (i+2) >= len(words):
                    # if we have something like this and we are at the start
                    # ['who', 'be', 'the]
                    #          ^  perm( cconv(a, perm(cconv(_, perm(c))))  <-- abc
                        orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[word[i-1]], np.roll(cconv(phiVec, np.roll(envVecs_static[words[i+1]], -1)), -1)), -1)

                    elif (i-3) < 0 and (i+1) >= len(words):
                    # if we have something like this and we are at the start
                    # ['who', 'be', 'the]
                    #                 ^  perm( cconv(a, perm(cconv(b, perm(_))))  <-- abc
                        orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[word[i-2]], np.roll(cconv(envVecs_static[words[i-1]], np.roll(phiVec, -1)), -1)), -1)

                    elif (i+1) >= len(words): #No because we have 4so we need to check first 3 ^^
                        #  ['who', 'be', 'the', 'man']
                                                # ^   # perm( cconv(a, perm(cconv(b, perm(_))))  <-- abc      
                        orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[words[i-3]], np.roll(cconv(envVecs_static[words[i-2]], np.roll(cconv(envVecs_static[words[i-1]], np.roll(phiVec, -1)), -1)), -1)), -1)
                    # elif len(words) == 2: # not done need to check more 
                        # print(i, 'i', 'i+1', i+1, 'words', words)
                        #  ['who', 'be', 'the']
                        # orderVecs_position[words[i]] += np.roll(cconv(phiVec, np.roll(cconv(envVecs_static[words[i+1]], np.roll(cconv(envVecs_static[words[i+2]], np.roll(envVecs[words[i+3]], -1)), -1)), -1)), -1)
                    # elif (i+1) >= len(words):
                    #     #  ['who', 'be', 'the', 'man]
                    #     #                         ^ 
                    #     orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[words[i-2]], np.roll(cconv(envVecs_static[words[i-1]], np.roll(phiVec, -1)), -1)), -1)#np.roll(cconv(envVecs_static[words[i-1]], np.roll(phiVec, -1)), -1)
                    else:
                        # Get 2grams from words
                        quadGrams = list(getNGrams(words, 4))
                        #FIX!!!! ^^^^^
                        # print(quadGrams)
                        # sys.exit()
                        # print(biGrams)
                        for tup in quadGrams:
                            # print('trigrams', tup, '----', triGrams, words[i], 'words[i]')
                            # sys.exit()
                            if words[i] == tup[0]:
                                orderVecs_position[words[i]] += np.roll(cconv(phiVec, np.roll(cconv(envVecs_static[tup[1]], np.roll(cconv(envVecs_static[tup[2]], np.roll(envVecs_static[tup[3]], -1)), -1)), -1)), -1)
                                # orderVecs_position[words[i]] +=  np.roll(cconv(phiVec, np.roll(cconv(envVecs_static[tup[1]], np.roll(envVecs_static[tup[2]], -1)), -1)), -1) #np.roll(cconv(phiVec, np.roll(envVecs_static[words[i+1]], -1)), -1)
                            elif words[i] == tup[1]:
                                orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[tup[0]], np.roll(cconv(phiVec, np.roll(cconv(envVecs_static[tup[2]], np.roll(envVecs_static[tup[3]], -1)), -1)), -1)), -1)
                                # orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[tup[0]], np.roll(cconv(phiVec, np.roll(envVecs_static[tup[2]], -1)), -1)), -1) #np.roll(cconv(envVecs_static[words[i-1]], np.roll(phiVec, -1)), -1)
                            elif words[i] == tup[2]:
                                # fix this ************************** ^^^^^^^^^^^^Maybe up no needed??? Possibly too tired to be sure :'( remember there are 1s no ngram possible so maybe not
                                orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[tup[0]], np.roll(cconv(envVecs_static[tup[1]], np.roll(cconv(phiVec, np.roll(envVecs_static[tup[3]], -1)), -1)), -1)), -1)
                                # orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[tup[0]], np.roll(cconv(envVecs_static[tup[2]], np.roll(phiVec, -1)), -1)), -1) #np.roll(cconv(envVecs_static[words[i-1]], np.roll(phiVec, -1)), -1)
                            elif words[i] == tup[3]:
                                orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[tup[0]], np.roll(cconv(envVecs_static[tup[1]], np.roll(cconv(envVecs_static[tup[2]], np.roll(phiVec, -1)), -1)), -1)), -1)
                            else:
                                continue
                    print('ending quadGrams...')
    # These are 4grams up ^^^^^^^^^^^^ #########################################
    # These are femteGrams up  #########################################
                    print('starting femteGrams...')
                    # if (i-1) < 0 and (i+1) >= len(words):
                    if len(words) == 1:
                        # pass
                        # if only have one word just add it no order nedded just the word
                        orderVecs_position[words[i]] += np.roll(envVecs_static[words[i]], -1)
                    elif (i-1) < 0 and (i+2) >= len(words):
                    # because we have something like this 
                    #  ['who', 'be']
                        # ^ we add 2 focus on the second
                        orderVecs_position[words[i]] += np.roll(cconv(phiVec, np.roll(envVecs_static[words[i+1]], -1)), -1)
                    elif (i-2) < 0 and (i+1) >= len(words):
                        #  ['who', 'be']
                                # ^ again 2 but order is first
                        orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[words[i-1]], np.roll(phiVec, -1)), -1)
                    elif (i-1) < 0 and (i+3) >= len(words):
                    # if we have something like this and we are at the start
                    # ['who', 'be', 'the]
                    #    ^  perm( cconv(_, perm(cconv(b, perm(c))))  <-- abc
                        orderVecs_position[words[i]] += np.roll(cconv(phiVec, np.roll(cconv(envVecs_static[word[i+1]], np.roll(envVecs_static[words[i+2]], -1)), -1)), -1)
                    
                    elif (i-2) < 0 and (i+2) >= len(words):
                    # if we have something like this and we are at the start
                    # ['who', 'be', 'the]
                    #          ^  perm( cconv(a, perm(cconv(_, perm(c))))  <-- abc
                        orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[word[i-1]], np.roll(cconv(phiVec, np.roll(envVecs_static[words[i+1]], -1)), -1)), -1)

                    elif (i-3) < 0 and (i+1) >= len(words):
                    # if we have something like this and we are at the start
                    # ['who', 'be', 'the]
                    #                 ^  perm( cconv(a, perm(cconv(b, perm(_))))  <-- abc
                        orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[word[i-2]], np.roll(cconv(envVecs_static[words[i-1]], np.roll(phiVec, -1)), -1)), -1)
    # Fix starting here for femteGrams
                    elif (i-1) < 0 and (i+4) >= len(words):
                    # if we have something like this and we are at the start
                    # ['who', 'be', 'the', 'first']
                    #    ^  perm( cconv(_, perm(cconv(b, perm(cconv(c, perm(d)))))  <-- abcd
                        orderVecs_position[words[i]] += np.roll(cconv(phiVec, np.roll(cconv(envVecs_static[word[i+1]], np.roll(cconv(envVecs_static[words[i+2]], np.roll(envVecs_static[words[i+3]], -1)), -1)), -1)), -1)
                    elif (i-2) < 0 and (i+3) >= len(words):
                        # FIX ME!!! I'm incomplete!!!!!!
                        # ['who', 'be', 'the', 'first']
                            #    ^  perm( cconv(a, perm(cconv(_, perm(cconv(c, perm(d)))))  <-- abcd
                        orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[word[i-1]], np.roll(cconv(phiVec, np.roll(cconv(envVecs_static[words[i+1]], np.roll(envVecs_static[words[i+2]], -1)), -1)), -1)), -1)
                    elif (i-3) < 0 and (i+2) >= len(words):
                        # FIX ME!!! I'm incomplete!!!!!!
                        # ['who', 'be', 'the', 'first']
                                    #    ^  perm( cconv(a, perm(cconv(b, perm(cconv(_, perm(d)))))  <-- abcd
                        orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[word[i-2]], np.roll(cconv(envVecs_static[words[i-1]], np.roll(cconv(phiVec, np.roll(envVecs_static[words[i+1]], -1)), -1)), -1)), -1)
                    elif (i-4) < 0 and (i+1) >= len(words):
                        # FIX ME!!! I'm incomplete!!!!!!
                        # ['who', 'be', 'the', 'first']
                                            #    ^  perm( cconv(a, perm(cconv(b, perm(cconv(c, perm(_)))))  <-- abcd
                        orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[word[i-3]], np.roll(cconv(envVecs_static[words[i-2]], np.roll(cconv(envVecs_static[words[i-1]], np.roll(phiVec, -1)), -1)), -1)), -1)
                    
                    # elif (i-2) < 0 and (i+2) >= len(words):
                    # # if we have something like this and we are at the start
                    # # ['who', 'be', 'the]
                    # #          ^  perm( cconv(a, perm(cconv(_, perm(c))))  <-- abc
                    #     orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[word[i-1]], np.roll(cconv(phiVec, np.roll(envVecs_static[words[i+1]], -1)), -1)), -1)

                    # elif (i-3) < 0 and (i+1) >= len(words):
                    # # if we have something like this and we are at the start
                    # # ['who', 'be', 'the]
                    # #                 ^  perm( cconv(a, perm(cconv(b, perm(_))))  <-- abc
                    #     orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[word[i-2]], np.roll(cconv(envVecs_static[words[i-1]], np.roll(phiVec, -1)), -1)), -1)


                    elif (i+1) >= len(words): #No because we have 4so we need to check first 3 ^^
                        #  ['who', 'be', 'the', 'man', 'first']
                    # #                                  ^  perm( cconv(a, perm(cconv(b, perm(_))))  <-- abc
                                                        # ^   # perm( cconv(a, perm(cconv(b, perm(_))))  <-- abc      
                        orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[words[i-3]], np.roll(cconv(envVecs_static[words[i-2]], np.roll(cconv(envVecs_static[words[i-1]], np.roll(phiVec, -1)), -1)), -1)), -1)
                    # elif len(words) == 2: # not done need to check more 
                        # print(i, 'i', 'i+1', i+1, 'words', words)
                        #  ['who', 'be', 'the']
                        # orderVecs_position[words[i]] += np.roll(cconv(phiVec, np.roll(cconv(envVecs_static[words[i+1]], np.roll(cconv(envVecs_static[words[i+2]], np.roll(envVecs[words[i+3]], -1)), -1)), -1)), -1)
                    # elif (i+1) >= len(words):
                    #     #  ['who', 'be', 'the', 'man]
                    #     #                         ^ 
                    #     orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[words[i-2]], np.roll(cconv(envVecs_static[words[i-1]], np.roll(phiVec, -1)), -1)), -1)#np.roll(cconv(envVecs_static[words[i-1]], np.roll(phiVec, -1)), -1)
                    else:
                        # Get 2grams from words
                        femteGrams = list(getNGrams(words, 5))
                        #FIX!!!! ^^^^^
                        # print(quadGrams)
                        # sys.exit()
                        # print(biGrams)
                        for tup in femteGrams:
                            # print('trigrams', tup, '----', triGrams, words[i], 'words[i]')
                            # sys.exit()
                            # FIX all down things they are for quads need to be for FEMTE!!!
                            if words[i] == tup[0]:
                                orderVecs_position[words[i]] += np.roll(cconv(phiVec, np.roll(cconv(envVecs_static[tup[1]], np.roll(cconv(envVecs_static[tup[2]], np.roll(cconv(envVecs_static[tup[3]], np.roll(envVecs_static[tup[4]], -1)), -1)), -1)), -1)), -1)
                                # orderVecs_position[words[i]] +=  np.roll(cconv(phiVec, np.roll(cconv(envVecs_static[tup[1]], np.roll(envVecs_static[tup[2]], -1)), -1)), -1) #np.roll(cconv(phiVec, np.roll(envVecs_static[words[i+1]], -1)), -1)
                            elif words[i] == tup[1]:
                                orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[tup[0]], np.roll(cconv(phiVec, np.roll(cconv(envVecs_static[tup[2]], np.roll(cconv(envVecs_static[tup[3]], np.roll(envVecs_static[tup[4]], -1)), -1)), -1)), -1)), -1)
                                # orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[tup[0]], np.roll(cconv(phiVec, np.roll(cconv(envVecs_static[tup[2]], np.roll(envVecs_static[tup[3]], -1)), -1)), -1)), -1)
                                # orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[tup[0]], np.roll(cconv(phiVec, np.roll(envVecs_static[tup[2]], -1)), -1)), -1) #np.roll(cconv(envVecs_static[words[i-1]], np.roll(phiVec, -1)), -1)
                            elif words[i] == tup[2]:
                                # fix this ************************** ^^^^^^^^^^^^Maybe up no needed??? Possibly too tired to be sure :'( remember there are 1s no ngram possible so maybe not
                                orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[tup[0]], np.roll(cconv(envVecs_static[tup[1]], np.roll(cconv(phiVec, np.roll(cconv(envVecs_static[tup[3]], np.roll(envVecs_static[tup[4]], -1)), -1)), -1)), -1)), -1)
                                # orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[tup[0]], np.roll(cconv(envVecs_static[tup[1]], np.roll(cconv(phiVec, np.roll(envVecs_static[tup[3]], -1)), -1)), -1)), -1)
                                # orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[tup[0]], np.roll(cconv(envVecs_static[tup[2]], np.roll(phiVec, -1)), -1)), -1) #np.roll(cconv(envVecs_static[words[i-1]], np.roll(phiVec, -1)), -1)
                            elif words[i] == tup[3]:
                                orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[tup[0]], np.roll(cconv(envVecs_static[tup[1]], np.roll(cconv(envVecs_static[tup[2]], np.roll(cconv(phiVec, np.roll(envVecs_static[tup[4]], -1)), -1)), -1)), -1)), -1)
                                # orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[tup[0]], np.roll(cconv(envVecs_static[tup[1]], np.roll(cconv(envVecs_static[tup[2]], np.roll(phiVec, -1)), -1)), -1)), -1)
                            elif words[i] == tup[4]:
                                orderVecs_position[words[i]] += np.roll(cconv(envVecs_static[tup[0]], np.roll(cconv(envVecs_static[tup[1]], np.roll(cconv(envVecs_static[tup[2]], np.roll(cconv(envVecs_static[tup[3]], np.roll(phiVec, -1)), -1)), -1)), -1)), -1)
                            else:
                                continue
                    print('ending femteGrams...')
                lineCounter += 1
                print('line', lineCounter, 'done')

    # These are femteGrams up ^^^^^^^^^^^^ #########################################

        for word in orderVecs_position.keys():
            # print(word)
            temp=np.array(orderVecs_position[word], dtype=float)
            temp += np.random.uniform(-0.1, 0.1, size=(dimension))
            # print('temp withount 0s', np.count_nonzero(temp==0))
            # sys.exit()
            temp=temp/np.linalg.norm(temp) #calculate Euclidean norm of temp
            orderVecs_position[word]=temp
            # print('**************************Normalising **********************************')
            # maybe this time i dnt need to normalze it here the eval is doing thats
        # print(word_space, 'embeddings to test')

        dfOrder = pd.DataFrame.from_dict(orderVecs_position, orient='index')
        dfOrder.to_csv(f'BEAGLE/orderVecs{epochsRITrain}Run{run}.gz', sep=' ', header=False, compression='gzip')

        ############################## word2vec creation and training here#############################

        epochsRI = 10
        w2vObjectRI = gensim.models.Word2Vec(min_count=1, sample=threshold, sg=1,size=dimension, negative=15, iter=epochsRI, window=3, workers=8) # create only the shell

        print('Starting vocab build')
        # t = time()
        w2vObjectRI.build_vocab(sentences, progress_per=10000) #here is the vocab being built as told in google groups gensim

        # print(w2vObject.wv['the'], 'before train')


        # In[7]:


        setOfGensimVocab = set(w2vObjectRI.wv.vocab)
        setOfWordNetVocab = set(orderVecs_position.keys()) #context not WN!!!!!!!!!!!!!!!!!!!

        print('len of sets w2v, wn', len(setOfGensimVocab), len(setOfWordNetVocab))

        setIntersection = setOfGensimVocab.intersection(setOfWordNetVocab)
        print('len of intersection', len(setIntersection))#, setIntersection)


        # In[ ]:


        for elem in setIntersection:
        #     print(elem)
        #     print('w2v:', w2vObject.wv[elem], type(w2vObject.wv[elem]))
        #     print('w2v:', embeddings_index[elem], type(embeddings_index[elem]))
        #     break
            if len(orderVecs_position[elem]) != 300:
                print('here', elem) #cast it to the fire
            w2vObjectRI.wv[elem] = np.asarray(orderVecs_position[elem], dtype=np.float32)
            print('moved', elem)
        print('Done!!!')


        # In[79]:


        # print(w2vObject.wv['the'], 'after before')
        w2vObjectRI.train(sentences, total_examples=w2vObjectRI.corpus_count, epochs=epochsRITrain)#w2vObject.iter)
        # print(w2vObject.wv['the'], 'after train')
        # sys.exit()
        w2vObjectRI.wv.save_word2vec_format(f'./BEAGLE/GensimOneBAEGLEOrder{epochsRITrain}Run{run}.pkl.gz')#, binary=False) #encoding='utf-8' )
        # print('saved')
        
        ############################## for this dict is over go to next this is order #########################

        # Test BEAGLE order
        wordXWordDict = dfOrder
        #Testing of the embeddings on TOEFL RI without negative info
        a = 0.0 # accuracy of the encodings    
        i = 0
        text_file = open(test_name, 'r')
        right_answers = 0.0 # variable for correct answers
        number_skipped_tests = 0.0 # some tests could be skipped if there are no corresponding words in the vocabulary extracted from the training corpus
        while i < number_of_tests:
                line = text_file.readline() #read line in the file
                words = line.split()  # extract words from the line
                words = [lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word, 'v'), 'n'), 'a') for word in
                        words]  # lemmatize words in the current test
                try:

                    if not(amount_dictionary.get(words[0]) is None): # check if there word in the corpus for the query word
                        k = 1
                        while k < 5:
                            # if amount_dictionary.get(words[k]) is None:
                            #     word_space[words[k]] = np.random.randn(dimension)
                            if np.array_equal(wordXWordDict.loc[words[k]], zero_vector): # if no representation was learnt assign a random vector
                                wordXWordDict.loc[words[k]] = np.random.randn(dimension)
                            k += 1
        #                 print(words[0], wordXWordDict[words[0]])
        #                 print(words[1], wordXWordDict[words[1]])
        #                 print(words[2], wordXWordDict[words[2]])
        #                 print(words[3], wordXWordDict[words[3]])
        #                 print(words[4], wordXWordDict[words[4]])
                        right_answers += tf.get_answer_mod([wordXWordDict.loc[words[0]],wordXWordDict.loc[words[1]],wordXWordDict.loc[words[2]],
                                    wordXWordDict.loc[words[3]],wordXWordDict.loc[words[4]]]) #check if word is predicted right
                except KeyError as k: # if there is no representation for the query vector than skip
                    number_skipped_tests += 1
                    print('error here', k, 'for this', k.args)
                    print("skipped test: " + str(i) + "; Line: " + str(words))
                except IndexError:
                    # print(i)
                    # print(line)
                    # print(words)
                    break
                except FloatingPointError as e:
                    print(e)
        #             traceback.print_exc()
        #             break
                    number_skipped_tests += 1
                    print("skipped test due to floating point: " + str(i) + "; Line: " + str(words))
                    
                i += 1
        text_file.close()
        a += 100 * right_answers / (number_of_tests - number_skipped_tests)
        print(str(dimension) + " Percentage of correct answers with BEAGLE order: " + str(100 * right_answers / (number_of_tests - number_skipped_tests)) + f"% skipped tests {number_skipped_tests}")
        dictToefl['BEAGLEOrder'].append((100 * right_answers / (number_of_tests - number_skipped_tests)))
