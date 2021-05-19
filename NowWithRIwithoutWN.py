#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This file is for the following
# Run 5 times each epoch plot avg
# Run RI same things
# Try to do 3rd option superposition with RI check norm sum them together and normalizing again

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


# get_ipython().run_line_magic('matplotlib', 'inline')
# plt.rcParams['figure.dpi'] = 200


# In[2]:


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 
lemmatizer = nltk.WordNetLemmatizer()
epochsNum = 10

def createRI():
    #@author: The first version of this code is the courtesy of Vadim Selyanik

    threshold = 15000 # Frequency threshold in the corpus ??
    dimension = 300 # Dimensionality for high-dimensional vectors
    lemmatizer = nltk.WordNetLemmatizer()  # create an instance of lemmatizer
    ones_number = 3 # number of nonzero elements in randomly generated high-dimensional vectors
    window_size = 2 #number of neighboring words to consider both back and forth. In other words number of words before/after current word
    zero_vector = np.zeros(dimension)
    test_name = "new_toefl.txt" # file with TOEFL dataset
    data_file_name = "lemmatized.text" # file with the text corpus

    amount_dictionary = {}

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

    dictionary = {} #vocabulary and corresponing random high-dimensional vectors
    word_space = {} #embedings




    #Create a dictionary with the assigned random high-dimensional vectors
    text_file = open(data_file_name, "r")
    for line in text_file: #read line in the file
        words = line.split() # extract words from the line
        for word in words:  # for each word
            if dictionary.get(word) is None: # If the word was not yed added to the vocabulary
                if amount_dictionary[word] < threshold:
                    dictionary[word] = tf.get_random_word_vector(dimension, ones_number) # assign a  
                else:
                    dictionary[word] = np.zeros(dimension) # frequent words are assigned with empty vectors. In a way they will not contribute to the word embedding

            if word_space.get(word) is None: # If the word was not yed added to the vocabulary
                    word_space[word] = np.zeros(dimension) # initialize to all zeros

    text_file.close()


    #Note that in order to save time we only create embeddings for the words needed in the TOEFL task

        #Find all unique words amongst TOEFL tasks and initialize their embeddings to zeros    
    number_of_tests = 0
    text_file = open(test_name, "r") #open TOEFL tasks
    for line in text_file:
    #        words = line.split()
    #        words = [lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word, 'v'), 'n'), 'a') for word in
    #                 words] # lemmatize words in the current test
    #        word_space[words[0]] = np.zeros(dimension)
    #        word_space[words[1]] = np.zeros(dimension)
    #        word_space[words[2]] = np.zeros(dimension)
    #        word_space[words[3]] = np.zeros(dimension)
    #        word_space[words[4]] = np.zeros(dimension)
            number_of_tests += 1
    text_file.close()


        # Processing the text to build the embeddings 
    text_file = open(data_file_name, "r")
    lines = [[],[],[],[]] # neighboring lines
    i = 2
    while i < 4:
            line = "\n"
            while line == "\n":
                line = text_file.readline()
            lines[i] = line.split()
            i += 1
    text_file.close()
    
    text_file = open(data_file_name, "r")
    line = text_file.readline()
    while line != "":
            if line != "\n":
                lines.append(line.split())
                words = lines[2]
                length = len(words)
                i = 0
                while i < length:
#                     print('..working..')
                    if not (word_space.get(words[i]) is None):
                        k = 1
                        word_space_vector = word_space[words[i]]
                        while (i - k >= 0) and (k <= window_size): #process left neighbors of the focus word
                            word_space[words[i]] = np.add(word_space[words[i]], np.roll(dictionary[words[i - k]], -1))         
                            k += 1
                        # Handle different situations if there was not enough neighbors on the left in the current line    
                        if k <= window_size and (len(lines[1])>0): 
                            if len(lines[1]) < 2:
                                if k != 1: #if one word on the left was already added
                                    word_space[words[i]] = np.add(word_space[words[i]], np.roll(dictionary[lines[1][0]], -1)) #update word embedding
                                else:
                                    word_space[words[i]] = np.add(word_space[words[i]],
                                                                  np.roll(dictionary[lines[1][0]], -1)) #update word embedding
                                    word_space[words[i]] = np.add(word_space[words[i]],
                                                                  np.roll(dictionary[lines[0][len(lines[0]) - 1]], -1)) #update word embedding
                            else:
                                if k != 1:
                                    word_space[words[i]] = np.add(word_space[words[i]],
                                                                  np.roll(dictionary[lines[1][len(lines[1]) - 1]], -1)) #update word embedding
                                else:
                                    word_space[words[i]] = np.add(word_space[words[i]],
                                                                  np.roll(dictionary[lines[1][len(lines[1]) - 1]], -1)) #update word embedding
                                    word_space[words[i]] = np.add(word_space[words[i]],
                                                                  np.roll(dictionary[lines[1][len(lines[1]) - 2]], -1)) #update word embedding

                        k = 1
                        while (i + k < length) and (k <= window_size): #process right neighbors of the focus word
                            word_space[words[i]] = np.add(word_space[words[i]], np.roll(dictionary[words[i + k]], 1)) #update word embedding
                            k += 1
                        if (k <= window_size) and (lines[3] != []): #DK added extra condition to handle the end of the corpus
                            if len(lines[3]) < 2:
                                if k != 1:
                                    word_space[words[i]] = np.add(word_space[words[i]], np.roll(dictionary[lines[3][0]], 1)) #update word embedding
                                else:
                                    word_space[words[i]] = np.add(word_space[words[i]], np.roll(dictionary[lines[3][0]], 1)) #update word embedding
                                    word_space[words[i]] = np.add(word_space[words[i]], np.roll(dictionary[lines[4][0]], 1)) #update word embedding
                            else:
                                if k != 1:
                                    word_space[words[i]] = np.add(word_space[words[i]], np.roll(dictionary[lines[3][0]], 1)) #update word embedding
                                else:
                                    word_space[words[i]] = np.add(word_space[words[i]], np.roll(dictionary[lines[3][0]], 1)) #update word embedding
                                    word_space[words[i]] = np.add(word_space[words[i]],
                                                              np.roll(dictionary[lines[3][1]], 1))

                    i += 1
                lines.pop(0)
            line = text_file.readline()

    #store embeddings
    for word in word_space.keys():
        print(word)
        temp=np.array(word_space[word], dtype=float)
        temp += np.random.uniform(-0.1, 0.1, size=(dimension))
#         print('temp withount 0s', np.count_nonzero(temp==0))
        # sys.exit()
        temp=temp/np.linalg.norm(temp) #calculate Euclidean norm of temp
        word_space[word]=temp
        print('**************************Normalising **********************************')
        # maybe this time i dnt need to normalze it here the eval is doing thats
    print(word_space, 'embeddings to test')

    fileSave = codecs.open('./storageWordVecRI.csv', 'w+')
    for word, vec in word_space.items():
        # print("\t".join(word.split('\t')[:101])+'\n')
        # text = "\t".join(row[0].split('\t')[:101])+'\n'
        text = f'{word}\t'
        for i in range(0, len(vec)):
            text += str(vec[i])
            if i+1 != len(vec):
                text += ' '
#         print(text + '\n') 
        fileSave.write(text +'\n')
    fileSave.close()


# In[3]:


def my_split(s):
    # print(list(filter(None, re.split(r'-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?', s))))
    # print(re.findall("-?\d+.?\d*(?:[Ee]-\d+)?", s))
    return list(re.split("-?\d+.?\d*(?:[Ee]-\d+)?", s))[0] ,list(re.findall("-?\d+.?\d*(?:[Ee]-\d+)?", s))
    # list(filter(None, re.split(r'-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?', s)))
    # list(filter(None, re.split(r'(-?[0-9]\.\d*[eE][-+]?[0-9]+)', s)))


# In[4]:


dimension = 300 # parameter for Word2vec size of vectors for word embedding

threshold = 0.00055 # parameter for Word2vec

sentences = []

file = open("lemmatized.text", "r")
for line in file: # read the file and create list which contains all sentences found in the text
    sentences.append(line.split())
# train word2vec on the two sentences

epochsRI = 11
runs = 6

# get_ipython().system('pwd')
for epoch in range(21):
    for run in range(3):
        w2vObjectRI = gensim.models.Word2Vec(min_count=1, sample=threshold, sg=1,size=dimension, negative=15, iter=epoch, window=3, workers=8) # create only the shell

        print('Starting vocab build')
        # t = time()
        w2vObjectRI.build_vocab(sentences, progress_per=10000) #here is the vocab being built as told in google groups gensim
        
#         sanity check just so im not going insane
        print('name', w2vObjectRI['name'])
        
        #Next embeddings_RI cells should try with RI
#         Create random random-indexing
        print('creating RI')
        createRI() #this new call from inside
        print('end creation stored now')
        # load matrix vocab into a dict of word:vect
        f = codecs.open(f'./storageWordVecRI.csv', encoding='utf-8')##os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
        embeddings_RI = {}
        for num, line in enumerate(f):
            values = my_split(line) # line.split('\t')
            word = values[0].rstrip()
            vector = values[1]
        #     print('values', values, '\n', word)
        #     sys.exit()
            if len(vector) != 300:
                print(word)
                continue
            else:
                coefs = np.asarray(vector, dtype=np.float32)
                embeddings_RI[word] = coefs
        # sys.exit()
        f.close()
        
#         set operations here
        setOfGensimVocab = set(w2vObjectRI.wv.vocab)
        setOfWordNetVocab = set(embeddings_RI.keys())

        print('len of sets w2v, wn', len(setOfGensimVocab), len(setOfWordNetVocab))

        setIntersection = setOfGensimVocab.intersection(setOfWordNetVocab)
#         print('len of intersection', len(setIntersection), setIntersection)

        
#         for elem in embeddings_RI.keys():
#             if len(embeddings_RI[elem]) != 300:
#                 print('here', elem) #cast it to the fire
#                 w2vObjectRI.wv[elem] = np.asarray(embeddings_RI[elem], dtype=np.float32)
#         print('Done!!!')
    
        for elem in setIntersection:
        #     print(elem)
        #     print('w2v:', w2vObject.wv[elem], type(w2vObject.wv[elem]))
        #     print('w2v:', embeddings_index[elem], type(embeddings_index[elem]))
        #     break
            if len(embeddings_RI[elem]) != 300:
                print('here', elem) #cast it to the fire
            w2vObjectRI.wv[elem] = np.asarray(embeddings_RI[elem], dtype=np.float32)
        print('Done!!!')

#         sanity check just so im not going insane
        print('name', w2vObjectRI['name'])

        # print(w2vObject.wv['the'], 'after before')
        w2vObjectRI.train(sentences, total_examples=w2vObjectRI.corpus_count, epochs=epoch)#w2vObject.iter)
        # print(w2vObject.wv['the'], 'after train')
        # sys.exit()
        w2vObjectRI.wv.save_word2vec_format(f'./WithRIRuns/W2V_OnlyRI{epoch}Run{run}.pkl.gz')# binary=False) #encoding='utf-8' )
        # print('saved')


# # In[7]:


# # %cd RI_DenisCode/
# get_ipython().system('pwd')
# # !python EmbeddingsRI_DK.py


# # In[ ]:





############### Start here #####

# # In[25]:


# numOfRun = 6

# for epoch in range(11):
#     for run in range(numOfRun):
#         # train all steps in one go
#         model = gensim.models.Word2Vec(sentences, min_count=1, sample=threshold, sg=1,size=dimension, negative=15, iter=epoch, window=3, workers=8) # create model using Word2Ve with the given parameters

#         # print(model.wv, 'in one go')

#         model.wv.save_word2vec_format(f'Averages/GensimEpochs{epoch}Run{run}.txt', binary=False) 
#         #only for word2vec clean
#         print('saved')
#         # sys.exit()
        


# # In[9]:


# w2vObject = gensim.models.Word2Vec(min_count=1, sample=threshold, sg=1,size=dimension, negative=15, iter=epochsNum, window=3, workers=8) # create only the shell

# print('Starting vocab build')
# # t = time()
# w2vObject.build_vocab(sentences, progress_per=10000) #here is the vocab being built as told in google groups gensim

############ here stop

# print(w2vObject.wv['the'], 'before train')


# # In[10]:


# f = codecs.open(f'../../../WordNetGraphHD/StorageEmbeddings/EmbeddingFormat{dimension}.txt', encoding='utf-8')##os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
# embeddings_index = {}
# for num, line in enumerate(f):
#     values = my_split(line) # line.split('\t')
#     word = values[0].rstrip()
#     # vector = ''.join(num for num in values[1:])
#     vector = values[1]
#     # print(word, vector)
#     if len(vector) != 300:
# #         print(line, 'here not 300')
#         continue
#         # print(my_split(line))
#         # sys.exit()
        
#     else:
#         coefs = np.asarray(vector, dtype=np.float32)
#     # print(coefs.shape[0])
#     # if coefs.shape[0] is not 300:
#         # print(word, num)
#             # np.asarray(values[1], dtype='float32') # weird flex but ok!
#         # print(coefs, 'coefs as np array?')
#         embeddings_index[word] = coefs
#     # print(word)
#     # sys.exit()
#     # if 'be' in word:
#     #     print('it does', word)
#     # print(word)
#     # print('as array', np.fromstring(values[1].rstrip(), dtype='float32', sep=' '))
#     # sys.exit()
#     # print(word)
#     # if useRI == True:
#     #     # print('using RI', vales[1:-1])
#     #     # print('strip \n ', values[1].rstrip())
#     #     coefs = np.fromstring(values[1].rstrip(), dtype='float32', sep=' ')
#     #     # print('Coefs success')
#     #     # print(coefs, 'coefs as np array?')
#     # else:
#     #     coefs = np.asarray(values[1:], dtype='float32')
#     # coefs = np.fromstring(vector.(), dtype='float32', sep=' ')
#     # coefs = np.asarray(vector)
#     # # print(coefs.shape[0])
#     # # if coefs.shape[0] is not 300:
#     #     # print(word, num)
#     #         # np.asarray(values[1], dtype='float32') # weird flex but ok!
#     #     # print(coefs, 'coefs as np array?')
#     # embeddings_index[word] = coefs
#     # print('word', word, 'vector', str(coefs))
# # sys.exit()
# f.close()


# # In[11]:


# setOfGensimVocab = set(w2vObject.wv.vocab)
# setOfWordNetVocab = set(embeddings_index.keys())

# print('len of sets w2v, wn', len(setOfGensimVocab), len(setOfWordNetVocab))

# setIntersection = setOfGensimVocab.intersection(setOfWordNetVocab)
# print('len of intersection', len(setIntersection), setIntersection)


# # In[12]:


# # i = w2vObject# for elem in w2vObject.wv.vocab:
# #     if elem in embeddings_index.keys():
# # #        
# # # print('be', w2vObject.wv[elem])
# # #         print(embeddings_index[elem])
# #         w2vObject.wv[elem] = embeddings_index[elem]
# #         i += 1
# # #         print('Found one', i)

# # print(i)
# for elem in setIntersection:
# #     print(elem)
# #     print('w2v:', w2vObject.wv[elem], type(w2vObject.wv[elem]))
# #     print('w2v:', embeddings_index[elem], type(embeddings_index[elem]))
# #     break
#     if len(embeddings_index[elem]) != 300:
#         print('here', elem) #cast it to the fire
#     w2vObject.wv[elem] = np.asarray(embeddings_index[elem], dtype=np.float32)
# print('Done!!!')


# # In[13]:


# for epoch in epochsNum:
#     for run in numOfRuns:
#         w2vObject.train(sentences, total_examples=w2vObject.corpus_count, epochs=epochsNum)#w2vObject.iter)
# #         print(w2vObject.wv, 'after train')
#         w2vObject.wv.save_word2vec_format(f'./GensimOneWNet{epoch}Run{run}.txt', binary=False) #encoding='utf-8' )
# # print('saved')
############## commented wordnet end

# In[37]:


#Next embeddings_RI cells should try with RI
# load matrix vocab into a dict of word:vect
# f = codecs.open(f'./storageWordVecRI_DK.csv', encoding='utf-8')##os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
# embeddings_RI = {}
# for num, line in enumerate(f):
#     values = my_split(line) # line.split('\t')
#     word = values[0].rstrip()
#     vector = values[1]
# #     print('values', values, '\n', word)
# #     sys.exit()
#     if len(vector) != 300:
#         print(word)
#         continue
#     else:
#         coefs = np.asarray(vector, dtype=np.float32)
#         embeddings_RI[word] = coefs
# # sys.exit()
# f.close()


# In[38]:


# print(embeddings_RI.keys())


# # In[76]:


# #create gensim vocab
# epochsRI = 10
# w2vObjectRI = gensim.models.Word2Vec(min_count=1, sample=threshold, sg=1,size=dimension, negative=15, iter=epochsRI, window=3) # create only the shell

# print('Starting vocab build')
# # t = time()
# w2vObjectRI.build_vocab(sentences, progress_per=10000) #here is the vocab being built as told in google groups gensim

# # print(w2vObject.wv['the'], 'before train')


# # In[7]:


# setOfGensimVocab = set(w2vObjectRI.wv.vocab)
# setOfWordNetVocab = set(embeddings_RI.keys())

# print('len of sets w2v, wn', len(setOfGensimVocab), len(setOfWordNetVocab))

# setIntersection = setOfGensimVocab.intersection(setOfWordNetVocab)
# print('len of intersection', len(setIntersection), setIntersection)


# # In[ ]:


# for elem in setIntersection:
# #     print(elem)
# #     print('w2v:', w2vObject.wv[elem], type(w2vObject.wv[elem]))
# #     print('w2v:', embeddings_index[elem], type(embeddings_index[elem]))
# #     break
#     if len(embeddings_RI[elem]) != 300:
#         print('here', elem) #cast it to the fire
#     w2vObjectRI.wv[elem] = np.asarray(embeddings_RI[elem], dtype=np.float32)
# print('Done!!!')


# # In[79]:


# # print(w2vObject.wv['the'], 'after before')
# w2vObject.train(sentences, total_examples=w2vObject.corpus_count, epochs=epochsRI)#w2vObject.iter)
# # print(w2vObject.wv['the'], 'after train')
# # sys.exit()
# w2vObject.wv.save_word2vec_format(f'./RIResults/GensimOneRI{epochsRI}.txt', binary=False) #encoding='utf-8' )
# # print('saved')


# In[81]:

# # commented starts from here
# fileNamesRI = ['outputsGensimOneRI1.csv', 'outputsGensimOneRI2.csv',
#                   'outputsGensimOneRI3.csv', 'outputsGensimOneRI4.csv',
#                   'outputsGensimOneRI5.csv', 'outputsGensimOneRI6.csv',
#                   'outputsGensimOneRI7.csv', 'outputsGensimOneRI8.csv',
#                   'outputsGensimOneRI9.csv', 'outputsGensimOneRI10.csv']
# # fileNamesWN = ['outputFilenamesRhos1epochWithWordNetTrue.csv', 'outputFilenamesRhos2epochWithWordNetTrue.csv',
# #               'outputFilenamesRhos3epochWithWordNetTrue.csv', 'outputFilenamesRhos4epochWithWordNetTrue.csv',
# #               'outputFilenamesRhos5epochWithWordNetTrue.csv', 'outputFilenamesRhos6epochWithWordNetTrue.csv', 
# #               'outputFilenamesRhos7epochWithWordNetTrue.csv', 'outputFilenamesRhos8epochWithWordNetTrue.csv',
# #               'outputFilenamesRhos9epochWithWordNetTrue.csv', 'outputFilenamesRhos10epochWithWordNetTrue.csv']
# x1 = []
# y1 = []
# x2 = []
# y2 = []
# x3 = []
# y3 = []
# x4 = []
# y4 = []
# x5 = []
# y5 = []
# x6 = []
# y6 = []
# x7 = []
# y7 = []
# x8 = []
# y8 = []
# x9 = []
# y9 = []
# x10 = []
# y10 = []
# x11 = []
# y11 = []
# x12 = []
# y12 = []
# x13 = []
# y13 = []

# for elem in fileNamesRI:
#     f = codecs.open('../../../eval-word-vectors/ResultsRI/'+elem, 'r', 'utf-8')
#     next(f)
#     for idx,line in enumerate(f):
# #         print(f'{idx} {line.split()}')
# #         x1.append(line.split()[0])
#         y1.append(float(line.split()[0]))
#         y2.append(float(line.split()[1]))
#         y3.append(float(line.split()[2]))
#         y4.append(float(line.split()[3]))
#         y5.append(float(line.split()[4]))
#         y6.append(float(line.split()[5]))
#         y7.append(float(line.split()[6]))
#         y8.append(float(line.split()[7]))
#         y9.append(float(line.split()[8]))
#         y10.append(float(line.split()[9]))
#         y11.append(float(line.split()[10]))
#         y12.append(float(line.split()[11]))
#         y13.append(float(line.split()[12]))
# print(y1)


# # In[83]:


# x1 = range(1,11)
# # y1 = [20,40,10]
# # plotting the line 1 points 
# plt.plot(y1, label = "EN-WS-353-ALL")
# # line 2 points
# x2 = range(1,11)
# # y2 = [40,10,30]
# # plotting the line 2 points 
# plt.plot(x2, y2, label = "EN-RW-STANFORD")

# x3 = range(1,11)
# # y3
# # plotting the line 3 points 
# plt.plot(x3, y3, label = "EN-WS-353-SIM")

# x4 = range(1,11)
# # y4
# # plotting the line 4 points 
# plt.plot(x4, y5, label = "EN-MTurk-771")


# x5 = range(1,11)
# # y5
# # plotting the line 5 points 
# plt.plot(x5, y5, label = "EN-YP-130")

# x6 = range(1,11)
# # y6
# # plotting the line 6 points 
# plt.plot(x6, y6, label = "EN-MEN-TR-3k")

# x7 = range(1,11)
# # y7
# # plotting the line 7 points 
# plt.plot(x7, y7, label = "EN-VERB-143")

# x8 = range(1,11)
# # y8
# # plotting the line 8 points 
# plt.plot(x8, y8, label = "EN-WS-353-REL")

# x9 = range(1,11)
# # y9
# # plotting the line 9 points 
# plt.plot(x9, y9, label = "EN-RG-65")

# x10 = range(1,11)
# # y10
# # plotting the line 10 points 
# plt.plot(x10, y10, label = "EN-MTurk-287")

# x11 = range(1,11)
# # y11
# # plotting the line 11 points 
# plt.plot(x11, y11, label = "EN-SimVerb-3500")

# x12 = range(1,11)
# # y12
# # plotting the line 12 points 
# plt.plot(x12, y12, label = "EN-SIMLEX-999")

# x13 = range(1,11)
# # y13
# # plotting the line 13 points 
# plt.plot(x13, y13, label = "EN-MC-30")



# plt.xlabel('x - axis')
# # Set the y axis label of the current axis.
# plt.ylabel('y - axis')
# # Set a title of the current axes.
# plt.title('Plot scores of Gensim RI')
# # show a legend on the plot
# plt.legend()
# # plt.ylim((0.0, 1.0))
# # Display a figure.
# plt.savefig('GensimRI.png')
# plt.show()


# # In[84]:


# x1 = range(1,11)
# # y1 = [20,40,10]
# # plotting the line 1 points 
# plt.plot(y1, label = "EN-WS-353-ALL")
# # line 2 points
# x2 = range(1,11)
# # y2 = [40,10,30]
# # plotting the line 2 points 
# plt.plot(x2, y2, label = "EN-RW-STANFORD")

# x3 = range(1,11)
# # y3
# # plotting the line 3 points 
# plt.plot(x3, y3, label = "EN-WS-353-SIM")

# x4 = range(1,11)
# # y4
# # plotting the line 4 points 
# plt.plot(x4, y5, label = "EN-MTurk-771")


# x5 = range(1,11)
# # y5
# # plotting the line 5 points 
# plt.plot(x5, y5, label = "EN-YP-130")

# x6 = range(1,11)
# # y6
# # plotting the line 6 points 
# plt.plot(x6, y6, label = "EN-MEN-TR-3k")

# x7 = range(1,11)
# # y7
# # plotting the line 7 points 
# plt.plot(x7, y7, label = "EN-VERB-143")

# x8 = range(1,11)
# # y8
# # plotting the line 8 points 
# plt.plot(x8, y8, label = "EN-WS-353-REL")

# x9 = range(1,11)
# # y9
# # plotting the line 9 points 
# plt.plot(x9, y9, label = "EN-RG-65")

# x10 = range(1,11)
# # y10
# # plotting the line 10 points 
# plt.plot(x10, y10, label = "EN-MTurk-287")

# x11 = range(1,11)
# # y11
# # plotting the line 11 points 
# plt.plot(x11, y11, label = "EN-SimVerb-3500")

# x12 = range(1,11)
# # y12
# # plotting the line 12 points 
# plt.plot(x12, y12, label = "EN-SIMLEX-999")

# x13 = range(1,11)
# # y13
# # plotting the line 13 points 
# plt.plot(x13, y13, label = "EN-MC-30")



# plt.xlabel('x - axis')
# # Set the y axis label of the current axis.
# plt.ylabel('y - axis')
# # Set a title of the current axes.
# plt.title('Plot scores of Gensim with RI')
# # show a legend on the plot
# # plt.legend()
# # plt.ylim((0.0, 1.0))
# # Display a figure.
# plt.savefig('GensimRINoLegend.png')
# plt.show()


# # In[ ]:


