#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import gensim
import numpy as np
from collections import defaultdict

# Participate in the 4705 lexical substitution competition (optional): NO
# Alias: [please invent some name]

def tokenize(s):
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos):
    # Part 1
    # return value a list
    possible_synonyms = []
    
    temp_set = set()
    for l in wn.lemmas(lemma, pos):
                for s_l in l.synset().lemmas():
                    if(s_l.name() != lemma):
                        temp_set.add(s_l.name())
                    
    possible_synonyms = list(temp_set)    
    
    return possible_synonyms

def smurf_predictor(context):
    """
    Just suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(contex):
    # replace for part 2
    # dict that count the occurency
    counter = defaultdict(int)
    lemma = context.lemma
    pos = context.pos
    
    for l in wn.lemmas(lemma, pos):
                for s_l in l.synset().lemmas():
                    if(s_l.name() != lemma):
                        counter[s_l.name()] += 1
    
    # find the word with highest frequency
    max_word = " "
    max_frequency = 0
    for word, frequency in counter.items():
        if(frequency > max_frequency):
            max_word = word
            max_frequency = frequency
        
    return max_word

def wn_simple_lesk_predictor(context):
    return None #replace for part 3        
   
class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self, context):
        # replace for part 4
        # same as part 1 
        possible_synonyms = get_candidates(context.lemma, context.pos)
        
        highest_similarity = 0.0 
        highest_synonym = ""
        for synonym in possible_synonyms:
            # ignoring the vocab that not in the model
            if(synonym not in self.model.vocab):
                continue
            temp_similarity = self.model.similarity(context.lemma, synonym)
            
            if(temp_similarity > highest_similarity):
                highest_similarity = temp_similarity
                highest_synonym = synonym
        
        return highest_synonym


    def predict_nearest_with_context(self, context): 
        # replace for part 5
        stop_words = stopwords.words('english')
        vector_target = self.model.wv[context.lemma]
        vector_sentence = np.zeros(vector_target.shape)
        possible_synonyms = get_candidates(context.lemma, context.pos)
        
        # possible 
        # remove the stop words within +-5 window and add the left data
        # build sentence vector within the window
        windows_words = []
        i = len(context.left_context)-1
        j = 0
        while(i>=0 and j<5):
            if(context.left_context[i] not in stop_words):
                windows_words.append(context.left_context[i])
            i -= 1
            j += 1
        
        k = 0
        while(k< len(context.right_context) and k<5):
            if(context.right_context[k] not in stop_words):
                windows_words.append(context.right_context[k])
            k += 1
            
        # compute the window words vector
        for word in windows_words:
            if(word not in self.model.vocab):
                continue
            vector_sentence += self.model.wv[word]
        
        # computer the synonym that has highest similarity with the sentence vector
        highest_similarity = 0.0 
        highest_synonym = ""
        for synonym in possible_synonyms:
            # ignoring the vocab that not in the model
            if(synonym not in self.model.vocab):
                continue
            temp_similarity = self.model.similarity(context.lemma, synonym)
            if(temp_similarity > highest_similarity):
                highest_similarity = temp_similarity
                highest_synonym = synonym
        
        return highest_synonym

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        #prediction = smurf_predictor(context) 
        # test for the first part
        #prediction = wn_frequency_predictor(context)


        # test for part 4
        #prediction = predictor.predict_nearest(context)

        # test for part 5
        //prediction = predictor.predict_nearest_with_context(context)

        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))








