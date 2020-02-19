import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Spring 2020
Homework 1 - Programming Component: Trigram Language Models
Yassine Benajiba
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """

    # sequence seen as a list 
    # n is the number deciding n-grams
    # padded sequence as a list
    temp_list = sequence
    if n==1:
        temp_list.insert(0,"START")
    temp_list.append("STOP")
    
    for i in range(n-1):
        temp_list.insert(0,"START");
    
    
    # create the n-grams sequence
    n_grams = []
    for i in range(len(temp_list)-n+1):
        temp = []
        for j in range(n): 
            temp.append(temp_list[i+j])
        n_grams.append(tuple(temp))
            
    return n_grams


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
        # add by Shusen Xu
        self.corpusfiles = corpusfile
        self.total_count_unigram = 0
        self.total_count_bigram = 0
        self.total_count_trigram = 0
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)
        


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
           
        '''
        self.unigramcounts = {} # might want to use defaultdict or Counter instead
        self.bigramcounts = {} 
        self.trigramcounts = {} 
        ''' 
        # use defaultdict
        self.unigramcounts = defaultdict(int)
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)
        
        
        ##Your code here
        # count unigramcounts
        for sentence in corpus:
            unigrams = get_ngrams(sentence,1)
            for t in unigrams:
                self.unigramcounts[t] += 1
                
        # count bigramcounts
        corpus = corpus_reader(self.corpusfiles, self.lexicon)
        for sentence in corpus:
            bigrams = get_ngrams(sentence,2)
            for t in bigrams:
                self.bigramcounts[t] += 1
                
        # count trigramcounts
        corpus = corpus_reader(self.corpusfiles, self.lexicon)
        for sentence in corpus:
            trigrams = get_ngrams(sentence,3)
            for t in trigrams:
                self.trigramcounts[t] += 1
        
        return
            
    
        # count unigramcounts

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
           
        temp_list = []
        temp_list.append(trigram[0])
        temp_list.append(trigram[1])
        
        
        
        # method 1
        # if the trigram is ('start', 'start', "anyword"), we consider the counts of it 
        # is equal to thec counts bigram('start', "anyword) because they both refer to same situation
        # and we treat the counts of bigram('start', 'start') as the count of unigram('start')
        
    
        if trigram[0] == 'START':
            return float(self.trigramcounts[trigram] / self.unigramcounts[('START',)])
            
        
        
        # if bigramcounts with unseen word or self.bigramcounts[tuple(temp_list)] == 0
        # we count trigramcounts also as 0
        if self.bigramcounts[tuple(temp_list)] == 0: return 0.0
        return float(self.trigramcounts[trigram] / self.bigramcounts[tuple(temp_list)])
       

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """    
        temp_list = []
        temp_list.append(bigram[0])
        return float(self.bigramcounts[bigram] / self.unigramcounts[tuple(temp_list)])
    
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.
        
        # unigram seen as one unigram, such as ('START',)
        # count the total number
    
        if self.total_count_unigram == 0:
            for key,value in self.unigramcounts.items():
                self.total_count_unigram += value
        
        # calculate possibility
        return float(self.unigramcounts[unigram] / self.total_count_unigram)

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        # every time compute the word with highest probability
        '''
        res_sentence = []
        temp = ''
        
        temp_list = []
        temp_list.append('START')
        temp_list.append('START')
        max_possibility = 0.0
        max_word = ''
        current_possibility = 1.0
        temp_possibility = 0.0
        
        for i in range(0, 20):
            max_possibility = 0.0 
            for trigram in self.trigramcounts.keys():
                if trigram[0] == temp_list[0] and trigram[1] == temp_list[1]:
                    temp_possibility = self.raw_trigram_probability(trigram)
                    if temp_possibility > max_possibility and trigram[2] != 'UNK':
                        max_possibility = temp_possibility
                        max_word = trigram[2]
                       
            temp_list.pop(0)
            temp_list.append(max_word)
            res_sentence.append(max_word)
            print(temp_list)
            
            if max_word == 'STOP':
                break
        return res_sentence   
        '''
        # may use the vertibi algorithm
        return
    
    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        
        temp_list = []
        temp_list.append(trigram[2])
        p_mle_unigram = self.raw_unigram_probability(tuple(temp_list))
        
        temp_list = []
        temp_list.append(trigram[1])
        temp_list.append(trigram[2])
        p_mle_bigram = self.raw_bigram_probability(tuple(temp_list))
        
        p_mle_trigram = self.raw_trigram_probability(trigram) 
        
        
        res = float(lambda1 * p_mle_unigram + lambda2 * p_mle_bigram + lambda3 * p_mle_trigram)
    
        return res
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        
        # here compute trigram model
        # list of tuple
        trigrams = get_ngrams(sentence,3)
        temp_probability = 0.0
        res = 0.0
        for trigram in trigrams:
            temp_probability = self.smoothed_trigram_probability(trigram)
            res += float(math.log2(temp_probability))
        
        #return float("-inf")
        return float(res)

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        # contain starts and ends
        total_number_words = 0
        pp = 0.0
        total_possibility = 0.0 
        for sentence in corpus:
            # contain one'START' and one'STOP'
            total_number_words += ( 2 + len(sentence))
            total_possibility += self.sentence_logprob(sentence)
        
        pp = float(math.pow(2, - (total_possibility / total_number_words)))
        
        #return float("inf") 
        return float(pp)




def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0  
    

        for f in os.listdir(testdir1):
            pp_high = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp_low = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            
            total += 1
            if pp_high <= pp_low: correct += 1
            
              
    
        for f in os.listdir(testdir2):
            pp_low = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp_high = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            
            total += 1
            if pp_low <= pp_high: correct += 1

        return float(correct / total)
        
    


if __name__ == "__main__":

    # model = TrigramModel(sys.argv[1])
    model = TrigramModel("./hw1_data/brown_train.txt")


    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    #dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    #pp = model.perplexity(dev_corpus)
    #print(pp)


    # Essay scoring experiment: 
    # acc = essay_scoring_experiment('train_high.txt', 'train_low.txt", "test_high", "test_low")
    # print(acc)

