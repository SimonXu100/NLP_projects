from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)    
        
        while state.buffer: 
            # pass
            # TODO: Write the body of this loop for part 4 
            # step 1: 
            features = self.extractor.get_input_representation(words, pos, state)
            possible_actions = self.model.predict(features.reshape([1,6]))
            possible_actions = possible_actions.reshape(91)
            # step 2: select the highest scoring permitted transition
           
            # create a possible action indices list sorted by their possibility(largest one comes first)
            # sorted_actions_indices = np.flipud(np.argsort(possible_actions))
            sorted_actions_indices = np.flipud(np.argsort(possible_actions))
            
            # going through and find the highest scoring permitted trasition
            for i in sorted_actions_indices:
                flag = False
                # check the current transition whether permitted or not
                if self.output_labels[i][0] == "shift":
                    if state.stack and len(state.buffer) == 1:
                        flag = False
                    else:
                        flag = True    
                        
                elif self.output_labels[i][0] == "left_arc":
                    if not state.stack:
                        flag = False
                    elif state.stack[-1] == 0:
                        flag = False
                    else:
                        flag = True
           
                elif self.output_labels[i][0] == "right_arc":
                    if not state.stack:
                        flag = False
                    else: flag = True
                
                # when flag == True, it states that the cuurent transition is permitted
                if flag == True:
                    transition = self.output_labels[i]
                    # update the state accordingly
                    if transition[0] == "shift":
                        state.shift()
                    elif transition[0] == "left_arc":
                        state.left_arc(transition[1])
                    elif transition[0] == "right_arc":
                        state.right_arc(transition[1])   
                    break
    
        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
        
