"""
COMS W4705 - Natural Language Processing - Spring 2020
Homework 2 - Parsing with Context Free Grammars 
Yassine Benajiba
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg

### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        # TODO, part 2
        # CKY for CFG
        
        table= None
        n = len(tokens)
        
        # content: possible nontermial ends in the specified position
        # eg{(i,j): ['NP']}
        table = defaultdict(list)
        
        # initialization
        for i in range(0,n):
            token = tokens[i]
            temp_list = []
            for lhs in self.grammar.rhs_to_rules[tuple([token])]:
                temp_list.append(lhs[0])
            table[(i,i+1)] = temp_list
         
        
        # CKY parsing for CFG
        for length in range(2, n+1):
            for i in range(0, n-length+1):
                j = i + length
                for k in range(i+1, j):
                        temp_B = table[(i,k)]
                        temp_C = table[(k,j)]
                        temp_tuples_list = []
                        for temp1 in temp_B:
                            for temp2 in temp_C:
                                temp_tuples_list.append(tuple([temp1,temp2]))
                        
                        for temp_tuple in temp_tuples_list:
                            if len(self.grammar.rhs_to_rules[temp_tuple]) > 0:
                                for lhs in self.grammar.rhs_to_rules[temp_tuple]:
                                    if lhs[0] not in table[(i,j)]:
                                        table[(i,j)].append(lhs[0])
                                                           
        # check if true
        if self.grammar.startsymbol in table[(0,n)]:
            return True
        
        return False 
        
       
    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        # TODO, part 3
        table= None
        probs = None
        
        # content: tuple of split array
        table = defaultdict(defaultdict)
        # content: the prob 
        probs = defaultdict(lambda: defaultdict(float))
        n = len(tokens)
    
        
        # initialization
        for i in range(0,n):
            token = tokens[i]
            for lhs in self.grammar.rhs_to_rules[tuple([token])]:
                probs[(i,i+1)][lhs[0]] = math.log(lhs[2], 2)
                table[(i,i+1)][lhs[0]] = token

        
        # parsing
        for length in range(2, n+1):
            for i in range(0, n-length+1):
                j = i + length
                for k in range(i+1, j):
                    temp_B_dict = table[(i,k)]
                    temp_C_dict = table[(k,j)]
                    temp_B = temp_B_dict.keys();
                    temp_C = temp_C_dict.keys();
                    temp_tuples_list = []
                    for temp1 in temp_B:
                        for temp2 in temp_C:
                            temp_tuples_list.append(tuple([temp1,temp2]))
                    # possible LHS
                    for temp_tuple in temp_tuples_list:
                            if len(self.grammar.rhs_to_rules[temp_tuple]) > 0:
                                for lhs in self.grammar.rhs_to_rules[temp_tuple]:
                                    # with log: multiply means addition
                                    new_log_prob = math.log(lhs[2], 2) + probs[(i,k)][temp_tuple[0]] + probs[(k,j)][temp_tuple[1]]
                                    if probs[(i,j)][lhs[0]] != 0.0:
                                        if probs[(i,j)][lhs[0]] < new_log_prob:
                                            probs[(i,j)][lhs[0]] = new_log_prob
                                            table[(i,j)][lhs[0]] = ((temp_tuple[0], i, k), (temp_tuple[1], k, j))
                                    else:
                                        probs[(i,j)][lhs[0]] = new_log_prob
                                        table[(i,j)][lhs[0]] = ((temp_tuple[0], i, k), (temp_tuple[1], k, j))  
                                    
        return table, probs


def get_tree(chart, i,j,nt): 
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4
    #Recursively traverse the parse chart to assemble this tree.
    temp_list = []
    temp_list.append(nt)
    # left child
    if type(chart[(i,j)][nt]) is not str:
        for child in chart[(i,j)][nt]:
            temp_list.append(get_tree(chart, child[1], child[2], child[0]))      
    else:
        temp_list.append(chart[(i,j)][nt])
    return tuple(temp_list)


if __name__ == "__main__":
    
    with open('atis3.pcfg','r') as grammar_file: 
        grammar = Pcfg(grammar_file) 
        parser = CkyParser(grammar)
        toks =['flights', 'from','miami', 'to', 'cleveland','.'] 
        #print(parser.is_in_language(toks))
        #table,probs = parser.parse_with_backpointers(toks)
        #assert check_table_format(chart)
        #assert check_probs_format(probs)
   
