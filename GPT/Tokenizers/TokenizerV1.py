import re 
import heapq


class My_tokenizer:
    def __init__(self, vocab=None, size=None,tokens=None):

        if vocab!=None:
            self.size= len(vocab.items())
            self.vocab=vocab

        if tokens!=None:
            self.tokens=tokens
        
        self.text=None

    def clean_up(self,text):
        def clean_text(text):
            text = text.replace("\n", " ")
            text = re.sub(r"\s+", " ", text)
            text = text.replace(r"."," <eos.>")
            text = text.replace(r"?"," <eos?>")
            text = text.replace(r"!"," <eos!>")
            return text.strip()
        def split_text(text):
            text = re.split(" ",text)
            return text


        new_text = clean_text(text)
        new_text = split_text(new_text)
        return new_text
    def fit(self,text,size):
        Vocabulary = {}
        for word in text:
            if word in Vocabulary.keys():
                Vocabulary[word]+=1
            else:
                Vocabulary[word]=1
 
        
        
        sorted_words = heapq.nlargest(size, Vocabulary.items(), key=lambda x: x[1])

        vocab={}
        for idx in range(len(sorted_words)):
            tok = sorted_words[idx][0]
            vocab[tok]=idx

        vocab["<unk>"]=idx+1

        self.unk_key = idx+1

        self.size = len(vocab.keys())
        self.vocab= vocab

        print("Fitted!")

        return self.vocab, [self.vocab.get(key) for key in ["<eos.>","<eos!>","<eos?>"]]            


        
    def transform(self,text,vocal=True):
        new_text=[]
        ctr=0
        unk_key=self.unk_key
        for word in text:
            if word not in self.vocab.keys():
                new_text.append(unk_key)
                ctr+=1
            else: 
                new_text.append(self.vocab[word])
        if vocal:
            print("Text out of vocabulary:",ctr/len(text))
        return new_text
    

import numpy as np

class Simple_embedding:
    def __init__(self, embedding_length,id_length,Positional_encoding = "Sine",):

        self.size = embedding_length
        self.pos_encoding = Positional_encoding
        self.id_length= id_length

        self.mu=0

        self.var=1

    #Innitialize weights: 
        
    def set_means(self,mu,var):

        self.mu=mu
        self.var=var

    def initialize(self):
        A = np.random.normal(self.mu,np.sqrt(self.var),(self.id_length,self.size))
        

        self.embedding_table = A
        return A

    def pos_enc(self, length):
        if length ==0:
            return np.zeros(1)
        if self.pos_encoding == "Sine":

            range_points = np.arange(0,np.pi,np.pi/length)
            encoding = np.sin(range_points)
            return encoding
            
        else: 

            return np.zeros(length)
            
        #if Positional_encoding != "Sine":
        #    raise RuntimeWarning("no positional encoding selected")
              
    def set_end_points(self, end_point_list): 
        self.end_points = list(end_point_list)


    def transform(self,token_list):
        v = self.embedding_table[token_list]
        
        counter = 0
        for token in range(len(token_list)):
            if token not in self.end_points:
                counter += 1
            else: 
                v[:,token-counter:token]+=self.pos_enc(counter)
                counter = 0


        return v
        
        
