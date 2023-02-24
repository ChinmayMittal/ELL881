import random
import math
from tqdm import tqdm
import argparse
import numpy as np
from sklearn.linear_model import LinearRegression
from preprocess import preprocess, read_data
from n_gram import create_n_grams

parser = argparse.ArgumentParser(description='N Gram Language Models with Good Turing Smoothing... ')
parser.add_argument("-n", action="store", default=1, type=int)
parser.add_argument("--generate", action="store", default=False, type=bool)
parser.add_argument("--generate_cnt", action="store", default=1, type=int)

args = parser.parse_args()


class SimpleGoodTuringLM():
    
    def __init__(self, n):
        
        self.n = n
        self.context = {} #### doubly index dict context -> next_word -> how many times next_word follows context
        self.context_cnt = {} ### count of how many times context has appeared        
        self.vocabulary = set(("<s>", "</s>", "<unk>"))
        
    def update(self, sentence):
        ### sentence is a list of tokens including <s> and </s>
        for token in sentence:
            self.vocabulary.add(token)
        ### store n-gram counts
        ngrams = create_n_grams(sentence, n=self.n) ### creates a list of ngrams
        for ngram in ngrams:
            context, next_word = ngram
            if( context not in self.context):
                self.context[context] = {}
                self.context_cnt[context] = 0
            if( next_word not in self.context[context]):
                self.context[context][next_word] = 0
            
            self.context[context][next_word] += 1
            self.context_cnt[context] += 1      
            
    def probability_for_next_word(self, context, token):
        
        if token not in self.vocabulary:
            token = "<unk>"
        
        if context not in self.context:
            return 1/len(self.vocabulary)
        
        frequency_count = {}
        for next_word in self.context[context]:
            f = self.context[context][next_word]
            if f in frequency_count:
                frequency_count[f] += 1
            else:
                frequency_count[f] = 1
        
        frequencies = list(frequency_count.keys())
        frequencies.sort()
        if(len(frequencies) <= 20):
            ### to few frequency counts for good turing, simple add-k smoothing
            if token in self.context[context]:
                return (0.1+self.context[context][token]) / (0.1*len(self.vocabulary)+self.context_cnt[context])
            else:
                return (0.1)/ (0.1*len(self.vocabulary)+self.context_cnt[context])
            
        N_r = [frequency_count[f] for f in frequencies] ### N_r
        # print(frequencies)
        # print(N_r)
        ### smooth out N_r to get Z_r; Z_r = 2*N_r /(t-q)
        Z_r = []
        for i in range(len(frequencies)):
            q, r, t = None, frequencies[i], None
            n_r, z_r = N_r[i], None
            if(i==0 and len(frequencies) == 1):
                z_r = n_r
            elif(i==0):
                q = 0
                t = frequencies[i+1]
                z_r = 2*n_r/(t-q)
            elif(i==len(frequencies)-1):
                z_r = n_r / (r-frequencies[i-1])
            else:
                q,t = frequencies[i-1], frequencies[i+1]
                z_r = 2*n_r/(t-q)
            Z_r.append(z_r)
            
        # print(Z_r)
        log_r = [math.log(f) for f in frequencies]
        log_Z_r = [math.log(z_r) for z_r in Z_r]
        
        ### fit line to log-log plot
        model = LinearRegression()
        x = np.array(log_r).reshape(-1, 1)
        y = np.array(log_Z_r)
        model.fit(x, y)

        c, prob = 0.0, None
        if token in self.context[context]:
            c = self.context[context][token]
        
        if c == 0.0:
            prob =  N_r[0] / self.context_cnt[context]
        elif c <= 20.0:
            try:
                c_ = (c+1)*frequency_count[c+1]/frequency_count[c]
                prob = c_ / self.context_cnt[context]
            except:
                prob = self.context[context][token] / self.context_cnt[context]
        else:
            N_c_1 = math.exp(model.predict(np.array([[math.log(c+1)]])).tolist()[0])
            N_c =  math.exp(model.predict(np.array([[math.log(c)]])).tolist()[0])
            c_ = (c+1)*N_c_1 / N_c
            prob = c_ / self.context_cnt[context]
        
        return prob
            
        
                
            
            
        
    def generate_word(self, context):
        
        p = random.random()
        possible_next_words = list(self.vocabulary)
        cur_prob = 0 
        for next_word in possible_next_words:
            cur_prob += self.probability_for_next_word(context, next_word)
            if( p <= cur_prob ):
                return next_word
    
    def generate_sentence(self, max_tokens):
        ### produces till </s>  or at max max_tokens
        context = ["<s>"]*(self.n-1)
        generated_words = []
        for _ in range(max_tokens):
            next_word = self.generate_word(tuple(context))
            if(next_word == "</s>"):
                break
            generated_words.append(next_word)
            if self.n > 1:
                context = context[1:] + [next_word]
            
        return " ".join(generated_words)
    
    def log_prob(self, sentence):
        ### sentence is a list of tokens in the sentence including <s> and </s>
        sentence = sentence[1:] ### remove the first <s>
        log_prob = 0.0
        context = ["<s>"] * (self.n-1) if (self.n > 1) else []
        for word in sentence:
            next_word_prob = self.probability_for_next_word(tuple(context), word)
            if(next_word_prob == 0.0  ):
                log_prob = float("-inf")
                break
            else:
                log_prob += math.log(next_word_prob)
            ### update context
            if self.n > 1 :
                context = context[1:] + [word]
            
    
        return log_prob
                
    
    def perplexity(self, text):
        
        log_prob_sum = 0.0
        token_cnt = 0
        _, tokenized_sentences = preprocess(text)
        for sent in tqdm(tokenized_sentences):
            log_prob = self.log_prob(sent)
            if(log_prob == float("-inf")):
                return float("inf")
            else:
                log_prob_sum += log_prob
            token_cnt += len(sent)-1
        
        return math.exp(-log_prob_sum / token_cnt)
            

if __name__ == "__main__":        
 
    train_books = range(1,6)
    val_books = range(6,7)
    LM = SimpleGoodTuringLM(n=args.n)
    ########### TRAINING ###################
    vocabulary = {"<s>" : 0, "</s>" : 0 , "<unk>" : 0}

    train_tokenized_sentences = []
    for book in train_books:
        train_book = f"./Harry_Potter_Text/Book{book}.txt"
        print(f"{train_book} ....")
        text = read_data(train_book)
        local_voc, tokenized_sentences = preprocess(text)
        for token in local_voc.keys():
            if token in vocabulary:
                vocabulary[token] += local_voc[token]
            else:
                vocabulary[token] = local_voc[token]
        train_tokenized_sentences.append(tokenized_sentences)


    print("Training ... ")  
    for tokenized_sentences in train_tokenized_sentences:
        for sent in tokenized_sentences:
            LM.update(sent)
    ###########################################################
    

    ################### TESTING ############ 
    print("TESTING .... ")
    test_book = f"./Harry_Potter_Text/Book7.txt"    
    test_text = read_data(test_book)
    print(LM.perplexity(test_text))    
    ########################################
    

    
    if(args.generate):
        for _ in range(args.generate_cnt):
            print(LM.generate_sentence(10))     
