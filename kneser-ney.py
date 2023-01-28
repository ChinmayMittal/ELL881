import random
import math
import argparse
from preprocess import preprocess, read_data
from n_gram import create_n_grams
from tqdm import tqdm

parser = argparse.ArgumentParser(description='N Gram Language Models ... ')
parser.add_argument("-n", action="store", default=1, type=int)
parser.add_argument("--generate", action="store", default=False, type=bool)
parser.add_argument("--generate_cnt", action="store", default=1, type=int)

args = parser.parse_args()

class KneserNeyLM():
    
    
    def __init__(self, n, d):
        
        self.n = n
        self.d = d
        self.vocabulary = set(("<s>", "</s>", "<unk>"))
        self.continuation_cnt = {} ### takes a string as a tuple of tokens and returns a set of unique single word contexts
        self.context = {} #### doubly index dict context -> next_word -> how many times next_word follows context
        self.context_cnt = {} ### count of how many times context has appeared
        self.lambda_ = {} ### normalizing factors for each context 
        
    def update(self, sentence):
        
        ### sentence is a list of tokens including <s> and </s>
        for token in sentence:
            self.vocabulary.add(token)
            
        ### i need normal counts for n-gram and continuation counts for n-1 gram ........ 1 gram     
        
        ### normal counts for n-grams   
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
            
        for i in range(self.n-1,0,-1):
            ### loops from n-1 to 1 for continuation counts
            ngrams = create_n_grams(sentence, n=i+1) ### for continuation count of i-grams i need to create (i+1)-grams
            for ngram in ngrams:
                context, next_word = ngram
                single_word_context = context[0] ### single word context of continuation string 
                continuation_string = context[1:] + (next_word, ) ### for each string will store possible single word contexts
                if(continuation_string in self.continuation_cnt):
                    self.continuation_cnt[continuation_string].add(single_word_context)
                else:
                    self.continuation_cnt[continuation_string] = set((single_word_context))
      
    
    def find_lambda(self, context):
        ### find normalizing value for context
        if context in self.lambda_: ### memoization
            return self.lambda_[context]
        
        if len(context) == self.n-1:
            #### normal counts
            if context not in self.context_cnt:
                ### this context did not appear in training set
                return 1
                
            denominator = self.context_cnt[context]
            numerator = self.d * (len(self.context[context]))
            self.lambda_[context] = numerator / denominator
            return self.lambda_[context]
        else:
            ### continuation counts
            denominator = 0.0
            numerator = 0.0
            for word in self.vocabulary:
                context_word = context + (word, )
                if(context_word in self.continuation_cnt):
                    denominator += len(self.continuation_cnt[context_word])
                    numerator += 1
            if denominator == 0.0 :
                self.lambda_[context] = 1 ### kind of like backoff first term is missing fall to second term in P_kn
            else:
                self.lambda_[context] = numerator*self.d / denominator
            return self.lambda_[context]
                    
                
                    
    def prob_KN(self, context, token):
        
        if token not in self.vocabulary:
            token = "<unk>"
        
        if(len(context) == 0 ): ### unigram case
            lambda_ = self.find_lambda(context)
            if self.n > 1:
                ### continuation counts
                denominator = 0.0
                for word in self.vocabulary:
                    if((word, ) in self.continuation_cnt):
                        denominator += len(self.continuation_cnt[(word, )])
                        
                numerator = 0.0
                if((token, ) in self.continuation_cnt):
                    numerator = max(len(self.continuation_cnt[(token, )])-self.d, 0.0)
                    
                return ( numerator / denominator) + (lambda_ / len(self.vocabulary))
            else:
                ### normal counts
                denominator = self.context_cnt[context]
                numerator = 0.0
                if(token in self.context[context]):
                    numerator = max((self.context[context][token])-self.d, 0.0)
                    
                return ( numerator / denominator) + (lambda_ / len(self.vocabulary))
            
        
        lambda_ = self.find_lambda(context)
        add_term = lambda_ * self.prob_KN(context[1:], token)
        
        if len(context) == self.n-1 :
            ## normal counts
            if context not in self.context_cnt:
                ### context did not appear in training
                return add_term ## backoff
            numerator = 0.0
            if(token in self.context[context]):
                numerator  = max((self.context[context][token])-self.d, 0.0)
            denominator = self.context_cnt[context]
            
            return ( numerator / denominator )  + add_term
        else:
            ### continuation counts
            denominator = 0.0
            for word in self.vocabulary:
                context_word = context + (word, )
                if(context_word in self.continuation_cnt):
                    denominator += len(self.continuation_cnt[context_word])
            if(denominator == 0.0):
                return add_term ## backoff
            numerator = 0.0 
            context_token = context + (token, )
            if(context_token in self.continuation_cnt):
                numerator = max(len(self.continuation_cnt[context_token])-self.d, 0.0)
            
            return (numerator/denominator) + add_term
                
    def generate_word(self, context):
        
        p = random.random()
        possible_next_words = list(self.vocabulary)
        cur_prob = 0 
        for next_word in possible_next_words:
            cur_prob += self.prob_KN(context, next_word)
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
            next_word_prob = self.prob_KN(tuple(context), word)
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
    LM = KneserNeyLM(n=args.n, d=0.75)
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
    for tokenized_sentences in tqdm(train_tokenized_sentences):
        for sent in tokenized_sentences:
            LM.update(sent)

    ################### TESTING ############ 
    print("TESTING .... ")
    test_book = f"./Harry_Potter_Text/Book7.txt"    
    test_text = read_data(test_book)
    print(LM.perplexity(test_text))    
    ########################################
    
    if(args.generate):
        for _ in range(args.generate_cnt):
            print(LM.generate_sentence(10))         
    
        
    
    