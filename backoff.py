import random
import math
from preprocess import preprocess, read_data
from n_gram import create_n_grams
from tqdm import tqdm

class StupidBackOffLM():
    
    def __init__(self, n, alpha=0.4):
        
        self.n = n
        self.alpha = alpha
        self.context = {} #### doubly index dict context -> next_word -> how many times next_word follows context
        self.context_cnt = {} ### count of how many times context has appeared
        self.vocabulary = set(("<s>", "</s>", "<unk>"))
        self.add_k_smoothing_on = False
        self.raw_score_sum = {}
        self.k = 0.0
        
    def update(self, sentence):
        
        ### sentence is a list of tokens including <s> and </s>
        for token in sentence:
            self.vocabulary.add(token)    

        for i in range(1, self.n+1):
            #### populate all type of ngram counts
            ngrams = create_n_grams(sentence, n=i) ### creates a list of ngrams
            for ngram in ngrams:
                context, next_word = ngram
                if( context not in self.context):
                    self.context[context] = {}
                    self.context_cnt[context] = 0
                if( next_word not in self.context[context]):
                    self.context[context][next_word] = 0
                
                self.context[context][next_word] += 1
                self.context_cnt[context] += 1
                
    def add_k_smoothing(self, k=1):
        #### will be used for unigram
        self.add_k_smoothing_on = True
        self.k = k
     
    def probability_for_next_word(self, context, token):
        
        if context in self.raw_score_sum:
            return ( 1 / self.raw_score_sum[context] ) * self.probability_for_next_word_helper(context, token)
        all_possible_tokens = self.vocabulary
        total_score_sum = 0.0
        for next_word in all_possible_tokens:
            total_score_sum += self.probability_for_next_word_helper(context, next_word)
        self.raw_score_sum[context] = total_score_sum
        return self.probability_for_next_word_helper(context, token)
                   
    def probability_for_next_word_helper(self, context, token):
        
        if token not in self.vocabulary:
            token = "<unk>"
        if len(context) == 0: ### recursive base case (unigram)
            if token in self.context[context]:
                if not self.add_k_smoothing_on:
                    return self.context[context][token] / self.context_cnt[context]
                else:
                    return ( self.context[context][token] + self.k ) / ( self.context_cnt[context] + self.k * len(self.vocabulary) ) 
            else:
                if not self.add_k_smoothing_on:
                    return 0
                else :
                    return (self.k) / (self.context_cnt[context] + self.k * len(self.vocabulary))
        if context not in self.context:
            return self.alpha*self.probability_for_next_word_helper(context[1:], token)
        
        if token not in self.context[context]:
            return self.alpha * self.probability_for_next_word_helper(context[1: ], token)
        else:
            n_gram_count = self.context[context][token]
            context_cnt = self.context_cnt[context]
            prob = n_gram_count / context_cnt
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
 
    train_books = range(1,7)
    n = 3
    LM = StupidBackOffLM(n=n)
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

    ### write vocabulary to file
    vocab_tokens = list(vocabulary.keys())
    vocab_tokens.sort( reverse=True, key=vocabulary.__getitem__ )
    with open("vocab.txt", "w") as f:
        for token in vocab_tokens:
            f.write(f"{token}:{vocabulary[token]}\n")
            
    for tokenized_sentences in train_tokenized_sentences:
        for sent in tokenized_sentences:
            LM.update(sent)
    ########### ADD-K Smoothing for base case unigram ############
    LM.add_k_smoothing(k=0.1)

    ################### TESTING ############ 
    print("TESTING .... ")
    test_book = f"./Harry_Potter_Text/Book7.txt"    
    test_text = read_data(test_book)


    print(LM.perplexity(test_text))
    print( LM.generate_sentence(10) )