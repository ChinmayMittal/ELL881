import random
import math
import argparse
from preprocess import preprocess, read_data
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='N Gram Language Models ... ')
parser.add_argument("-n", action="store", default=1, type=int)
parser.add_argument("--generate", action="store", default=False, type=bool)
parser.add_argument("--generate_cnt", action="store", default=1, type=int)
parser.add_argument("--smoothing", action="store", default=False, type=bool)
parser.add_argument("--val", action="store", default=False, type=bool)
parser.add_argument("--plot", action="store", default=False, type=bool)

args = parser.parse_args()

def create_n_grams(tokens, n=1):
    ### tokens is a list of tokens of a sentence including <s> and </s>
    tokens = tokens[1:] ### remove the first <s>
    tokens = ["<s>"] * (n-1) + tokens ## appropriate sentence padding depending on the model
    n_grams = [ (tuple([tokens[i-p-1] for p in reversed(range(n-1))]), tokens[i]) for i in range(n-1, len(tokens))]
    return n_grams

class NGramLM():
    
    def __init__(self, n):
        self.n = n
        
        self.context = {} #### doubly index dict context -> next_word -> how many times next_word follows context
        self.context_cnt = {} ### count of how many times context has appeared
        self.vocabulary = set(("<s>", "</s>", "<unk>"))
        self.add_k_smoothing_lazy_update = False
        self.k = 0
        
    def update(self, sentence):
        
        ### sentence is a list of tokens including <s> and </s>
        for token in sentence:
            self.vocabulary.add(token)
        
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
            
                
    def add_k_smoothing(self, k=1, lazy_update=False):
        
        if not lazy_update:
            all_context = self.context.keys()
            for context in all_context:
                self.context_cnt[context] += k*len(self.vocabulary) ### every word in vocabulary is seen k more times
                for word in self.vocabulary:
                    
                    if context not in self.context:
                        self.context[context] = {}
                    if word not in self.context[context]:
                        self.context[context][word] = 0
                    
                    self.context[context][word] += k
        else:
            self.add_k_smoothing_lazy_update = True
            self.k = k
                  
    def probability_for_next_word(self, context, token):
        ### probability of token given context
        if token not in self.vocabulary:
            token = "<unk>"
        if context not in self.context:
            return 1/len(self.vocabulary) ### to ensure probability

        try:
            if ( token not in self.context[context]):
                n_gram_count = 0
            else:
                n_gram_count =  self.context[context][token]
            context_count = self.context_cnt[context]
            if not self.add_k_smoothing_lazy_update:
                prob = n_gram_count / context_count
            else:
                prob = (n_gram_count+self.k) / (context_count + self.k*len(self.vocabulary))
        except:
            prob = 0.0
        
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
        for sent in tokenized_sentences:
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
    LM = NGramLM(n=args.n)
    ########### TRAINING ###################
    vocabulary = {"<s>" : 0, "</s>" : 0 , "<unk>" : 0}

    train_tokenized_sentences = []
    val_text = None
    
    ### load train set
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

    ### load val set ####
    for book in val_books:
        val_book = f"./Harry_Potter_Text/Book{book}.txt"
        print(f"{val_book} ....")
        val_text = read_data(val_book)
    ### end loading val set ##

    ### write vocabulary to file #####
    vocab_tokens = list(vocabulary.keys())
    vocab_tokens.sort( reverse=True, key=vocabulary.__getitem__ )
    with open("vocab.txt", "w") as f:
        for token in vocab_tokens:
            f.write(f"{token}:{vocabulary[token]}\n")
    ###################################       
    
    ### Update Model  ####
    if not args.val:
        for tokenized_sentences in train_tokenized_sentences:
            for sent in tokenized_sentences:
                LM.update(sent)
        if args.smoothing:
            print("Smoothing .... ")
            LM.add_k_smoothing(k=0.1, lazy_update=True) 
    ############################
    
    #### validation for add-k smoothing ###################
    
    if args.val:
        search_space = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 1, 2, 3]
        val_perplexity = [] 
        best_model = None
        best_k = 0
        best_perplexity = float("inf")
        for k in search_space:
            LM = NGramLM(n=args.n)
            for tokenized_sentences in train_tokenized_sentences:
                for sent in tokenized_sentences:
                    LM.update(sent) 
            LM.add_k_smoothing(k=k, lazy_update=True)
            val_perpl = LM.perplexity(val_text)
            val_perplexity.append(val_perpl)
            if val_perpl < best_perplexity:
                best_perplexity = val_perpl
                best_model = LM
                best_k = k
        LM = best_model
        print(best_k)
        if args.plot:
            
            plt.plot(search_space, val_perplexity)
            plt.xlabel("K for Add-K Smoothing")
            plt.ylabel("Validation Set Perplexity")
            plt.title(f"Hyperparameter Tuning N:{args.n}")
            plt.grid(True)
            plt.savefig(f"graph-{args.n}.png")                       

    
 
    #########################################################
    
    ################### TESTING ############ 
    test_book = f"./Harry_Potter_Text/Book7.txt"    
    test_text = read_data(test_book)
    print(LM.perplexity(test_text))
    #########################################################
    
    if(args.generate):
        for _ in range(args.generate_cnt):
            print(LM.generate_sentence(20))
