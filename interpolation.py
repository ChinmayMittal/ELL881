import random
import math
from n_gram import NGramLM
from preprocess import preprocess, read_data


class InterpolationLM():
    
    def __init__(self, n, interpolation_weights = None):
        ### will be an interpolation of n-gram (n-1)-gram ........ 1-gram models
        self.n = n
        self.models = [NGramLM(n=i) for i in range(1,n+1)]
        if(interpolation_weights is None):
            self.interpolation_weights = [1/n for _ in range(1,n+1)]
        else:
            self.interpolation_weights = interpolation_weights
        
        self.vocabulary = set()
    
    def update(self, sentence):

        ### sentence is a list of tokens including <s> and </s>
        for token in sentence:
            self.vocabulary.add(token)
        
        for model in self.models:
            model.update(sentence)
        
    def probability_for_next_word(self, context, token):
        ### context is a tuple of length n-1
        probabilities = [ self.models[i-1].probability_for_next_word( context=context[self.n-i:], token=token) for i in range(1, self.n+1)]  
        prob = 0.0
        for i in range(1,self.n+1):
            prob += probabilities[i-1]*self.interpolation_weights[i-1]
        
        return prob
    
    def generate_word(self, context):
        
        p = random.random()
        possible_next_words = self.vocabulary
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
    train_books = range(1,7)
    LM = InterpolationLM(n=3, interpolation_weights=[0.25, 0.25, 0.5])
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
    print("Trained")
    #### add-k smoothing ###################
    for i in range(1, 2):
        LM.models[i-1].add_k_smoothing(k=0.1)

    ################### TESTING ############ 
    test_book = f"./Harry_Potter_Text/Book7.txt"    
    test_text = read_data(test_book)


    print(LM.perplexity(test_text))
    print( LM.generate_sentence(10) )
    