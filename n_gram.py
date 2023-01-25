import random
import math
from preprocess import preprocess, read_data

def create_n_grams(tokens, n=1):
    ### tokens is a list of tokens of a sentence including <s> and </s>
    tokens = tokens[1:] ### remove the first <s>
    tokens = ["<s>"] * (n-1) + tokens ## appropriate sentence padding depending on the model
    n_grams = [ (tuple([tokens[i-p-1] for p in reversed(range(n-1))]), tokens[i]) for i in range(n-1, len(tokens))]
    return n_grams

class NGramLM():
    
    def __init__(self, n):
        self.n = n
        
        self.context = {}
        self.n_gram_counter = {}  
        self.vocabulary = set(("<s>", "</s>", "<unk>"))
        
    def update(self, sentence):
        
        ### sentence is a list of tokens including <s> and </s>
        for token in sentence:
            self.vocabulary.add(token)
        
        ngrams = create_n_grams(sentence, n=self.n) ### creates a list of ngrams
        for ngram in ngrams:
            context, next_word = ngram
            if(context in self.context):
                self.context[context].append(next_word)
            else:
                self.context[context] = [next_word]
            
            if ngram in self.n_gram_counter:
                self.n_gram_counter[ngram] += 1
            else:
                self.n_gram_counter[ngram] = 1
                
    def add_k_smoothing(self, k=1):
        
        all_context = self.context.keys()
        for context in all_context:
            for word in self.vocabulary:
                for _ in range(k):
                    self.context[context].append(word)
                    if (context, word) in self.n_gram_counter:
                        self.n_gram_counter[(context,word)] += 1
                    else:
                        self.n_gram_counter[(context,word)] = 1
        
                  
    def probability_for_next_word(self, context, token):
        ### probability of token given context
        if token not in self.vocabulary:
            token = "<unk>"
        if context not in self.context:
            return 0.0
        
        try:
            if ((context, token) not in self.n_gram_counter):
                n_gram_count = 0
            else:
                n_gram_count =  self.n_gram_counter[(context, token)]
            context_count = float(len(self.context[context]))
            prob = n_gram_count / context_count
        except:
            prob = 0.0
        
        return prob
    
    def generate_word(self, context):
        
        return random.choices(self.context[context], k=1)[0]
    
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
                
                
            
        
        
 
train_books = range(1,7)
LM = NGramLM(n=1)
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

#### add-k smoothing ###################
# LM.add_k_smoothing(k=1)

################### TESTING ############ 
test_book = f"./Harry_Potter_Text/Book7.txt"    
test_text = read_data(test_book)
test_voc, test_tokenized_sentences = preprocess(test_text)

inf_count = 0 
for sent in test_tokenized_sentences:
    log_prob = LM.log_prob(sent)
    if(log_prob != float("-inf")):
        print(log_prob)
    else:
        inf_count += 1
print(f"{inf_count/len(test_tokenized_sentences)*100}%")
print( LM.generate_sentence(10) )
