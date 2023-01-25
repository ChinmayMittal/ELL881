import random
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
        
    def update(self, sentence):
        
        ### sentence is a list of tokens including <s> and </s>
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
                
                
    def probability_for_next_word(self, context, token):
        ### probability of token given context
        try:
            n_gram_count = self.n_gram_counter[(context, token)]
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
    
    def log_prob()
            
        
        
 
books = range(1,7)
LM = NGramLM(n=4)
for book in books:
    text = read_data(f"./Harry_Potter_Text/Book{book}.txt")
    tokenized_sentences = preprocess(text)
    for sent in tokenized_sentences:
        LM.update(sent)
        
print( LM.generate_sentence(10) )
