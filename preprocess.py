import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download("punkt")

def read_data(datapath, lowercase=True, remove_page_demaractations=True):

    with open(datapath) as f:
        lines = f.readlines() ### reads all lines
    
    lines = [line for line in lines if line != "\n" ]   ### removes lines which are just \n
    
    if lowercase:
        lines = [line.lower() for line in lines] 
    
    if remove_page_demaractations:
        
        page_demaracations = ["page | ", "p a g e | "]
        new_lines = []
        
        for line in lines:
            demaracation_found = False
            for page_demaracation in page_demaracations:
                if(len(line) >= len(page_demaracation) and line[:len(page_demaracation)] == page_demaracation):
                    demaracation_found = True
                    
            if not demaracation_found:
                new_lines.append(line)
                
        lines = new_lines
        
        
    
    text = "".join(lines) 
    text.replace("\n", " ")
    text = re.sub(r"[^a-z0-9.?! ']", "", text)
    
    return text    

def preprocess(text):

    
    ### create sentences
    sentences = sent_tokenize(text) ### list of sentences , each sentence is a string
    ### add sentence delimiters and tokenize sentences
    tokenized_sentences = []
    total_number_tokens = 0
    vocabulary  = set()
    for sent in sentences:
        tokenized_sentence = ["<s>"] + word_tokenize(sent) + ["</s>"]
        tokenized_sentences.append(tokenized_sentence)
        total_number_tokens += len(tokenized_sentence)
        vocabulary.update(tokenized_sentence)
        

    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Number of Tokens: {total_number_tokens}")
    
    # # print(vocabulary)
    
    return tokenized_sentences ## list of lists, where each list has several tokens in a sentence including the start of senetence and end of sentence

# text = read_data("./Harry_Potter_Text/Book1.txt") ### returns an entire string with some preprocessing
# print(preprocess(text)[:5])