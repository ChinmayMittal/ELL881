def read_file(filename):
    ## reads entire filename as a list, each sentence(including \n) is one string in the list
    with open(filename, "r") as file:
        text = file.readlines()
    return text

def process_text(text):
    ### finds sentences from the text file and their corresponding labels,
    ### X is a list of lists and Y is a list of lists
    ### X[0] will be a list with each word in the sentences
    ### Y[0] will be a list with the tag for each word in X[0]
    
    X = []
    Y = []
    sentenceX = []
    sentenceY = []
    for line in text:
        split = line.split(" ")
        if len(split) > 1:
            sentenceX.append(split[0].lower())
            sentenceY.append(split[1].replace("\n", ""))
        else:
            X.append(sentenceX)
            Y.append(sentenceY)
            sentenceX = []
            sentenceY = []
    return X, Y


def max_seq_len(X):
    ### finds the largest sentences in the dataset
    max_len = 0
    for sent in X:
        max_len = max(max_len, len(sent))
    return max_len

def pad_seq(seq, pad_len, pad_value):
    
    if(len(seq) > pad_len):
        return seq[:pad_len]
    else:
        return seq + ((pad_len-len(seq))*[pad_value])
    
    
def find_unique_labels(Y):
    ### Y is a list of lists, each list containts tags
    unique_tags = set()
    for sentence_predictions in Y:
        for tag in sentence_predictions:
            unique_tags.add(tag)
                 
    return unique_tags

def get_scores(predY, trueY):
    
    from sklearn.metrics import f1_score
    trueY_O = [i for i, x in enumerate(trueY) if x == "O"] ## indices where true value is "O"
    ### consider only those indices where true value is not 'O'
    predY = [predY[i] for i in range(len(predY)) if i not in trueY_O]
    trueY = [trueY[i] for i in range(len(trueY)) if i not in trueY_O]

    print("Micro F1 score: ", f1_score(trueY, predY, average="micro"))
    print("Macro F1 score: ", f1_score(trueY, predY, average="macro"))
    print("Average F1 score: ", (f1_score(trueY, predY, average="micro") + f1_score(trueY, predY, average="macro")) / 2)