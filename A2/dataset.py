import torch
import torch.nn as nn
from torchtext.vocab import GloVe, vocab, build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader
from utilities import read_file, process_text, max_seq_len, pad_seq, find_unique_labels
from collections import OrderedDict
from functools import partial

def get_vocab_and_embeddings(cache="embeddings", name='6B', dim=300, unk_index=0):
    
    glove_vectors = GloVe(cache=cache, name=name, dim=dim)
    glove_vocab = vocab(glove_vectors.stoi)
    glove_vocab.insert_token("<UNK>",unk_index)
    glove_vocab.set_default_index(unk_index)

    pretrained_embeddings = glove_vectors.vectors
    ### zero embedding for <UNK>
    pretrained_embeddings = torch.cat((torch.zeros(1,pretrained_embeddings.shape[1]),pretrained_embeddings))
    return glove_vocab, pretrained_embeddings
    
def get_label_vocab(dataset):
    
    unique_tags = find_unique_labels(dataset.Y)
    label_vocab = vocab(OrderedDict([tag,1] for tag in unique_tags))
    label_vocab.set_default_index(-1)
    
    return label_vocab

class TextDataset(Dataset):
    
    def __init__(self, file):
        
        self.file = file
        text = read_file(self.file)
        self.X, self.Y = process_text(text)
        self.max_len = max_seq_len(self.X)

        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        
        sentence = self.X[index]
        labels = self.Y[index]
        
        sentence = pad_seq(sentence, self.max_len, "<PAD>")
        labels = pad_seq(labels, self.max_len, pad_value="<PAD>")
        
        return sentence, labels
        
        
def collate_fn(batch, text_vocab, label_vocab):
    
    sent_list, lab_list = [], []
    for sent, labels in batch:
        text_token_indices = text_vocab(sent)
        label_indices = label_vocab(labels)
        sent_list.append(torch.tensor(text_token_indices, dtype=torch.long))
        lab_list.append(torch.tensor(label_indices, dtype= torch.long))
        
    return torch.stack(sent_list), torch.stack(lab_list)
        
    
def get_dataloader(dataset, batch_size, text_vocab, label_vocab, shuffle=False):
    
    return   DataLoader(dataset, batch_size=batch_size, collate_fn=partial(collate_fn, text_vocab=text_vocab, label_vocab=label_vocab), shuffle=shuffle)

        

### will return a tuple of sentences and labels
# sent, lab = next(iter(dataloader))
# print(sent[0]) ### 1-D tensor of sequence length, each item being index for lookup in embedding table
# print(lab[0]) ### 1-D tensor of sequence length, each item being the class number, -1 for timestep to be ignored
