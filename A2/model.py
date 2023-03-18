import torch
import torch.nn as nn

class NamedEntityRecgNet(nn.Module):
    
    def __init__(self, num_labels, embeddings, input_dim=300, hidden_dim=512, embeddings_freeze=True):
        ### embeddings is a tensor
        ### num_labels is the number of Named Entity Labels on which the model is trained
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=embeddings_freeze)
        self.LSTM = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, num_layers=1)
        self.linear = nn.Linear(in_features=hidden_dim, out_features=num_labels)
        
    def forward(self, sentences):
        
        sent_embedding = self.embeddings(sentences)
        lstm_out, _ = self.LSTM(sent_embedding)
        pred_logits = self.linear(lstm_out)
        return pred_logits