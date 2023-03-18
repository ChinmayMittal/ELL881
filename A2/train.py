import argparse
import yaml
from dataset import TextDataset, get_vocab_and_embeddings, get_label_vocab, get_dataloader
from model import NamedEntityRecgNet
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

def train(config):
    
    print(config)
    train_dataset = TextDataset(file=config["train_path"])
    test_dataset = TextDataset(file=config["test_path"])
    text_vocab, word_embeddings = get_vocab_and_embeddings(dim=config["embed_dim"])
    label_vocab = get_label_vocab(train_dataset)
    
    train_dataloader = get_dataloader(train_dataset, batch_size=config["train_batch_size"], text_vocab=text_vocab, label_vocab=label_vocab, shuffle=config["train_shuffle"])
    test_dataloader = get_dataloader(test_dataset, batch_size=config["test_batch_size"], text_vocab=text_vocab, label_vocab=label_vocab)
    number_of_labels = len(label_vocab)
    
    print(f"Number of Training Labels: {number_of_labels}")
    print(f"Number of Training Sentences: {len(train_dataset)}")
    
    model = NamedEntityRecgNet(num_labels=number_of_labels, embeddings=word_embeddings, input_dim=config["embed_dim"], hidden_dim=config["hidden_dim"], embeddings_freeze=config["freeze_embeddings"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    optimizer = Adam(model.parameters(), lr = float(config["learning_rate"]))
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
    
    writer = SummaryWriter("runs/")
    running_loss = 0 
    
    for epoch in range(1,config["num_epochs"]+1):
        
        print(f"Starting Epoch Number: {epoch}/{config['num_epochs']}")
        model.train()
        
        for batch_idx, (sentences, labels) in tqdm(enumerate(train_dataloader)):
            
            optimizer.zero_grad()
            pred = model(sentences)
            N, T, C = pred.shape ## BATCH * TIME_STEPS * N_CLASSES
            pred_class = torch.argmax(pred, dim=2) ## N * T

            ### pred   N * T * C => NT * C | labels N*T => NT
            loss = criterion(pred.view(-1, C), labels.view(-1,))
            
            loss.backward()
            optimizer.step()
            
            loss = loss.item()
            
            padding_mask = (labels > -1)
            
            non_pad_labels = labels[padding_mask]
            accuracy = (((pred_class[padding_mask] == non_pad_labels).sum()) / non_pad_labels.shape[0]).item()
            
            if(running_loss == 0):
                running_loss = loss
            else:
                running_loss = 0.99*running_loss + 0.01*loss
            
            writer.add_scalar('training loss', loss, (epoch-1)*len(train_dataloader) + batch_idx )
            writer.add_scalar("training accuracy", accuracy, (epoch-1)*len(train_dataloader) + batch_idx )
            
            if(batch_idx%10 == 0):
                print(f"\rEpoch:{epoch} | Batch Idx: {batch_idx} | Running Loss: {running_loss} | Accuracy: {accuracy*100:.2f} %")
                
        print(f"End of Epoch ... Testing ")
        model.eval()
        
        with torch.no_grad():
            
            total_correct, total_samples = 0, 0
            for batch_idx, (sentences, labels) in enumerate(test_dataloader):
                
                pred = model(sentences)
                N, T, C = pred.shape ## BATCH * TIME_STEPS * N_CLASSES
                pred_class = torch.argmax(pred, dim=2) ## N * T
                
                padding_mask = (labels > -1) ### will also mask labels in test which were not present in the train set
                
                non_pad_labels = labels[padding_mask]
                total_correct += ((pred_class[padding_mask] == non_pad_labels).sum()).item()
                total_samples += non_pad_labels.shape[0]
                
            accuracy = total_correct / total_samples
            writer.add_scalar("testing accuracy", accuracy, epoch-1)
                
                
                
                
                
        
    print("Finished Training ... ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser() ## for command line argument for config
    parser.add_argument("--config", type=str, required=True, help="path to the yaml config file for training")
    args = parser.parse_args()

    ## load the yaml config as a dict
    with open(args.config, "r") as buffer:
        config = yaml.safe_load(buffer)
    
    ### train using the config specifications
    train(config)
    