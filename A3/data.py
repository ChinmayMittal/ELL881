
import pandas as pd
from transformers import BertTokenizerFast

def get_data_label_details(file_path):
    
    df = pd.read_csv(file_path)
    labels = [label.split() for label in df["labels"].values.tolist()]
    unique_labels = set()
    for sent_label in labels:
        [unique_labels.add(token_lb) for token_lb in sent_label]
    
    num_unique_labels = len(unique_labels)
    print(f"Number of Unique Labels: {num_unique_labels}")
    label_to_idx = { label : idx for idx, label in enumerate(sorted(unique_labels))}
    idx_to_label = { idx : label for idx, label in enumerate(sorted(unique_labels))}
    return num_unique_labels, label_to_idx, idx_to_label


    
    