import torch
import torch.nn as nn
import gensim
import gensim.downloader as api
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import string
import numpy as np
import re
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
import pandas as pd
from torch.optim import Adam
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
from sklearn.metrics import f1_score
import string

GLOVE_MODEL = KeyedVectors.load('glove.model')
PUNCTUATION = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
EMPTY_LINE = ['\n', '\t', ' ', '', '\r']

# Create a translation table that maps every punctuation character to None
translator = str.maketrans('', '', string.punctuation)


def load_data(filename):
    with open(filename,'r',encoding='utf-8') as f:
        lines = f.readlines()

    sentences, all_labels = [], []
    sentence, labels = [], []
    for line in lines:
        if line != '\n' and line != '':
            word = line.strip().split('\t')
            if word == [''] or len(word) < 2:
                continue
            # Remove punctuation from the word
            word[0] = word[0].translate(translator)
            label = word[1]
            sentence.append(word[0])
            labels.append(0 if label == 'O' else 1)
        else:  # end of a sentence
            if any(labels):  # if there is at least one non-zero label
                sentences.append(sentence)
                all_labels.append(labels)
            sentence, labels = [], []  # reset for the next sentence
    # handle the last sentence if it didn't end with a newline
    if sentence and any(labels):
        sentences.append(sentence)
        all_labels.append(labels)
    return sentences, all_labels

# Define the dataset
class SentimentDataSet(Dataset):

    def __init__(self, dataset, model):
        self.dataset = dataset
        self.sentences = self.dataset['sentence'].tolist()
        self.labels = [label for sentence in self.dataset for label in sentence[1]]  # Flatten the labels
        self.model = model
        #flat_labels = [item for sublist in self.labels for item in sublist]
        self.tags_to_idx = {tag: idx for idx, tag in enumerate(sorted(list(set(self.labels))))}
        self.idx_to_tag = {idx: tag for tag, idx in self.tags_to_idx.items()}
        if model == 'w2v':
            model = Word2Vec.load('word2vec.model')
        elif model == 'glove':
            model = api.load("glove-wiki-gigaword-100")
        else:
            raise KeyError(f"{model} is not a supported vector type")
        representation, labels = [], []
        for sen, cur_labels in zip(self.sentences, self.labels):
            cur_rep = []
            for word in sen:
                word = re.sub(r'\W+', '', word.lower()) # get representation for each word
                if word not in model.key_to_index:
                    continue
                vec = model[word]
                cur_rep.append(vec)
            if len(cur_rep) == 0:
                #print(f'Sentence {sen} cannot be represented!')
                continue
            cur_rep = np.stack(cur_rep[0])  # HW TODO: change to token level classification
            representation.append(cur_rep)
            labels.append(cur_labels)
        self.labels = labels
        representation = np.stack(representation)
        self.tokenized_sen = representation
        self.vector_dim = representation.shape[-1]

    def __getitem__(self, item):
        cur_sen = self.tokenized_sen[item]
        cur_sen = torch.FloatTensor(cur_sen).squeeze()
        label = self.labels[item]
        label = self.tags_to_idx[label]
        data = {"input_ids": cur_sen, "labels": label}
        return data

    def __len__(self):
        return len(self.labels)

# Define the neural network model
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size1=512, hidden_size2=256, output_size=2, dropout_rate=0.2):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  # New hidden layer
        self.fc3 = nn.Linear(hidden_size2, output_size)  # Adjusted to take input from second hidden layer
        self.action1 = nn.Tanh()
        self.action2 = nn.Tanh()  # Activation function for the new hidden layer
        self.dropout = nn.Dropout(dropout_rate)  # Added dropout
        weight= torch.tensor([0.05, 0.95], dtype=torch.float)
        self.loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, input_ids, labels=None):
        x = self.fc1(input_ids)
        x = self.action1(x)
        x = self.dropout(x)
        x = self.fc2(x)  # New hidden layer
        x = self.action2(x)  # Activation function for the new hidden layer
        x = self.fc3(x)  # Adjusted to take input from second hidden layer
        if labels is None:
            return x, None
        loss = self.loss(x, labels)
        return x, loss
# Train the model based on the training data and neural network

def train(model, data_sets, optimizer, num_epochs: int, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loaders = {"train": DataLoader(data_sets["train"], batch_size=batch_size, shuffle=True,),
                    "test": DataLoader(data_sets["test"], batch_size=batch_size, shuffle=False)}
    model.to(device)
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            labels, preds = [], []

            for batch in data_loaders[phase]:
                batch_size = 0
                for k, v in batch.items():
                    batch[k] = v.to(device)
                    batch_size = v.shape[0]

                optimizer.zero_grad()
                if phase == 'train':
                    outputs, loss = model(**batch)
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs, loss = model(**batch)
                pred = outputs.argmax(dim=-1).clone().detach().cpu()
                labels += batch['labels'].cpu().view(-1).tolist()
                preds += pred.view(-1).tolist()
                running_loss += loss.item() * batch_size

            epoch_loss = running_loss / len(data_sets[phase])
            epoch_acc = f1_score(labels, preds)

            epoch_acc = round(epoch_acc, 5)

            if phase.title() == "test":
                print(f'{phase.title()} Loss: {epoch_loss:.4e} Accuracy: {epoch_acc}')
            else:
                print(f'{phase.title()} Loss: {epoch_loss:.4e} Accuracy: {epoch_acc}')
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                with open('model.pkl', 'wb') as f:
                    torch.save(model, f)
        print()

    print(f'Best Validation Accuracy: {best_acc:4f}')

def collate_fn(batch):
    # Assuming that each element in "batch" is a tuple (data, label)
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Pad sequences
    data = pad_sequence(data, batch_first=True)

    return data, labels

# Load and preprocess data
model = GLOVE_MODEL
train_sentences, train_labels = load_data('/home/student/.virtualenvs/NLP_HW2/data/train.tagged')
train_dataset = pd.DataFrame({'sentence': train_sentences, 'label': train_labels})
dev_sentences, dev_labels = load_data('/home/student/.virtualenvs/NLP_HW2/data/dev.tagged')
dev_dataset = pd.DataFrame({'sentence': dev_sentences, 'label': dev_labels})

# Initialize the model, criterion, and optimizer

train_dataset = SentimentDataSet(train_dataset, 'glove')
print('created dataset')
dev_dataset = SentimentDataSet(dev_dataset, 'glove')
print('created dataset')
data_sets = {"train": train_dataset, "test": dev_dataset}
input_size = train_dataset.vector_dim
model = FFNN(input_size)
optimizer = Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
train(model, data_sets, optimizer, 10)

"""
# Training the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_features)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(dev_features)
    _, predicted = torch.max(outputs.data, 1)
    f1 = f1_score(dev_labels, predicted.numpy())
    print(f'F1 Score: {f1:.4f}')
    """