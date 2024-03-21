import torch
import torch.nn as nn
import gensim.downloader as api
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import re
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from sklearn.metrics import f1_score
import string


# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a translation table that maps every punctuation character to None
translator = str.maketrans('', '', string.punctuation)

# Load the word2vec model
word2vec = api.load('glove-twitter-200')

# List of punctuation marks
PUNCTUATION = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
EOS = ['','\n','\t','\r','\v','\f']
STOP_WORDS = ['a', 'an', 'the', 'in', 'on', 'at', 'to', 'of', 'by', 'and', 'or', 'for', 'with', 'is', 'are', 'was', 'were', 'am', 'do', 'does', 'did', 'has', 'have', 'had', 'can', 'could', 'will', 'shall', 'may', 'might', 'must', 'should']

# check if a word contains punctuation
def has_punctuation(word):
    if word in PUNCTUATION:
        return '<PUNCT>'
    return word

#check if a line is url
def is_url(word):
    if 'http' in word:
        return '<URL>'
    return word

def load_data(filename):
    with open(filename,'r',encoding='utf-8') as f:
        lines = f.readlines()
    scentence_rep = {}
    sentences, all_labels = [], []
    sentence, labels = [], []
    for line in lines:
        if line not in EOS:
            word = line.strip().split('\t')
            if word == [''] or len(word) < 2:
                continue
            label = 0 if word[1] == 'O' else 1
            word[0] = word[0].translate(translator)
            word[0] = is_url(word[0])
            word[0] = has_punctuation(word[0])
            sentence.append(word[0].lower())
            labels.append(label)
        else:  # end of a sentence
            if any(labels):  # if there is at least one non-zero label
                sentences.append(sentence)
                all_labels.append(labels)
                scentence_rep[' '.join(sentence)] = labels
            sentence, labels = [], []  # reset for the next sentence
    # handle the last sentence if it didn't end with a newline
    if sentence and any(labels):
        sentences.append(sentence)
        all_labels.append(labels)
        scentence_rep[' '.join(sentence)] = labels
    return sentences, all_labels, scentence_rep

def load_test_data(filename):
    with open(filename,'r',encoding='utf-8') as f:
        lines = f.readlines()
        scentence, scentences = [],[]
        og_scentence,og_scentences = [], []
        for line in lines:
            if line not in EOS:
                og_scentence.append(line.strip())
                word = line.strip()
                word = word.translate(translator)
                word = is_url(word)
                word = has_punctuation(word)
                scentence.append(word.lower())
            else:
                og_scentences.append(og_scentence)
                scentences.append(scentence)
                scentence,og_scentence = [],[]
    return scentences, og_scentences      

# Find the maximum length of a sentence
def max_length(scentences):
    max_len = 0
    for scentence in scentences:
        if len(scentence) > max_len:
            max_len = len(scentence)
    return max_len

# Change all the scentences to the same length
def pad_sentence(s, max_len):
    return s + [''] * (max_len - len(s))





# Load training and development data
scentence_rep = {}
train_scentences, train_labels, scentence_rep['train']= load_data('/home/student/.virtualenvs/NLP_HW2/data/train.tagged')
dev_scentences, dev_labels,scentence_rep['dev']= load_data('/home/student/.virtualenvs/NLP_HW2/data/dev.tagged')
test_scentences, test_og = load_test_data('/home/student/.virtualenvs/NLP_HW2/data/test.untagged')
train_and_dev_scentences = train_scentences + dev_scentences
train_and_dev_labels = train_labels + dev_labels

def padding(scentence_rep, max_len):
    padded_scentence_rep = []
    for scentence in scentence_rep:
        padded_scentence = []
        for word in scentence:
            padded_scentence.append(torch.tensor(word, device=device))
        padded_scentence_rep.append(pad_sequence(padded_scentence, batch_first=True))
    return padded_scentence_rep

class NERDataset(Dataset):
    def __init__(self, scentences,labels, word2vec, device):
        self.scentence_rep = scentence_rep
        self.word2vec = word2vec
        self.device = device
        self.scentences = scentences
        if labels is not None:
            self.labels = [torch.tensor(label, device=device,dtype=torch.long) for label in labels]
        else:
            self.labels = None
        self.features = []
        for scentence in self.scentences:
            feature = []
            for word in scentence:
                word = re.sub(r'\W+', '', word.lower())
                if word not in self.word2vec.key_to_index:
                    vec = np.zeros(self.word2vec.vector_size)
                    feature.append(vec)
                    continue
                vec = self.word2vec[word]
                feature.append(vec)
            self.features.append(torch.tensor(feature, device=self.device, dtype=torch.float32))
        self.features = padding(self.features, min(max_length(self.scentences), 100))


    def __len__(self):
        return len(self.scentences)
    
    def __getitem__(self, idx):
        return self.scentences[idx], self.labels[idx]


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,bidirectional=True)
        self.h1 = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.Tanh(),
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.weight = torch.tensor([0.15, 0.85], device=device)
        self.loss = nn.CrossEntropyLoss(weight=self.weight)
        self.device = device
        
    def forward(self, x, labels=None):
        out, _ = self.lstm(x)
        out = self.h1(out)
        out = self.fc(out)
        if labels is None:
            return out, None
        loss = self.loss(out.view(-1, 2), labels.view(-1)[:len(out.view(-1, 2))])
        return out, loss

# Create the dataset
train_dataset = NERDataset(train_and_dev_scentences,train_and_dev_labels, word2vec, device)
dev_dataset = NERDataset(test_scentences,None, word2vec, device)

# Create the dataloaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=8, shuffle=False)

# Define the model
input_size = word2vec.vector_size
hidden_size = 256
num_layers = 4
output_size = 2
model = LSTM(input_size, hidden_size, num_layers, output_size, device).to(device)

# Define the optimizer
optimizer = Adam(model.parameters(), lr=0.001)



#Define the Predictor for the labels
def predict(model, data_sets, optimizer, num_epochs: int, batch_size=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            preds = []
            batch = data_sets[phase]
            if phase == 'train':
                for k, v in zip(batch.features, batch.labels):
                    outputs, loss = model(k,v)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                for k in batch.features:
                    with torch.no_grad():
                        if k.shape[0] == 0:
                            continue
                        outputs, _= model(k)
                        preds += [outputs.argmax(dim=1).view(-1).cpu().tolist()]
        print(f"Epoch {epoch + 1}/{num_epochs} - Done!\n")
        

    return preds

# Train the model, and predict the labels
test_predict = predict(model, {"train": train_dataset, "test": dev_dataset}, optimizer, 10, 8)
output_path = '/home/student/.virtualenvs/NLP_HW2/data/comp_205736879_325110773.tagged'
with open(output_path, 'w') as f:
    for scentence, label in zip(test_og, test_predict):
        for word, lab in zip(scentence, label):
            f.write(f"{word}\t{'O' if lab == 0 else 1}\n")
        f.write('\n')