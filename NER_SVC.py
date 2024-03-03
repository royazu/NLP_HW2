from gensim.models import Word2Vec
import re
import numpy as np
from sklearn.svm import SVC
import string
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler




STOP_WORDS = ['a', 'an', 'the', 'in', 'on', 'at', 'to', 'of', 'by', 'and', 'or', 'for', 'with', 'is', 'are', 'was', 'were', 'am', 'do', 'does', 'did', 'has', 'have', 'had', 'can', 'could', 'will', 'shall', 'may', 'might', 'must', 'should']
PUNCTUATION = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
WORD_2_VEC_PATH = 'word2vec-google-news-300'
GLOVE_PATH = 'glove-twitter-200'

# check if a word contains punctuation
def has_punctuation(word):
    for char in word:
        if char in string.punctuation:
            return True
    return False

def load_data(filename):
    with open(filename,'r',encoding='utf-8') as f:
        lines = f.readlines()
    
    scentences, scentence = [], []
    labels, scentence_labels = [], []
    for line in lines:
        if line != '\n' and line != '':
            word = line.strip().split('\t')
            if word[0].lower() in STOP_WORDS or word == [''] or has_punctuation(word[0]) or len(word) < 2:
                continue
            label = word[1]
            scentence.append(word[0])
            scentence_labels.append(0 if label == 'O' else 1)
        else:
            scentences.append(scentence)
            labels.append(scentence_labels)
            scentence, scentence_labels = [], []
    return scentences, labels

# Load training and development data
train_scentences, train_labels = load_data('/home/student/.virtualenvs/NLP_HW2/data/train.tagged')
dev_scentences, dev_labels = load_data('/home/student/.virtualenvs/NLP_HW2/data/dev.tagged')

# Load test data
test_scentences = load_data('/home/student/.virtualenvs/NLP_HW2/data/test.untagged')

# Convert lists to numpy arrays for compatibility with sklearn
train_scentences = np.array(train_scentences)
train_labels = np.array(train_labels)
mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform(train_labels).flatten()
dev_scentences = np.array(dev_scentences)
dev_labels = np.array(dev_labels)
dev_labels = mlb.fit_transform(dev_labels).flatten()

test_scentences = np.array(test_scentences)

# Train Word2Vec model
model = Word2Vec(sentences=train_scentences, window=5, size=100, workers=4, min_count=1)
model.train(train_scentences, total_examples=len(train_scentences), epochs=10)
model.save('word2vec.model')

# Extract features from the sentences
def extract_features(scentences, model):
    features = []
    for sen in scentences:
        representation = []
        for word in sen:
            word = re.sub(r'\W+', '', word.lower())
            if word not in model.wv.vocab.keys():
                continue
            vec = model[word]
            representation.append(vec)
        if len(representation) == 0:
            continue
        features.append(np.mean(representation, axis=0))
    return features
    
# Create and train the model
svc_model = SVC(kernel='poly', C=10, gamma=0.1, degree=3, coef0=1, probability=True, class_weight='balanced')
scalar = StandardScaler()
train_features = extract_features(train_scentences, model)
train_features = scalar.fit_transform(train_features)
if len(train_features) < len(train_labels):
    train_labels = train_labels[:len(train_features)]
svc_model.fit(train_features, train_labels)
dev_features = extract_features(dev_scentences, model)
dev_features = scalar.fit_transform(dev_features)

# Predict the labels
dev_predictions = svc_model.predict(dev_features)

# Print the F1 score
print(f'f1 score: {f1_score(dev_labels[:len(dev_predictions)], dev_predictions)}')
