import sys

import nltk
from nltk.corpus import brown
import numpy
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
# Load the Brown corpus with Universal Dependencies tags
# proportion is a float
# Returns a tuple of lists (sents, tags)
def load_training_corpus(proportion=1.0):
    brown_sentences = brown.tagged_sents(tagset='universal')
    num_used = int(proportion * len(brown_sentences))

    corpus_sents, corpus_tags = [None] * num_used, [None] * num_used
    for i in range(num_used):
        corpus_sents[i], corpus_tags[i] = zip(*brown_sentences[i])
    return (corpus_sents, corpus_tags)


# Generate word n-gram features
# words is a list of strings
# i is an int
# Returns a list of strings
def get_ngram_features(words, i):
    n = len(words)
    if i < 0 or i > n - 1:
        raise IndexError
    features = []

    if n < 2: # n = 1 or n = 0
        features.append('prevbigram-<s>')
        features.append('nextbigram-</s>')
        features.append('prevskip-<s>')
        features.append('nextskip-</s>')
        features.append('prevtrigram-<s>-<s>')
        features.append('nexttrigram-</s>-</s>')
        if n == 1: # i = 0
            features.append('centertrigram-<s>-' + words[i] + '-</s>')
        else: # empty string (this should be unreachable...)
            features.append('centertrigram-<s>-</s>-</s>')
    elif n < 3: # n = 2, check if its the first or second word
        if i == 0:
            features.append('prevbigram-<s>')
            features.append('nextbigram-' + words[i + 1])
        else: # i = 1
            features.append('prevbigram-' + words[i - 1])
            features.append('nextbigram-</s>')
        features.append('prevskip-<s>')
        features.append('nextskip-</s>')
        if i == 0:
            features.append('prevtrigram-<s>-<s>')
            features.append('nexttrigram-' + words[i + 1] + '-</s>')
            features.append('centertrigram-<s>-' + words[i] + '-' + words[i + 1])
        else: # i = 1
            features.append('prevtrigram-<s>-' + words[i - 1])
            features.append('nexttrigram-</s>-</s>')
            features.append('centertrigram-' + words[i - 1] + '-' + words[i] + '-</s>')
    else: # n >= 3, check if its the first, second, second last, or last word
        if i == 0:
            features.append('prevbigram-<s>')
        else:
            features.append('prevbigram-' + words[i - 1])
            
        if i < n - 1:            
            features.append('nextbigram-' + words[i + 1])
        else:
            features.append('nextbigram-</s>')

        if i < 2:
            features.append('prevskip-<s>')
        else:
            features.append('prevskip-' + words[i - 2])

        if i < n - 2:
            features.append('nextskip-' + words[i + 2])
        else:
            features.append('nextskip-</s>')

        if i == 0:
            features.append('prevtrigram-<s>-<s>')
        elif i == 1:
            features.append('prevtrigram-<s>-' + words[i - 1])
        else:
            features.append('prevtrigram-' + words[i - 2] + '-' + words[i - 1])
            
        if i == n - 1:
            features.append('nexttrigram-</s>-</s>')
        elif i == n - 2:
            features.append('nexttrigram-' + words[i + 1] + '-</s>')
        else:
            features.append('nexttrigram-' + words[i + 1] + '-' + words[i + 2])

        if i == 0:
            features.append('centertrigram-<s>-' + words[i] + '-' + words[i + 1])
        elif i == n - 1:
            features.append('centertrigram-' + words[i - 1] + '-' + words[i] + '-</s>')
        else:
            features.append('centertrigram-' + words[i - 1] + '-' + words[i] + '-' + words[i + 1])
    return features


# Generate word-based features
# word is a string
# returns a list of strings
def get_word_features(word):
    n = len(word)
    features = []
    if n == 0:
        return features
    
    features.append('word-' + word)

    if word[0].isupper():
        features.append('capital')

    if word.isupper() and word.isalpha():
        features.append('allcaps')

    shape = ''
    for c in word:
        if c.isupper():
            shape += 'X'
        elif c.isalpha():
            shape += 'x'
        elif c.isdigit():
            shape += 'd'
        else:
            shape += c
    features.append('wordshape-' + shape)

    short_shape = shape[0]
    for i in range(1, len(shape)):
        if shape[i] != shape[i - 1]:
            short_shape += shape[i]
    features.append('short-wordshape-' + short_shape)

    if any(c.isdigit() for c in word):
        features.append('number')

    if '-' in word:
        features.append('hyphen')

    features.append('prefix1-' + word[0])
    if n > 1:
        features.append('prefix2-' + word[:2])
    if n > 2:
        features.append('prefix3-' + word[:3])
    if n > 3:
        features.append('prefix4-' + word[:4])

    features.append('suffix1-' + word[-1])
    if n > 1:
        features.append('suffix2-' + word[-2:])
    if n > 2:
        features.append('suffix3-' + word[-3:])
    if n > 3:
        features.append('suffix4-' + word[-4:])
    return features
    

# Wrapper function for get_ngram_features and get_word_features
# words is a list of strings
# i is an int
# prevtag is a string
# Returns a list of strings
def get_features(words, i, prevtag):
    features = get_ngram_features(words, i) + get_word_features(words[i])
    features.append('tagbigram-' + prevtag)
    for i in range(len(features)):
        if features[i].startswith('wordshape-') or features[i].startswith('short-wordshape-'):
            continue
        features[i] = features[i].lower()
    return features

# Remove features that occur fewer than a given threshold number of time
# corpus_features is a list of lists, where each sublist corresponds to a sentence and has elements that are lists of strings (feature names)
# threshold is an int
# Returns a tuple (corpus_features, common_features)
def remove_rare_features(corpus_features, threshold=5):
    common_features = []
    feature_counts = {}

    for sentence in corpus_features:
        for feature_list in sentence:
            for feature in feature_list:
                if feature in feature_counts:
                    feature_counts[feature] += 1
                else:
                    feature_counts[feature] = 1
    common_features = [feature for feature, count in feature_counts.items() if count >= threshold]
    new_corpus_features = []
    for sentence in corpus_features:
        new_sentence = []
        for feature_list in sentence:
            new_feature_list = [feature for feature in feature_list if feature in common_features]
            new_sentence.append(new_feature_list)
        new_corpus_features.append(new_sentence)
    return (new_corpus_features, common_features)



# Build feature and tag dictionaries
# common_features is a set of strings
# corpus_tags is a list of lists of strings (tags)
# Returns a tuple (feature_dict, tag_dict)
def get_feature_and_label_dictionaries(common_features, corpus_tags):
    pos = 0
    feature_dict = {}
    for feature in common_features:
        if not feature in feature_dict.keys():
            feature_dict[feature] = pos
            pos += 1

    pos = 0
    tag_dict = {}
    for tag_list in corpus_tags:
        for tag in tag_list:
            if not tag in tag_dict.keys():
                tag_dict[tag] = pos
                pos += 1
            
    return (feature_dict, tag_dict)

# Build the label vector Y
# corpus_tags is a list of lists of strings (tags)
# tag_dict is a dictionary {string: int}
# Returns a Numpy array
def build_Y(corpus_tags, tag_dict):
    y = []
    for tag_list in corpus_tags:
        for tag in tag_list:
            y.append(tag_dict[tag])
    return numpy.array(y)

# Build a sparse input matrix X
# corpus_features is a list of lists, where each sublist corresponds to a sentence and has elements that are lists of strings (feature names)
# feature_dict is a dictionary {string: int}
# Returns a Scipy.sparse csr_matrix
def build_X(corpus_features, feature_dict):
    rows, cols, values = [], [], []
    num_examples = 0
    for i in range(len(corpus_features)): # sentence
        for j in range(len(corpus_features[i])): # feature list
            for k in range(len(corpus_features[i][j])): # feature
                if corpus_features[i][j][k] in feature_dict:
                    rows.append(num_examples) # word/feature list is the training example
                    cols.append(feature_dict[corpus_features[i][j][k]])
                    values.append(1)
            num_examples += 1
    rows, cols, values = numpy.array(rows), numpy.array(cols), numpy.array(values)
    return csr_matrix((values, (rows, cols)), shape=(num_examples, len(feature_dict)))


# Train an MEMM tagger on the Brown corpus
# proportion is a float
# Returns a tuple (model, feature_dict, tag_dict)
def train(proportion=1.0):
    corpus_sents, corpus_tags = load_training_corpus(proportion)
    corpus_features = []
    for i in range(len(corpus_sents)):
        sentence_features = []
        for j in range(len(corpus_sents[i])):
            if j == 0:
                features = get_features(corpus_sents[i], j, '<S>')
            else:
                features = get_features(corpus_sents[i], j, corpus_tags[i][j - 1])
            sentence_features.append(features)
        corpus_features.append(sentence_features)
    corpus_features, common_features = remove_rare_features(corpus_features)
    feature_dict, tag_dict = get_feature_and_label_dictionaries(common_features, corpus_tags)
    X = build_X(corpus_features, feature_dict)
    Y = build_Y(corpus_tags, tag_dict)
    lr = LogisticRegression(class_weight='balanced', solver='saga', multi_class='multinomial')
    lr.fit(X, Y)
    return (lr, feature_dict, tag_dict)


# Load the test set
# corpus_path is a string
# Returns a list of lists of strings (words)
def load_test_corpus(corpus_path):
    with open(corpus_path) as inf:
        lines = [line.strip().split() for line in inf]
    return [line for line in lines if len(line) > 0]


# Predict tags for a test sentence
# test_sent is a list containing a single list of strings
# model is a trained LogisticRegression
# feature_dict is a dictionary {string: int}
# reverse_tag_dict is a dictionary {int: string}
# Returns a tuple (Y_start, Y_pred)
def get_predictions(test_sent, model, feature_dict, reverse_tag_dict):
    n = len(test_sent)
    t = len(reverse_tag_dict)
    Y_pred = numpy.empty((n - 1, t, t))
    for i in range(1, n):
        features = []
        for _, tag in reverse_tag_dict.items():
            features.append(get_features(test_sent, i, tag))
        x = build_X([features], feature_dict)
        probs = model.predict_log_proba(x)
        Y_pred[i - 1] = probs
    features = get_features(test_sent, 0, '<S>')
    x = build_X([[features]], feature_dict)
    Y_start = model.predict_log_proba(x)
    return (Y_start, Y_pred)


# Perform Viterbi decoding using predicted log probabilities
# Y_start is a Numpy array of size (1, T)
# Y_pred is a Numpy array of size (n-1, T, T)
# Returns a list of strings (tags)
def viterbi(Y_start, Y_pred):
    n = Y_pred.shape[0] + 1
    t = Y_pred.shape[1]
    V = numpy.empty((n, t))
    BP = numpy.empty((n, t))
    V[0] = Y_start
    for i in range(1, n):
        for j in range(t):
            max_tag, max_prob = 0, -numpy.inf
            for k in range(t):
                prob = V[i - 1, k] + Y_pred[i - 1, k, j]
                if prob > max_prob:
                    max_tag = k
                    max_prob = prob
            V[i, j] = max_prob
            BP[i, j] = max_tag
    best_tags = []
    pred_tag = numpy.argmax(V[n - 1])
    best_tags.append(int(pred_tag))
    for i in range(n - 1, 0, -1):
        pred_tag = BP[i, int(pred_tag)]
        best_tags.append(int(pred_tag))
    best_tags.reverse()
    return best_tags


# Predict tags for a test corpus using a trained model
# corpus_path is a string
# model is a trained LogisticRegression
# feature_dict is a dictionary {string: int}
# tag_dict is a dictionary {string: int}
# Returns a list of lists of strings (tags)
def predict(corpus_path, model, feature_dict, tag_dict):
    corpus = load_test_corpus(corpus_path)
    reverse_tag_dict = {v: k for k, v in tag_dict.items()}
    corpus_tags = []
    for sentence in corpus:
        Y_start, Y_pred = get_predictions(sentence, model, feature_dict, reverse_tag_dict)
        tags = viterbi(Y_start, Y_pred)
        tags = [reverse_tag_dict[tag] for tag in tags]
        corpus_tags.append(tags)
    return corpus_tags


def main(args):
    model, feature_dict, tag_dict = train(0.25)

    predictions = predict('test.txt', model, feature_dict, tag_dict)
    for test_sent in predictions:
        print(test_sent)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
