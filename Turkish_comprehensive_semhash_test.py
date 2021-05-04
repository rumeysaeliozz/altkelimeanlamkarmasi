from __future__ import unicode_literals
import sys
import re
import os
from itertools import product
import codecs
import json
import csv
import spacy
import sklearn
import string
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn import model_selection
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2
from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.neighbors.nearest_centroid import NearestCentroid
import math
import random
from tqdm import tqdm
import nltk
import pandas as pd
from nltk.corpus import wordnet
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
# import lemmatizer
import pickle

from typing import List
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM, java
from unidecode import unidecode

ZEMBEREK_PATH = r'C:\Users\golive\Desktop\nlp_yazi\zemberek-all-0.11.1.jar'
startJVM(getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % (ZEMBEREK_PATH))
turkish_morphology = JClass('zemberek.morphology.TurkishMorphology')
morphology = turkish_morphology.createWithDefaults()

# import os
# os.environ['LDFLAGS'] = '-framework CoreFoundation -framework SystemConfiguration'
# !pip3 install spacy
print(sys.path)

nlp = spacy.load('en_core_web_lg')
# print('Running')

try:
    with open('EsAnlamlilar.csv', 'r', encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=',')
        mydict = {key: row for key, *row in reader}

except IOError:
    print("Please generate csv file named EsAnlamlilar")


def get_synonyms(word, number=3):
    synonyms = []
    word = word.lower()
    try:
        synonyms.append(mydict[word][0])
    except:
        pass

    synonyms = list(OrderedDict.fromkeys(synonyms))
    # synonyms = list(OrderedDict.fromkeys(synonyms))
    return synonyms[:number]


print(get_synonyms("search", -1))

# Hyperparameters
benchmark_dataset = ''  # Choose from 'AskUbuntu', 'Chatbot' or 'WebApplication'
oversample = True  # Whether to oversample small classes or not. True in the paper
synonym_extra_samples = True  # Whether to replace words by synonyms in the oversampled samples. True in the paper
augment_extra_samples = False  # Whether to add random spelling mistakes in the oversampled samples. False in the paper
additional_synonyms = 0  # How many extra synonym augmented sentences to add for each sentence. 0 in the paper
additional_augments = 0  # How many extra spelling mistake augmented sentences to add for each sentence. 0 in the paper
mistake_distance = 0  # How far away on the keyboard a mistake can be
VECTORIZER = ""  # which vectorizer to use. choose between "count", "hash", and "tfidf"

RESULT_FILE = "result_file.csv"
METADATA_FILE = "metadata5.csv"
NUMBER_OF_RUNS_PER_SETTING = 10
# stopwords = ["için","bir","mı","mi","mu","mü","ki","iyi","var","gibi","herhangi","bazı","en","midir","misiniz","mıdır","ya","ve"
#                 "mısınız","ile","lütfen","dan","den","tan","ten","te","ta","da","de","a","e","i","ı","teki","veya","ama","olarak","bir şekilde"]
stopwords = ["bir","a","e","i","ı","dan","den","tan","ten","te","ta","da","de","midir","misiniz","mıdır","ya","ve","mısınız"]
# Comprehensive settings testing
# for benchmark_dataset, (oversample, synonym_extra_samples, augment_extra_samples), additional_synonyms, additional_augments, mistake_distance, VECTORIZER in product(['AskUbuntu', 'Chatbot', 'WebApplication'], [(False, False, False),(True, False, False),(True, False, True),(True, True, False),(True, True, True)], [0,4], [0,4], [2.1], ["tfidf", "hash", "count"]):

# Settings from the original paper
for benchmark_dataset, (oversample, synonym_extra_samples,
                        augment_extra_samples), additional_synonyms, additional_augments, mistake_distance, VECTORIZER in product(
        ['AskUbuntu', 'Chatbot', 'WebApplication'], [(True, True, False)], [0], [0], [2.1], ["tfidf"]):

    if benchmark_dataset == "Chatbot":
        intent_dict = {"DepartureTime": 0, "FindConnection": 1}
    elif benchmark_dataset == "AskUbuntu":
        intent_dict = {"Make Update": 0, "Setup Printer": 1, "Shutdown Computer": 2, "Software Recommendation": 3,
                       "None": 4}
    elif benchmark_dataset == "WebApplication":
        intent_dict = {"Download Video": 0, "Change Password": 1, "None": 2, "Export Data": 3, "Sync Accounts": 4,
                       "Filter Spam": 5, "Find Alternative": 6, "Delete Account": 7}

    filename_train = "datasets/KL/" + benchmark_dataset + "/train.csv"
    filename_test = "datasets/KL/" + benchmark_dataset + "/test.csv"

    df = pd.read_csv(filename_test, sep='\t',header = None)

    def read_CSV_datafile(filename):
        X = []
        y = []
        with open(filename, 'r', encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for i, row in enumerate(reader):
                X.append(row[2])
                if benchmark_dataset == 'AskUbuntu':
                    y.append(intent_dict[row[1]])
                elif benchmark_dataset == 'Chatbot':
                    y.append(intent_dict[row[1]])
                else:
                    y.append(intent_dict[row[1]])
        return X, y


    def tokenize(doc):
        """
        Returns a list of strings containing each token in `sentence`
        """
        # return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])",
        #                            doc) if i != '' and i != ' ' and i != '\n']
        tokens = []
        doc = nlp.tokenizer(doc)
        for token in doc:
            tokens.append(token.text)
        return tokens


    try:
        with open('revisedDict.pkl', 'rb') as f:
            revisedDict = pickle.load(f)
    except IOError:
        print("Please run trainLexicon.py to generate revisedDict.pkl file")


    def preprocess(doc):
        global lemma_words
        clean_tokens = []

        # doc = nlp(doc)
        punc = ["``","'"]
        doc = nltk.word_tokenize(doc)

        for token in doc:
            token = str(token).lower()

            token = token.split("'")[0]
            print('token: ', str(token))
            try:
                lemma_words = str(morphology.analyzeAndDisambiguate(str(token)).bestAnalysis()[0].getLemmas()[0])
            except:
                print("except:",str(token))
            if  lemma_words == 'UNK':
                lemma_words = token
            # findings = unidecode(str(findings))

            if not str(lemma_words) in stopwords:
                clean_tokens.append(str(lemma_words))
        return " ".join(clean_tokens)


    # ********* Data augmentation part **************
    class MeraDataset():
        """ Class to find typos based on the keyboard distribution, for QWERTY style keyboards

            It's the actual test set as defined in the paper that we comparing against."""

        def __init__(self, dataset_path):
            """ Instantiate the object.
                @param: dataset_path The directory which contains the data set."""
            self.dataset_path = dataset_path
            self.X_test, self.y_test, self.X_train, self.y_train = self.load()
            self.keyboard_cartesian = {'q': {'x': 0, 'y': 0}, 'w': {'x': 1, 'y': 0}, 'e': {'x': 2, 'y': 0},
                                       'r': {'x': 3, 'y': 0}, 't': {'x': 4, 'y': 0}, 'y': {'x': 5, 'y': 0},
                                       'u': {'x': 6, 'y': 0}, 'i': {'x': 7, 'y': 0}, 'o': {'x': 8, 'y': 0},
                                       'p': {'x': 9, 'y': 0}, 'a': {'x': 0, 'y': 1}, 'z': {'x': 0, 'y': 2},
                                       's': {'x': 1, 'y': 1}, 'x': {'x': 1, 'y': 2}, 'd': {'x': 2, 'y': 1},
                                       'c': {'x': 2, 'y': 2}, 'f': {'x': 3, 'y': 1}, 'b': {'x': 4, 'y': 2},
                                       'm': {'x': 6, 'y': 2}, 'j': {'x': 6, 'y': 1}, 'g': {'x': 4, 'y': 1},
                                       'h': {'x': 5, 'y': 1}, 'k': {'x': 7, 'y': 1}, 'ö': {'x': 11, 'y': 0},
                                       'l': {'x': 8, 'y': 1}, 'v': {'x': 3, 'y': 2}, 'n': {'x': 5, 'y': 2},
                                       'ß': {'x': 10, 'y': 2}, 'ü': {'x': 10, 'y': 2}, 'ä': {'x': 10, 'y': 0}}
            self.nearest_to_i = self.get_nearest_to_i(self.keyboard_cartesian)
            self.splits = self.stratified_split()

        def get_nearest_to_i(self, keyboard_cartesian):
            """ Get the nearest key to the one read.
                @params: keyboard_cartesian The layout of the QWERTY keyboard for English

                return dictionary of eaculidean distances for the characters"""
            nearest_to_i = {}
            for i in keyboard_cartesian.keys():
                nearest_to_i[i] = []
                for j in keyboard_cartesian.keys():
                    if self._euclidean_distance(i, j) < mistake_distance:  # was > 1.2
                        nearest_to_i[i].append(j)
            return nearest_to_i

        def _shuffle_word(self, word, cutoff=0.7):
            """ Rearange the given characters in a word simulating typos given a probability.

                @param: word A single word coming from a sentence
                @param: cutoff The cutoff probability to make a change (default 0.9)

                return The word rearranged
                """
            word = list(word.lower())
            if random.uniform(0, 1.0) > cutoff:
                loc = np.random.randint(0, len(word))
                if word[loc] in self.keyboard_cartesian:
                    word[loc] = random.choice(self.nearest_to_i[word[loc]])
            return ''.join(word)

        def _euclidean_distance(self, a, b):
            """ Calculates the euclidean between 2 points in the keyboard
                @param: a Point one
                @param: b Point two

                return The euclidean distance between the two points"""
            X = (self.keyboard_cartesian[a]['x'] - self.keyboard_cartesian[b]['x']) ** 2
            Y = (self.keyboard_cartesian[a]['y'] - self.keyboard_cartesian[b]['y']) ** 2
            return math.sqrt(X + Y)

        def _get_augment_sentence(self, sentence):
            return ' '.join([self._shuffle_word(item) for item in sentence.split(' ')])

        def _augment_sentence(self, sentence, num_samples):
            """ Augment the dataset of file with a sentence shuffled
                @param: sentence The sentence from the set
                @param: num_samples The number of sentences to genererate

                return A set of augmented sentences"""
            sentences = []
            for _ in range(num_samples):
                sentences.append(self._get_augment_sentence(sentence))
            sentences = list(set(sentences))

            return sentences + [sentence]

        def _augment_split(self, X_train, y_train, num_samples=100):
            """ Split the augmented train dataset
                @param: X_train The full array of sentences
                @param: y_train The train labels in the train dataset
                @param: num_samples the number of new sentences to create (default 1000)

                return Augmented training dataset"""
            Xs, ys = [], []
            for X, y in zip(X_train, y_train):
                tmp_x = self._augment_sentence(X, num_samples)
                sample = [[Xs.append(item), ys.append(y)] for item in tmp_x]
            #             print(X, y)
            #             print(self.augmentedFile+str(self.nSamples)+".csv")

            with open("./datasets/KL/Chatbot/train_augmented.csv", 'w', encoding='utf8') as csvFile:
                fileWriter = csv.writer(csvFile, delimiter='\t')
                for i in range(0, len(Xs) - 1):
                    fileWriter.writerow([Xs[i] + '\t' + ys[i]])
                    # print(Xs[i], "\t", ys[i])
                    # print(Xs[i])
                # fileWriter.writerows(Xs + ['\t'] + ys)
            return Xs, ys

        # Randomly replaces the nouns and verbs by synonyms
        def _synonym_word(self, word, cutoff=0.5):
            if random.uniform(0, 1.0) > cutoff and len(get_synonyms(word)) > 0:
                return random.choice(get_synonyms(word))
            return word

        # Randomly replace words (nouns and verbs) in sentence by synonyms
        def _get_synonym_sentence(self, sentence, cutoff=0.5):
            return ' '.join([self._synonym_word(item, cutoff) for item in sentence.split(' ')])

        # For all classes except the largest ones; add duplicate (possibly augmented) samples until all classes have the same size
        def _oversample_split(self, X_train, y_train, synonym_extra_samples=False, augment_extra_samples=False):
            """ Split the oversampled train dataset
                @param: X_train The full array of sentences
                @param: y_train The train labels in the train dataset

                return Oversampled training dataset"""

            classes = {}
            for X, y in zip(X_train, y_train):
                if y not in classes:
                    classes[y] = []
                classes[y].append(X)

            max_class_size = max([len(entries) for entries in classes.values()])

            Xs, ys = [], []
            for y in classes.keys():
                for i in range(max_class_size):
                    sentence = classes[y][i % len(classes[y])]
                    if i >= len(classes[y]):
                        if synonym_extra_samples:
                            sentence = self._get_synonym_sentence(sentence)
                        if augment_extra_samples:
                            sentence = self._get_augment_sentence(sentence)
                    Xs.append(sentence)
                    ys.append(y)

            # with open(filename_train+"augment", 'w', encoding='utf8') as csvFile:
            #    fileWriter = csv.writer(csvFile, delimiter='\t')
            #    for i in range(0, len(Xs)-1):
            #        fileWriter.writerow([Xs[i] + '\t' + ys[i]])

            return Xs, ys

        def _synonym_split(self, X_train, y_train, num_samples=100):
            """ Split the augmented train dataset
                @param: X_train The full array of sentences
                @param: y_train The train labels in the train dataset
                @param: num_samples the number of new sentences to create (default 1000)

                return Augmented training dataset"""
            Xs, ys = [], []
            for X, y in zip(X_train, y_train):
                sample = [[Xs.append(self._get_synonym_sentence(X)), ys.append(y)] for item in
                          range(additional_synonyms)]
            #             print(X, y)

            # with open(filename_train+"augment", 'w', encoding='utf8') as csvFile:
            #    fileWriter = csv.writer(csvFile, delimiter='\t')
            #    for i in range(0, len(Xs)-1):
            #        fileWriter.writerow([Xs[i] + '\t' + ys[i]])
            return Xs, ys

        def load(self):
            """ Load the file for now only the test.csv, train.csv files hardcoded

                return The vector separated in test, train and the labels for each one"""
            with open(self.dataset_path, encoding="utf-8") as csvfile:
                readCSV = csv.reader(csvfile, delimiter='	')
                all_rows = list(readCSV)
                #             for i in all_rows:
                #                 if i ==  28823:
                #                     print(all_rows[i])
                X_test = [a[2] for a in all_rows]
                y_test = [a[1] for a in all_rows]

            with open(self.dataset_path, encoding="utf-8") as csvfile:
                readCSV = csv.reader(csvfile, delimiter='\t')
                all_rows = list(readCSV)
                X_train = [a[2] for a in all_rows]
                y_train = [a[1] for a in all_rows]
            return X_test, y_test, X_train, y_train



        def stratified_split(self):
            """ Split data whole into stratified test and training sets, then remove stop word from sentences

                return list of dictionaries with keys train,test and values the x and y for each one"""
            self.X_train, self.X_test = (
            [preprocess(sentence) for sentence in self.X_train], [preprocess(sentence) for sentence in self.X_test])

            if oversample:
                self.X_train, self.y_train = self._oversample_split(self.X_train, self.y_train, synonym_extra_samples,
                                                                    augment_extra_samples)
            if additional_synonyms > 0:
                self.X_train, self.y_train = self._synonym_split(self.X_train, self.y_train, additional_synonyms)
            if additional_augments > 0:
                self.X_train, self.y_train = self._augment_split(self.X_train, self.y_train, additional_augments)

            splits = [{"train": {"X": self.X_train, "y": self.y_train},
                       "test": {"X": self.X_test, "y": self.y_test}}]

            return splits

        def get_splits(self):
            """ Get the splitted sentences

                return splitted list of dictionaries"""
            return self.splits


    # ****************************************************

    print("./datasets/KL/" + benchmark_dataset + "/train.csv")
    t0 = time()
    dataset = MeraDataset("./datasets/KL/" + benchmark_dataset + "/train.csv")


    splits = dataset.get_splits()
    xS_train = []
    yS_train = []
    for elem in splits[0]["train"]["X"]:
        xS_train.append(elem)


    for elem in splits[0]["train"]["y"]:
        yS_train.append(intent_dict[elem])
    preprocess_time = time() - t0


    X_train_raw, y_train_raw = read_CSV_datafile(filename=filename_train)
    X_test_raw, y_test_raw = read_CSV_datafile(filename=filename_test)

    X_train_raw = xS_train
    y_train_raw = yS_train

    print("Training data samples: \n", X_train_raw, "\n\n")

    print("Class Labels: \n", y_train_raw, "\n\n")

    print("Size of Training Data: {}".format(len(X_train_raw)))


    #
    #
    #

    # # SemHash

    def find_ngrams(input_list, n):
        return zip(*[input_list[i:] for i in range(n)])


    def semhash_tokenizer(text):
        tokens = text.split(" ")
        final_tokens = []
        for unhashed_token in tokens:
            hashed_token = "#{}#".format(unhashed_token)
            final_tokens += [''.join(gram)
                             for gram in list(find_ngrams(list(hashed_token), 4))]
        return final_tokens


    def semhash_corpus(corpus):
        new_corpus = []
        for sentence in corpus:
            sentence = preprocess(sentence)
            tokens = semhash_tokenizer(sentence)
            new_corpus.append(" ".join(map(str, tokens)))
        return new_corpus


    t0 = time()
    X_train_raw = semhash_corpus(X_train_raw)
    X_test_raw = semhash_corpus(X_test_raw)
    semhash_time = time() - t0



    def get_vectorizer(corpus, preprocessor=None, tokenizer=None):
        if VECTORIZER == "count":
            vectorizer = CountVectorizer(analyzer='word')  # ,ngram_range=(1,1))
            vectorizer.fit(corpus)
            feature_names = vectorizer.get_feature_names()
        elif VECTORIZER == "hash":
            vectorizer = HashingVectorizer(analyzer='word', n_features=2 ** 10, non_negative=True)
            vectorizer.fit(corpus)
            feature_names = None
        elif VECTORIZER == "tfidf":
            vectorizer = TfidfVectorizer(analyzer='word')
            vectorizer.fit(corpus)
            feature_names = vectorizer.get_feature_names()
        else:
            raise Exception("{} is not a recognized Vectorizer".format(VECTORIZER))
        return vectorizer, feature_names


    def trim(s):
        """Trim string to fit on terminal (assuming 80-column display)"""
        return s if len(s) <= 80 else s[:77] + "..."


    # #############################################################################
    # Benchmark classifiers
    def benchmark(clf, X_train, y_train, X_test, y_test, target_names,
                  print_report=True, feature_names=None, print_top10=False,
                  print_cm=True):
        new_list = []
        print('_' * 80)
        print("Training: ")
        print(clf)
        t0 = time()
        clf.fit(X_train, y_train)
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)

        t0 = time()
        pred = clf.predict(X_test)
        print(pred)
        test_time = time() - t0

        print("test time:  %0.3fs" % test_time)

        score = metrics.accuracy_score(y_test, pred)
        f1_score = metrics.f1_score(y_test, pred, average='weighted')
        lookup = {value: key for key, value in intent_dict.items()}
        print(lookup)

        for p in pred:
            new_list.append(lookup[p])
        #
        df["pred"] = new_list
        df.to_csv(benchmark_dataset + "_"+str(clf).split('(')[0] + ".csv", sep='\t', encoding='utf-8')


        #bad_pred = X_test[pred != y_test]

        print("accuracy:   %0.3f" % score)
        #print("Accuracy: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))

        if hasattr(clf, 'coef_'):
            print("dimensionality: %d" % clf.coef_.shape[1])
            print("density: %f" % density(clf.coef_))

            if print_top10 and feature_names is not None:
                print("top 10 keywords per class:")
                for i, label in enumerate(["Make Update", "Setup Printer", "Shutdown Computer","Software Recommendation", "None"]):
                    top10 = np.argsort(clf.coef_[i])[-10:]
                    print(trim("%s: %s" % (label, " ".join([feature_names[i] for i in top10]))))


        if print_report:
            print("classification report:")
            print(metrics.classification_report(y_test, pred,labels = range(len(target_names)),
                                                target_names=target_names))

        if print_cm:
            print("confusion matrix:")
            print(metrics.confusion_matrix(y_test, pred))

        with open("./"+RESULT_FILE, 'a', encoding='utf8') as csvFile:
            fileWriter = csv.writer(csvFile, delimiter='\t')
            fileWriter.writerow([benchmark_dataset,str(clf), str(score), str(train_time), str(test_time)])
        with open("./"+"_"+RESULT_FILE, 'a', encoding='utf8') as csvFile:
            fileWriter = csv.writer(csvFile, delimiter='\t', lineterminator="\n")
            fileWriter.writerow(['{0:.3f}'.format(score)])


        print()
        clf_descr = str(clf).split('(')[0]
        return clf_descr, score, train_time, test_time, f1_score




    def plot_results(results):
        # make some plots
        indices = np.arange(len(results))

        results = [[x[i] for x in results] for i in range(4)]

        clf_names, score, training_time, test_time = results
        training_time = np.array(training_time) / np.max(training_time)
        test_time = np.array(test_time) / np.max(test_time)

        plt.figure(figsize=(12, 8))
        plt.title("Score")
        plt.barh(indices, score, .2, label="score", color='navy')
        plt.barh(indices + .3, training_time, .2, label="training time",
                 color='c')
        plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
        plt.yticks(())
        plt.legend(loc='best')
        plt.subplots_adjust(left=.25)
        plt.subplots_adjust(top=.95)
        plt.subplots_adjust(bottom=.05)

        for i, c in zip(indices, clf_names):
            plt.text(-.3, i, c)

        plt.show()




    def data_for_training():
        vectorizer, feature_names = get_vectorizer(X_train_raw, preprocessor=preprocess, tokenizer=tokenize)

        X_train = vectorizer.transform(X_train_raw).toarray()
        X_test = vectorizer.transform(X_test_raw).toarray()

        return X_train, y_train_raw, X_test, y_test_raw, feature_names

    t0 = time()
    X_train, y_train, X_test, y_test, feature_names = data_for_training()
    vectorize_time = time()-t0

    # with open("./"+METADATA_FILE, 'a', encoding='utf8') as csvFile:
    #         fileWriter = csv.writer(csvFile, delimiter='\t')
    #         fileWriter.writerow([benchmark_dataset,str(oversample),str(synonym_extra_samples),str(augment_extra_samples),str(additional_synonyms),str(additional_augments),str(mistake_distance),str(preprocess_time),str(semhash_time),str(vectorize_time)])





    for _ in enumerate(range(NUMBER_OF_RUNS_PER_SETTING)):
        split = 0
        target_names = None
        if benchmark_dataset == "Chatbot":
            target_names = ["Departure Time", "Find Connection"]
        elif benchmark_dataset == "AskUbuntu":
            target_names = ["Make Update", "Setup Printer", "Shutdown Computer","Software Recommendation", "None"]
        elif benchmark_dataset == "WebApplication":
            target_names = ["Download Video", "Change Password", "None", "Export Data", "Sync Accounts",
                      "Filter Spam", "Find Alternative", "Delete Account"]
        print("Train Size: {}\nTest Size: {}".format(X_train.shape[0], X_test.shape[0]))
        results = []
        #alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
        parameters_mlp={'hidden_layer_sizes':[(100,50), (300, 100),(300,200,100)]}
        parameters_RF={ "n_estimators" : [50,60,70],
               "min_samples_leaf" : [1, 11]}
        k_range = list(range(3,7))
        parameters_knn = {'n_neighbors':k_range}
        knn=KNeighborsClassifier(n_neighbors=5)
        for clf, name in [
                (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
                (GridSearchCV(knn,parameters_knn, cv=5),"gridsearchknn"),
                #(Perceptron(n_iter=50), "Perceptron"),
                (GridSearchCV(MLPClassifier(activation='tanh'),parameters_mlp, cv=5),"gridsearchmlp"),
                (PassiveAggressiveClassifier(max_iter=50), "Passive-Aggressive"),
                (GridSearchCV(RandomForestClassifier(n_estimators=10),parameters_RF, cv=5),"gridsearchRF")
        ]:

            print('=' * 80)
            print(name)
            result = benchmark(clf, X_train, y_train, X_test, y_test, target_names,
                                     feature_names=feature_names)
            results.append(result)

           # print('parameters')
           # print(clf.grid_scores_[0])
            #print('CV Validation Score')
           # print(clf.grid_scores_[0].cv_validation_scores)
           # print('Mean Validation Score')
           # print(clf.grid_scores_[0].mean_validation_score)
           # grid_mean_scores = [result.mean_validation_score for result in clf.grid_scores_]
           # print(grid_mean_scores)
           # plt.plot(k_range, grid_mean_scores)
           # plt.xlabel('Value of K for KNN')
           # plt.ylabel('Cross-Validated Accuracy')

        parameters_Linearsvc = [{'C': [1, 10], 'gamma': [0.1,1.0]}]
        for penalty in ["l2", "l1"]:
        #    print('=' * 80)
        #    print("%s penalty" % penalty.upper())
            # Train Liblinear model
            grid=(GridSearchCV(LinearSVC,parameters_Linearsvc, cv=10),"gridsearchSVC")
            #results.append(benchmark(LinearSVC(penalty=penalty), X_train, y_train, X_test, y_test, target_names,
            #                         feature_names=feature_names))

            result = benchmark(LinearSVC(penalty=penalty, dual=False,tol=1e-3),
                                     X_train, y_train, X_test, y_test, target_names,
                                     feature_names=feature_names)
            results.append(result)

            # Train SGD model
            result = benchmark(SGDClassifier(alpha=.0001,max_iter=50,
                                                   penalty=penalty),
                                     X_train, y_train, X_test, y_test, target_names,
                                     feature_names=feature_names)
            results.append(result)

        # Train SGD with Elastic Net penalty
        #print('=' * 80)
        #print("Elastic-Net penalty")
        results.append(benchmark(SGDClassifier(alpha=.0001,max_iter=50,
                                               penalty="elasticnet"),
                                 X_train, y_train, X_test, y_test, target_names,
                                 feature_names=feature_names))

        # Train NearestCentroid without threshold
        #print('=' * 80)
        #print("NearestCentroid (aka Rocchio classifier)")
        results.append(benchmark(NearestCentroid(),
                                 X_train, y_train, X_test, y_test, target_names,
                                 feature_names=feature_names))

        # Train sparse Naive Bayes classifiers
        #print('=' * 80)
        #print("Naive Bayes")
        results.append(benchmark(MultinomialNB(alpha=.01),
                                 X_train, y_train, X_test, y_test, target_names,
                                 feature_names=feature_names))

        result = benchmark(BernoulliNB(alpha=.01),
                                 X_train, y_train, X_test, y_test, target_names,
                                 feature_names=feature_names)
        results.append(result)


        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        result = benchmark(Pipeline([
                                      ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                                                      tol=1e-3))),
                                      ('classification', LinearSVC(penalty="l2"))]),
                                 X_train, y_train, X_test, y_test, target_names,
                                 feature_names=feature_names)
        results.append(result)

        kfold = model_selection.KFold(n_splits=2, random_state=0)
        results.append(benchmark(LogisticRegression(C=1.0, class_weight=None, dual=False,
              fit_intercept=True, intercept_scaling=1, max_iter=100,
              multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
              solver='liblinear', tol=0.0001, verbose=0, warm_start=False),
                                 X_train, y_train, X_test, y_test, target_names,
                                 feature_names=feature_names))

        #plot_results(results)
        # print(results)


