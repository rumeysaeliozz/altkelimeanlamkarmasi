import os
import csv
import clip
import torch
import fasttext
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel


def get_qi_pair(file_path, lang):
    """
    file_path: file path of the file you want ot read
    lang: which language you want to use ('en', 'tr)
    :return: [inputs], [intents]
    """
    with open(file_path) as csvfile:
        read_csv = csv.reader(csvfile, delimiter='\t')
        all_rows = list(read_csv)

    inputs = []
    intents = []
    for row in all_rows:
        if lang == "tr":
            inputs.append(row[2])
        else:
            inputs.append(row[0])
        intents.append(row[1])
    return inputs, intents


class Extractor:
    def __init__(self):
        self.model = None

    def clip_model_load(self, device):
        if not isinstance(self.model, torch.jit._script.RecursiveScriptModule):
            self.model, _ = clip.load("ViT-B/32", device=device)

    def clip_features(self, inputs, device="cpu"):
        text = clip.tokenize(inputs).to(device)
        self.clip_model_load(device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
        return text_features

    def cc_model_load(self):
        if not isinstance(self.model, fasttext.FastText._FastText):
            self.model = fasttext.load_model('cc.tr.300.bin')

    def cc_features(self, inputs):
        self.cc_model_load()
        vectors = []
        for text in inputs:
            vectors.append(self.model.get_sentence_vector(text))
        return vectors

    def get_features(self, model_name, inputs, device="cpu"):
        if model_name == "cc":
            return self.cc_features(inputs)
        elif model_name == "clip":
            return self.clip_features(inputs, device)
        else:
            print("Wrong model name! Only use 'cc' or 'clip'")
            raise


extractor = Extractor()
model_name = "clip"
lang = "en"

folders = ["AskUbuntu", "WebApplication", "Chatbot"]
data = pd.DataFrame(columns=folders)
for folder in folders:
    print(folder)



    train_inputs, y_train = get_qi_pair(os.path.join(folder, 'train.csv'), lang)
    test_inputs, y_test = get_qi_pair(os.path.join(folder, 'test.csv'), lang)

    x_train = extractor.get_features(model_name, train_inputs)
    x_test = extractor.get_features(model_name, test_inputs)

    parameters_knn = {
        'n_neighbors': list(range(1, 11))}
    parameters_mlp = {
        'hidden_layer_sizes': [(100, 50), (300, 100), (300, 200, 100)],
        'activation': ["logistic", "tanh"],
        'max_iter': [500, 700]}
    parameters_rf = {
        "n_estimators": [10, 20, 30, 40, 50],
        "min_samples_leaf": [1, 5, 11]
    }
    parameters_lsvc = {
        "penalty": ["l1", "l2"],
        "dual": [False],
        "tol": [1e-3]
    }
    parameters_sgd = {
        "penalty": ["elasticnet", "l1", "l2"],
        "alpha": [.0001],
        "max_iter": [50]
    }

    for clf, name in [
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (GridSearchCV(KNeighborsClassifier(), parameters_knn, cv=5), "GS-KNN"),
        (GridSearchCV(MLPClassifier(), parameters_mlp, cv=5), "GS-MLP"),
        (GridSearchCV(RandomForestClassifier(), parameters_rf, cv=5), "GS-RandomForest"),
        (GridSearchCV(LinearSVC(), parameters_lsvc, cv=5), "GS-LinearSVC"),
        (PassiveAggressiveClassifier(max_iter=50), "Passive-Aggressive"),
        (Pipeline([('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False, tol=1e-3))),
                   ('classification', LinearSVC(penalty="l2"))]), "Pipeline"),
        (LogisticRegression(C=1.0, class_weight=None, dual=False,
                            fit_intercept=True, intercept_scaling=1, max_iter=100,
                            multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
                            solver='liblinear', tol=0.0001, verbose=0, warm_start=False),
         "LogisticRegression"),
        # (MultinomialNB(alpha=.01), "MultinomialNB"),  # featurelar arasında negatif değerler olduğu için kullanılamaz
        (BernoulliNB(alpha=.01), "BernoulliNB"),
        (NearestCentroid(), "NearestCentroid"),
        (GridSearchCV(SGDClassifier(), parameters_sgd, cv=5), "GS-SGD")
    ]:
        print("\t" + name)
        clf.fit(x_train, y_train)
        test_score = clf.score(x_test, y_test)
        data.loc[name, folder] = test_score
data.to_csv(lang + "_" + model_name + ".csv")
