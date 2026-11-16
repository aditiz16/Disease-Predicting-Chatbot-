# This will evaluate the model on the test dataset and prove its credibility 
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score
from dataset import SymptomDataset

def textbook_test(train,test):
    # dataset = SymptomDataset(train)
    X_train, X_test, y_train, y_test = train_test_split(train.features(), train.labels(), test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy on the test set: {accuracy:.4f}")
    X1_test, y1_test = test.features(), test.labels()
    accuracy_test = model.score(X1_test, y1_test)
    print(f"Accuracy on the external test set: {accuracy_test:.4f}")

def count_overlap(train,test):
    # dataset_train = SymptomDataset(train)
    # dataset_test = SymptomDataset(test)
    # train_symptoms = set(train.symptoms())
    # test_symptoms = set(test.symptoms())
    shared_rows = pd.merge(train.data, test.data, on=train.symptoms().tolist(), how='inner')
    print(f"Number of overlapping rows: {len(shared_rows)}")
    train_symptoms = set(train.symptoms())
    test_symptoms = set(test.symptoms())
    overlap = train_symptoms.intersection(test_symptoms)
    # print(f"Number of overlapping symptoms: {len(overlap)}")
    # print(f"Overlapping symptoms: {overlap}")

def crossvalidate(dataset,folds=5):
    model = LogisticRegression(max_iter=1000)
    scores = cross_val_score(model, dataset.features(), dataset.labels(), cv=folds)
    print(f"Cross-validation scores: {scores}")
    print(f"Mean accuracy: {np.mean(scores):.4f}")

def hide_symptoms(row,symptoms,k,rng):
    # This function will randomnly hide k symptoms from the given row of symptoms 
    # row: a pandas Series representing the symptoms of a patient
    # symptoms: a list of all possible symptoms
    # k: number of symptoms to hide
    # rng: a random number generator
    present_symptoms = row[row == 1].index.tolist()
    keep_symptoms = rng.choice(present_symptoms, size=min(k, len(present_symptoms)), replace=False)
    hidden_row=row.copy()
    hidden_row[:] = 0
    hidden_row[keep_symptoms] = 1
    return hidden_row

def realistic_test(dataset,k,seed=0):
    rng = np.random.default_rng(seed)
    X_train, X_test, y_train, y_test = train_test_split(dataset.features(), dataset.labels(), test_size=0.2, random_state=seed)
    # modified_features = dataset.features().apply(lambda row: hide_symptoms(row, dataset.symptoms(), k, rng), axis=1)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    X_hidden = X_test.apply(lambda row: hide_symptoms(row, dataset.symptoms(), k, rng), axis=1)
    top1=model.score(X_hidden, y_test)
    probs = model.predict_proba(X_hidden)
    top3_names = model.classes_[np.argsort(probs, axis=1)[:, -3:]]
    top3=np.mean([x in names for x,names in zip(y_test, top3_names)])
    # accuracy = model.score(X_test, y_test)
    # print(f"Accuracy after hiding {k} symptoms: {accuracy:.4f}")
    print(f"Top-1 accuracy with only {k} symptoms shown: {top1:.4f}")
    print(f"Top-3 accuracy with only {k} symptoms shown: {top3:.4f}")

if __name__ == "__main__":
    train_file = "data/Training.csv"
    test_file = "data/Testing.csv"
    dataset_train = SymptomDataset(train_file)
    dataset_test = SymptomDataset(test_file)
    print("Evaluating the model on the test dataset...")
    textbook_test(dataset_train, dataset_test)
    count_overlap(dataset_train, dataset_test)
    for k in (2, 3, 4):
        realistic_test(dataset_train, k=k, seed=0)
    # realistic_test(dataset_train, k=5, seed=0)