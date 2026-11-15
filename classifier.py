from sklearn.linear_model import LogisticRegression
from dataset import SymptomDataset
import pandas as pd

# dataset = SymptomDataset("data/Training.csv")

# model = LogisticRegression(max_iter=1000)
# model.fit(dataset.features(), dataset.labels())
# print("Model trained successfully.")
# print(model.classes_)
# print(model.predict(dataset.features().iloc[[0]]))
# print(model.predict_proba(dataset.features().iloc[[0]]))

class DiseaseClassifier:
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = LogisticRegression(max_iter=1000)
        self.train_model()

    def train_model(self):
        self.model.fit(self.dataset.features(), self.dataset.labels())
        print("Model trained successfully.")

    def predict(self, symptoms):
        input_data = {symptom: 1 if symptom in symptoms else 0 for symptom in self.dataset.symptoms()}
        input_df = pd.DataFrame([input_data])
        prediction = self.model.predict(input_df)
        return prediction[0]

    def predict_proba(self, symptoms):
        input_data = {symptom: 1 if symptom in symptoms else 0 for symptom in self.dataset.symptoms()}
        input_df = pd.DataFrame([input_data])
        probabilities = self.model.predict_proba(input_df)
        return dict(zip(self.model.classes_, probabilities[0]))

if __name__ == "__main__":
    dataset = SymptomDataset("data/Training.csv")
    classifier = DiseaseClassifier(dataset)
    test_symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions']
    prediction = classifier.predict(test_symptoms)
    probabilities = classifier.predict_proba(test_symptoms)
    print(f"Predicted disease: {prediction}")
    print(f"Prediction probabilities: {probabilities}")