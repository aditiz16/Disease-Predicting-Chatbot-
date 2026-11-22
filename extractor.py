# This will fuction as fetcing and matching against the dataset.
from dataset import SymptomDataset
import difflib
import re

# symptoms = SymptomDataset("data/Training.csv").symptoms()
# # print(symptoms)
# input_text = "itcing"
# print(f"Input text: {input_text}")
# for i in (0.6,0.7,0.8,0.9):
#     diff=difflib.get_close_matches(input_text, symptoms, n=5, cutoff=i)
#     print(f"Cutoff: {i}, Matches: {diff}")


class SymptomExtractor:
    SYNONYMS = {
        "itching": ["itch", "itchy", "pruritus"],
        "skin_rash": ["rash", "eruption", "dermatitis"],
        "nodal_skin_eruptions": ["nodular rash", "nodular eruptions"],
        "vomiting": ["throwing up", "emesis", "nausea"],
        "muscle_wasting": ["muscle loss", "atrophy"],
        "patches_in_throat": ["throat patches", "throat lesions"],
        "high_fever": ["fever"],
        "fatigue": ["tired", "tiredness"],
        "itching": ["itch", "itchy", "pruritus", "itchiness"]
    }
    def __init__(self, dataset):
        self.dataset = dataset
        self.symptoms = dataset.symptoms()
        self.phrases = self._generate_phrases()

    def _normalise(self, text):
        text = text.lower().replace("_", " ")
        return " ".join(re.findall(r"[a-z]+", text))

    def _generate_phrases(self):
        # phrase as a person would type it -> dataset name
        phrases = {symptom.replace("_", " "): symptom for symptom in self.symptoms}
        for symptom, syn_list in self.SYNONYMS.items():
            if symptom in self.symptoms:
                for synonym in syn_list:
                    phrases[synonym] = symptom
        return phrases

    def _match_phrases(self, text):
        found = []
        text = f" {text} "
        for phrase in sorted(self.phrases, key=len, reverse=True):
            if f" {phrase} " in text:
                found.append(self.phrases[phrase])
                text = text.replace(f" {phrase} ", " ")   
        return found, text

    def _match_typos(self, text):
        found = []
        words = text.split()
        for size in (3, 2, 1):                    
            for i in range(len(words) - size + 1):
                group = words[i:i + size]
                if "#" in group:                   
                    continue
                candidate = " ".join(group)
                if len(candidate) < 5:            
                    continue
                match = difflib.get_close_matches(candidate, list(self.phrases), n=1, cutoff=0.85)
                if match:
                    found.append(self.phrases[match[0]])
                    words[i:i + size] = ["#"] * size
        return found
    
    def extract_symptoms(self, text):
        text = self._normalise(text)
        found, leftover = self._match_phrases(text)
        found += self._match_typos(leftover)
        return list(dict.fromkeys(found))

    def reset(self):
        self.symptoms = self.dataset.symptoms()
        self.phrases = self._generate_phrases()

# if __name__ == "__main__":
#     dataset = SymptomDataset("data/Training.csv")
#     extractor = SymptomExtractor(dataset)
#     test_inputs = [
#         "I have an itchy rash and nodular eruptions.",
#         "Experiencing pruritus and dermatitis.",
#         "Feeling a bit of itcing and skin_rash.",
#         "I have a nodular rash and some itchiness.",
#         "My skin has a rash and I feel itchy.",
#         "I have internal itching and a nodular skin eruption.",
#         "I have a skin rash and some pruritus.",
#         "Hello I am Aditi",
#         "I have coughs but no fever"
#     ]
#     for input_text in test_inputs:
#         extracted = extractor.extract_symptoms(input_text)
#         print(f"Input: {input_text}\nExtracted Symptoms: {extracted}\n")