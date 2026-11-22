from email.mime import text
from sys import flags

from matplotlib import lines

from classifier import DiseaseClassifier
from dataset import SymptomDataset
from extractor import SymptomExtractor

# data=SymptomDataset("data/Training.csv")
# extractor= SymptomExtractor(data)
# clf=DiseaseClassifier(data)
# clf.train_model()

# text=input()
# found = extractor.extract_symptoms(text)
# print(found,clf.predict(found))

class Chatbot:
    """A symptom-checker you can talk to."""
 
    RED_FLAGS = {
        "chest_pain", "breathlessness", "weakness_of_one_body_side", "slurred_speech",
        "altered_sensorium", "coma", "blood_in_sputum", "stomach_bleeding", "fast_heart_rate",
    }
    CONFIDENCE_THRESHOLD = 0.15
    QUIT_WORDS = {"exit", "quit", "bye"}
 
    def __init__(self, dataset_path="data/Training.csv"):
        self.dataset = SymptomDataset(dataset_path)               # loaded once...
        self.extractor = SymptomExtractor(self.dataset)           # ...shared here
        self.classifier = DiseaseClassifier(self.dataset)         # ...and here
        self.classifier.train_model()
        self.known_symptoms = []                                  # the chatbot's memory
 
    def reset(self):
        """Forget the conversation so far."""
        self.known_symptoms = []
 
    def _red_flags(self):
        """The remembered symptoms that need urgent care."""
        return sorted(set(self.known_symptoms) & self.RED_FLAGS)
 
    def _describe(self, top):
        """Turn [(disease, probability), ...] into readable lines."""
        lines = ["Possible conditions:"]
        for disease, probability in top:
            lines.append(f"  - {disease.title()}: {probability:.0%}")
        return "\n".join(lines)
 
    def respond(self, text):
        """Return the bot's reply to one message (no printing here)."""
        if text.strip().lower() in ("reset", "start over"):
            self.reset()
            return "OK, let's start again. What symptoms do you have?"
 
        found = self.extractor.extract_symptoms(text)
        if not found:
            return ("I couldn't recognise any symptoms. "
                    "Try something like: 'itching and a skin rash'.")
 
        for symptom in found:
            if symptom not in self.known_symptoms:
                self.known_symptoms.append(symptom)
 
        readable = ", ".join(s.replace("_", " ") for s in self.known_symptoms)
        lines = [f"Symptoms so far: {readable}."]
 
        if self._red_flags():
            lines.insert(0, "WARNING: Some of these symptoms can be signs of an emergency. "
                            "Please get medical help now: call 112, or 108 for an ambulance.")
 
        top = self.classifier.predict_top(self.known_symptoms)
        if top[0][1] < self.CONFIDENCE_THRESHOLD:
            lines.append("These symptoms fit many conditions, so I can't narrow it down yet. "
                         "Can you tell me more?")
        else:
            lines.append(self._describe(top))
 
        lines.append("This is not a diagnosis. Please see a doctor.")
        return "\n".join(lines)
 
    def run(self):
        """The chat loop: read a message, reply, repeat until the user quits."""
        print("Welcome to the Symptom Chatbot!")
        print("Describe your symptoms. Type 'reset' to start over, or 'exit' to quit.\n")
        while True:
            try:
                user_input = input("You: ")
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye! Take care.")
                break
            if user_input.strip().lower() in self.QUIT_WORDS:
                print("Goodbye! Take care.")
                break
            if not user_input.strip():
                continue
            print(f"Chatbot: {self.respond(user_input)}\n")
 
 
if __name__ == "__main__":
    Chatbot().run()