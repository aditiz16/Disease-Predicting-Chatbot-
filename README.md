# Symptom Checker Chatbot

A command-line chatbot that takes symptoms in plain English, such as *"I have itching and a skin rash"*, and suggests the three most likely conditions using a machine-learning model.

I first built this for my Honours seminar in AI/ML at SCTR's Pune Institute of Computer Technology (May 2025). The original code was lost, so I rebuilt it from scratch in object-oriented Python, and evaluated it properly this time.

> ⚠️ **Not medical advice.** This is a learning project trained on a small public dataset. It cannot diagnose anything. If you feel unwell, please see a doctor.

## What it does

- **Understands everyday text:** matches symptom names, synonyms ("pruritus" → itching) and small typos ("itcing").
- **Remembers the conversation:** "I have a fever", then "I also have chills", counts both.
- **Shows its top 3 guesses** with probabilities, and asks for more detail when it isn't confident.
- **Flags emergency symptoms** (such as chest pain or slurred speech) before anything else.

```text
You: I have a fever
Chatbot: Symptoms so far: high fever.
These symptoms fit many conditions, so I can't narrow it down yet. Can you tell me more?
This is not a diagnosis. Please see a doctor.
```

## Project structure

| File | What it does |
| --- | --- |
| `dataset.py` | `SymptomDataset`: loads and cleans the data (4,920 rows → 304 unique patients) |
| `classifier.py` | `DiseaseClassifier`: trains a logistic regression model and returns the top 3 diseases |
| `extractor.py` | `SymptomExtractor`: turns free text into the dataset's symptom names |
| `chatbot.py` | `Chatbot`: the conversation loop, memory and safety checks |
| `evaluate.py` | Measures how accurate the model really is |
| `plots.py` | PCA, symptom heatmap and correlation plots |

## Getting started

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python3 -m venv .venv
source .venv/bin/activate
pip install pandas scikit-learn matplotlib
```

Download the data from Kaggle (a free login is needed) and put the CSV files in a `data/` folder:

- `Training.csv` and `Testing.csv` from [Disease Prediction Using Machine Learning](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning)
- `symptom_Description.csv` and `symptom_precaution.csv` from [Disease Symptom Prediction](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)

Then run:

```bash
python chatbot.py     # talk to the chatbot ('reset' to start over, 'exit' to quit)
python evaluate.py    # see the accuracy results
python visualise.py       # create the plots
```

## How accurate is it?

The standard test (train on `Training.csv`, test on `Testing.csv`) gives **100%**, but all 41 test rows also appear in the training data, so that number is meaningless.

A fairer test trains on 80% of the unique patients, tests on
