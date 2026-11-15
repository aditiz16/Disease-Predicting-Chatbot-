import pandas as pd
# df=pd.read_csv("data/Training.csv")
# print (df.shape)
# print (df.head())
class SymptomDataset:
    TARGET = 'prognosis'
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.raw_rows = len(self.data)
        self._clean()
    # Cleaning the dataset
    def _clean(self):
        self.drop_unnamed_columns()
        self.fix_symptom_names()
        self.fix_disease_names()
        self.check_missing_values()
        self.data=self.data.drop_duplicates().reset_index(drop=True)
    def drop_unnamed_columns(self):
        unnamed_cols = [col for col in self.data.columns if "Unnamed" in col]
        self.data.drop(columns=unnamed_cols, inplace=True)
    
    def fix_symptom_names(self):
        self.data.columns = [col.strip().lower().replace(" ", "_") for col in self.data.columns]
    def fix_disease_names(self):
        self.data[self.TARGET] = self.data[self.TARGET].str.strip().str.lower()
    def check_missing_values(self):
        missing_values = self.data.isnull().sum()
        if missing_values.any():
            print("Missing values found:")
            print(missing_values[missing_values > 0])
        else:
            print("No missing values found.")
    # Features , labels and symptoms
    def symptoms(self):
        return self.data.columns[self.data.columns != self.TARGET]
    def features(self):
        return self.data[self.symptoms()]
    def labels(self):
        return self.data[self.TARGET]
    
    def clean(self):
        print("Data cleaned successfully.")
        print(f"Number of unique prognoses: {self.data[self.TARGET].nunique()}")
        print(f"Shape of the cleaned dataset: {self.data.shape}")
        print (self.data.head())

# if __name__ == "__main__":
#     dataset = SymptomDataset("data/Training.csv")
#     dataset.clean()

