# Importing necessary libraries
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class NutrientDeficiencyPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_prep()
        self.train_model()

    def data_prep(self):
        # Load data and preprocess
        self.data = pd.read_excel(self.data_path)
        self.data.dropna(subset=['def_symptom'], inplace=True)
        self.data['symptoms_list'] = self.data['def_symptom'].apply(lambda x: x.split(', '))
        self.mlb = MultiLabelBinarizer()
        symptoms_encoded = self.mlb.fit_transform(self.data['symptoms_list'])
        symptoms_df = pd.DataFrame(symptoms_encoded, columns=self.mlb.classes_)
        
        self.le = LabelEncoder()
        self.data['nutrient_encoded'] = self.le.fit_transform(self.data['nutrient'])
        
        self.X = symptoms_df
        self.Y = self.data['nutrient_encoded']
        
    def train_model(self):
        # Split data and train model
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=40)
        
        self.nb_model = GaussianNB()
        self.nb_model.fit(X_train, Y_train)
        
        preds_train = self.nb_model.predict(X_train)
        preds_test = self.nb_model.predict(X_test)
        print(classification_report(Y_train, preds_train, digits=3))
        print(classification_report(Y_test, preds_test, digits=3))
        
        cf_matrix = confusion_matrix(Y_test, preds_test)
        plt.figure(figsize=(12, 8))
        sns.heatmap(cf_matrix, annot=True, fmt="d", cmap='Oranges')
        plt.title("Confusion Matrix on Test Data")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
    
    def identify_nutrients(self, user_symptoms):
        # Predict nutrient deficiency based on symptoms
        if not user_symptoms:
            # If user symptoms list is empty, print an error message and return
            print("Error: No symptoms were provided. Please enter at least one symptom.")
            return None, "No dietary recommendations available due to lack of symptoms."
        
        input_features = self.mlb.transform([user_symptoms])
        prediction_encoded = self.nb_model.predict(input_features)[0]
        nutrient_prediction = self.le.inverse_transform([prediction_encoded])[0]
        dietary_recs = self.data.loc[self.data['nutrient'] == nutrient_prediction, 'diet_rec']
        dietary_recommendations = ", ".join(dietary_recs.unique())
        
        return nutrient_prediction, dietary_recommendations
    
    def collect_user_symptoms(self):
        # Collect symptoms from the user
        user_input = input("Please enter your symptoms separated by a comma (e.g., headache, fatigue): ")
        user_symptoms = [symptom.strip() for symptom in user_input.split(',')]
        nutrient_deficiency, dietary_recommendation = self.identify_nutrients(user_symptoms)
        if nutrient_deficiency:
            print(f"Based on your symptoms, the predicted nutrient deficiency is: {nutrient_deficiency}.")
            print(f"Dietary recommendations: {dietary_recommendation}")

if __name__ == "__main__":
    dataset_path = "new_db.xlsx"
    predictor = NutrientDeficiencyPredictor(dataset_path)
    predictor.collect_user_symptoms()
