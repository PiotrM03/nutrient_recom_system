from flask import Flask, request, jsonify
from rf_index import NutrientDeficiencyPredictor
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
predictor = NutrientDeficiencyPredictor('new_db.xlsx')

@app.route('/predict', methods=['POST'])
def predict_nutrient_deficiency():
    data = request.json
    print("Received data:", data)
    symptoms = data['symptoms']
    nutrient, recommendations = predictor.identify_nutrients(symptoms)
    return jsonify({
        'nutrient': nutrient,
        'recommendations': recommendations
    })

if __name__ == '__main__':
    app.run(debug=True)
