from flask import Flask, request, jsonify
import pandas as pd
import joblib
from utils import extract_features_from_transactions, explain_reason, suggest_improvements


app = Flask(__name__)

model = joblib.load('model/credit_score_model.pkl')
scaler = joblib.load('model/scaler.pkl')

feature_cols = [
    'Total Sent', 'Total Received', 'Num Sent', 'Num Received',
    'Failed Tx', 'P2P Ratio', 'Merchant Ratio', 'Avg Sent',
    'Avg Received', 'Frequency', 'Circular Tx', 'Expense/Income Ratio'
]

@app.route('/predict_score', methods=['POST'])
def predict_score():
    data = request.get_json()

    if not data or 'transactions' not in data:
        return jsonify({'error': 'Missing transaction data'}), 400

    try:
        tx_df = pd.DataFrame(data['transactions'])
        tx_df['Timestamp'] = pd.to_datetime(tx_df['Timestamp'])
        tx_df = tx_df.sort_values(by='Timestamp', ascending=False)
        last_five_tx = tx_df.head(5).to_dict(orient='records')

        features = extract_features_from_transactions(tx_df)
        X_user = scaler.transform(features[feature_cols])
        predicted_score = float(model.predict(X_user)[0])
        explanation = explain_reason(features.iloc[0])
        improvements = suggest_improvements(features.iloc[0])

        return jsonify({
            'credit_score': round(predicted_score, 2),
            'explanation': explanation,
            'improvements': improvements,
            'last_5_transactions': last_five_tx
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':  
    app.run(debug=True)