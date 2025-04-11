from flask import Flask, request, jsonify
import pandas as pd
import joblib
from pymongo import MongoClient
from utils import extract_features_from_transactions, explain_reason, suggest_improvements,clean_transaction_keys


app = Flask(__name__)

# MongoDB Connection
mongo_uri = "mongodb+srv://root:root@ipo-builder.lpq9ub9.mongodb.net/"
client = MongoClient(mongo_uri)
db = client["fin_credit_db"]  # Database name
transactions_collection = db["transactions"]  # Collection for storing transactions
credit_scores_collection = db["credit_scores"]  # Collection for storing credit scores

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

        # Extract features
        features = extract_features_from_transactions(tx_df)
        X_user = scaler.transform(features[feature_cols])
        predicted_score = float(model.predict(X_user)[0])

        # Explanations
        explanation = explain_reason(features.iloc[0])
        improvements = suggest_improvements(features.iloc[0])

        # Get last 5 transactions (sorted by time descending)
        tx_df['Timestamp'] = pd.to_datetime(tx_df['Timestamp'])
        last_5 = tx_df.sort_values(by='Timestamp', ascending=False).head(5)
        last_5_cleaned = clean_transaction_keys(last_5.to_dict(orient='records'))
        
        # Store transactions in MongoDB
        if 'user_id' in data:
            user_id = data['user_id']
            # Store transactions
            for tx in data['transactions']:
                tx['user_id'] = user_id
                transactions_collection.insert_one(tx)
            
            # Store credit score result
            credit_score_record = {
                'user_id': user_id,
                'credit_score': round(predicted_score, 2),
                'timestamp': pd.Timestamp.now(),
                'explanation': explanation,
                'improvements': improvements
            }
            credit_scores_collection.insert_one(credit_score_record)

        return jsonify({
            'credit_score': round(predicted_score, 2),
            'explanation': explanation,
            'improvements': improvements,
            'last_5_transactions': last_5_cleaned
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# New endpoint to retrieve user transaction history
@app.route('/user_history/<user_id>', methods=['GET'])
def get_user_history(user_id):
    try:
        # Get user transactions
        transactions = list(transactions_collection.find(
            {'user_id': user_id}, 
            {'_id': 0}  # Exclude MongoDB ObjectId
        ))
        
        # Get user credit scores
        scores = list(credit_scores_collection.find(
            {'user_id': user_id},
            {'_id': 0}  # Exclude MongoDB ObjectId
        ))
        
        return jsonify({
            'user_id': user_id,
            'transactions': transactions,
            'credit_scores': scores
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':  
    app.run(debug=True)