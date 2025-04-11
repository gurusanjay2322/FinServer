from flask import Flask, request, jsonify
import pandas as pd
import joblib
from pymongo import MongoClient
from utils import extract_features_from_transactions, explain_reason, suggest_improvements,clean_transaction_keys,get_deepseek_suggestions  

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

    if not data or 'user_id' not in data:
        return jsonify({'error': 'Missing user_id parameter'}), 400

    try:
        user_id = data['user_id']
        
        # Fetch user document from MongoDB
        user_data = transactions_collection.find_one({'user_id': user_id})
        
        if not user_data:
            return jsonify({'error': f'No user found with ID {user_id}. Please upload transactions first.'}), 404
            
        if 'transactions' not in user_data or not user_data['transactions']:
            return jsonify({'error': f'No transactions found for user {user_id}. Please upload transactions first.'}), 404
            
        # Convert transactions array to DataFrame
        tx_df = pd.DataFrame(user_data['transactions'])

        # Extract features
        features = extract_features_from_transactions(tx_df)
        X_user = scaler.transform(features[feature_cols])
        predicted_score = float(model.predict(X_user)[0])

        # Explanations
        explanation = explain_reason(features.iloc[0])
        improvements = suggest_improvements(features.iloc[0])
        deepseek_advice = get_deepseek_suggestions(explanation, improvements)
        # Get last 5 transactions (sorted by time descending)
        tx_df['Timestamp'] = pd.to_datetime(tx_df['Timestamp'])
        last_5 = tx_df.sort_values(by='Timestamp', ascending=False).head(5)
        last_5_cleaned = clean_transaction_keys(last_5.to_dict(orient='records'))
        
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
    'deepseek_advice': deepseek_advice,
    'last_5_transactions': last_5_cleaned
})


    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/user_credit_history/<user_id>', methods=['GET'])
def get_user_credit_history(user_id):
    try:
        # Get user credit scores history
        scores = list(credit_scores_collection.find(
            {'user_id': user_id},
            {'_id': 0}  # Exclude MongoDB ObjectId
        ))
        
        if not scores:
            return jsonify({'message': f'No credit history found for user {user_id}'}), 404
        
        return jsonify({
            'user_id': user_id,
            'credit_history': scores
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upload_transactions', methods=['POST'])
def upload_transactions():
    data = request.get_json()
    
    if not data or 'user_id' not in data or 'transactions' not in data:
        return jsonify({'error': 'Missing user_id or transactions data'}), 400
        
    try:
        user_id = data['user_id']
        transactions = data['transactions']
        
        # Check if user already exists
        existing_user = transactions_collection.find_one({'user_id': user_id})
        
        if existing_user:
            # Update existing user's transactions
            transactions_collection.update_one(
                {'user_id': user_id},
                {'$set': {'transactions': transactions}}
            )
            return jsonify({
                'message': f'Updated transactions for user {user_id}',
                'transaction_count': len(transactions)
            })
        else:
            # Create new user document
            user_document = {
                'user_id': user_id,
                'transactions': transactions
            }
            transactions_collection.insert_one(user_document)
            return jsonify({
                'message': f'Added new user {user_id} with transactions',
                'transaction_count': len(transactions)
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':  
    app.run(debug=True)