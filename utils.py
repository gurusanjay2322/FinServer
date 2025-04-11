import pandas as pd
import numpy as np
import os
import google.generativeai as genai

# Load Gemini API key from environment variable
genai.configure(api_key="AIzaSyDnocuEMTbVguWoxBwBGICbjk7hxrx7T_c")

def extract_features_from_transactions(user_tx_df):
    user_tx_df['Timestamp'] = pd.to_datetime(user_tx_df['Timestamp'])

    sent = user_tx_df[user_tx_df['Type'] == 'Sent']
    received = user_tx_df[user_tx_df['Type'] == 'Received']

    total_sent = sent['Amount (INR)'].sum()
    total_received = received['Amount (INR)'].sum()
    num_sent = len(sent)
    num_received = len(received)
    failed_tx = len(sent[sent['Status'] == 'FAILED'])

    p2p_ratio = len(sent[sent['To Type'] == 'P2P']) / num_sent if num_sent else 0
    merchant_ratio = len(sent[sent['To Type'] == 'Merchant']) / num_sent if num_sent else 0
    avg_sent = sent['Amount (INR)'].mean() if num_sent else 0
    avg_received = received['Amount (INR)'].mean() if num_received else 0

    span_days = (sent['Timestamp'].max() - sent['Timestamp'].min()).days if num_sent > 1 else 1
    frequency = num_sent / span_days if span_days else 0

    circular_tx = len(user_tx_df[user_tx_df['Sender UPI ID'] == user_tx_df['Receiver UPI ID']])
    expense_income_ratio = total_sent / total_received if total_received > 0 else 10

    return pd.DataFrame([{
        'Total Sent': total_sent,
        'Total Received': total_received,
        'Num Sent': num_sent,
        'Num Received': num_received,
        'Failed Tx': failed_tx,
        'P2P Ratio': p2p_ratio,
        'Merchant Ratio': merchant_ratio,
        'Avg Sent': avg_sent,
        'Avg Received': avg_received,
        'Frequency': frequency,
        'Circular Tx': circular_tx,
        'Expense/Income Ratio': expense_income_ratio
    }])


def suggest_improvements(row):
    suggestions = []

    if row['Failed Tx'] > 2:
        suggestions.append("Too many failed transactions. Ensure stable internet and sufficient balance.")
    if row['Circular Tx'] > 2:
        suggestions.append("Avoid repeated self-transfers. It looks suspicious.")
    if row['Expense/Income Ratio'] > 1.5:
        suggestions.append("You're spending more than earning. Try to save more.")
    if row['Frequency'] < 0.2:
        suggestions.append("Increase your transaction frequency for a healthier profile.")
    if row['Avg Sent'] > 2 * row['Avg Received']:
        suggestions.append("You're sending a lot more than receiving. Try to balance it.")
    if row['Total Received'] < 1000:
        suggestions.append("Try increasing your inflow by improving income sources.")
    if row['Total Sent'] > 10000 and row['Total Received'] < 5000:
        suggestions.append("You're sending too much with little inflow. Not ideal.")
    if row['Num Sent'] > 20 and (row['Failed Tx'] / row['Num Sent']) > 0.2:
        suggestions.append("High failure rate in sent transactions. Check your payment app.")

    if not suggestions:
        return {
            "message": "You're on the right track! Keep maintaining your healthy financial habits.",
            "suggestions": []
        }

    return {
        "message": "Here are a few things you can improve to boost your credit score:",
        "suggestions": suggestions
    }


def explain_reason(row):
    reasons = []
    if row['Failed Tx'] > 2:
        reasons.append("Too many failed transactions")
    if row['Circular Tx'] > 2:
        reasons.append("Suspicious circular transfers")
    if row['Expense/Income Ratio'] > 1.5:
        reasons.append("Spending more than earning")
    if row['Frequency'] < 0.2:
        reasons.append("Irregular transaction activity")
    return " | ".join(reasons) if reasons else "Healthy behavior"


def clean_transaction_keys(transactions):
    cleaned = []
    for tx in transactions:
        new_tx = {}
        for key, value in tx.items():
            clean_key = key.replace(" (INR)", "").replace(" ", "_")
            new_tx[clean_key] = value
        cleaned.append(new_tx)
    return cleaned


def get_deepseek_suggestions(explanation: str, improvements: dict) -> str:
    prompt = f"""
    A user has the following financial behavior:
    - Explanation: {explanation}
    - Suggestions: {', '.join(improvements['suggestions']) if improvements['suggestions'] else "None"}

    Write a professional but friendly 5-6 sentence suggestion to help the user improve their credit score.
    """

    try:
        model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")
        response = model.generate_content([prompt])  # Make sure it's wrapped in a list
        return response.text.strip()
    except Exception as e:
        return f"Sorry, we couldn't fetch advice from Gemini: {str(e)}"

