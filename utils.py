import pandas as pd
import numpy as np

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
        suggestions.append(
            "You have too many failed transactions. Ensure you have a stable internet connection and enough account balance while making payments."
        )
    if row['Circular Tx'] > 2:
        suggestions.append(
            "Avoid sending money to yourself repeatedly. This behavior might appear suspicious and impact your financial credibility."
        )
    if row['Expense/Income Ratio'] > 1.5:
        suggestions.append(
            "You're spending more than you're earning. Try to reduce unnecessary expenses and increase your savings."
        )
    if row['Frequency'] < 0.2:
        suggestions.append(
            "Your transaction activity is quite irregular. Regular and consistent transactions improve your financial profile."
        )
    if row['Avg Sent'] > 2 * row['Avg Received']:
        suggestions.append(
            "You're sending a lot more money than you're receiving. Try to maintain a better balance between your inflow and outflow."
        )
    if row['Total Received'] < 1000:
        suggestions.append(
            "Your account receives a low amount of funds. Try to increase your income sources or maintain a healthier inflow."
        )
    if row['Total Sent'] > 10000 and row['Total Received'] < 5000:
        suggestions.append(
            "You're sending out a large amount with low incoming funds. This creates an imbalance in your financial behavior."
        )
    if row['Num Sent'] > 20 and (row['Failed Tx'] / row['Num Sent']) > 0.2:
        suggestions.append(
            "A significant portion of your transactions are failing. Check your app, bank, or network settings to ensure smoother transactions."
        )
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
