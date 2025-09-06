import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset inline (label = 'spam' or 'ham' (not spam))
data = {
    'text': [
        "Congratulations! You've won a free ticket to Bahamas. Call now!",
        "Hey, are we still meeting tomorrow?",
        "Urgent! Your account has been compromised, verify your details.",
        "Don't forget the meeting at 10 AM.",
        "Win a brand new car by entering the contest!",
        "Please review the attached document.",
        "Get cheap meds online without prescription.",
        "Can you send me the report by EOD?",
        "You have been selected for a $1000 Walmart gift card.",
        "Lunch at 1 pm?"
    ],
    'label': [
        'spam', 'ham', 'spam', 'ham', 'spam',
        'ham', 'spam', 'ham', 'spam', 'ham'
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Split data into features and labels
X = df['text']
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Vectorize text data into TF-IDF features
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict on test data
y_pred = model.predict(X_test_vec)

# Evaluate model
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
