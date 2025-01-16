import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

# Download NLTK data files
nltk.download('stopwords')
nltk.download('punkt')

# Load dataset (modify path if needed)
data = pd.read_csv("C:/Users/SUBHIKSHA/Desktop/internship/sentiment.csv", encoding='latin1')

# Assign column names based on the dataset
data.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

# Map sentiment values (0 -> negative, 2 -> neutral, 4 -> positive)
sentiment_mapping = {0: 'negative', 2: 'neutral', 4: 'positive'}
data['sentiment'] = data['sentiment'].map(sentiment_mapping)

# Preprocessing function
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r"http\S+", " ", text)
    # Remove mentions (e.g., @username)
    text = re.sub(r"@\w+", " ", text)
    # Remove non-alphanumeric characters
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    # Convert text to lowercase
    text = text.lower()
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Stem tokens
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

# Apply preprocessing
data['processed_text'] = data['text'].apply(preprocess_text)

# Define features and labels
X = data['processed_text']
y = data['sentiment']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initialize Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train_vec, y_train)

# Predict on test data
y_pred = model.predict(X_test_vec)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# Display metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot sentiment distribution
sns.countplot(data=data, x='sentiment', palette='viridis')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
