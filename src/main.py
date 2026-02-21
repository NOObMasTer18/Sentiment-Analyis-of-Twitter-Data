import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from wordcloud import WordCloud

# -----------------------------
# STEP 1: Load Data
# -----------------------------
twitts_train = pd.read_csv("data/twitter_training.csv")
twitts_valid = pd.read_csv("data/twitter_validation.csv")

# Rename columns
column_name = ['TweetID','Entity','Sentiment','Tweet_Content']
twitts_train.columns = column_name
twitts_valid.columns = column_name

# Combine dataframes
twitts = pd.concat([twitts_train, twitts_valid], ignore_index=True)
print("Initial Shape:", twitts.shape)

# -----------------------------
# STEP 2: Data Cleaning
# -----------------------------
twitts.dropna(inplace=True)
twitts.drop_duplicates(inplace=True)
print("After Cleaning:", twitts.shape)

twitts = twitts[['Entity','Sentiment','Tweet_Content']]

# -----------------------------
# STEP 3: Text Preprocessing
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

twitts['Clean_Tweet'] = twitts['Tweet_Content'].apply(clean_text)

# -----------------------------
# STEP 4: Encode Sentiment
# -----------------------------
twitts = twitts[twitts['Sentiment'].isin(['Positive','Negative','Neutral'])]
print("Classes:\n", twitts['Sentiment'].value_counts())

# -----------------------------
# STEP 5: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    twitts['Clean_Tweet'], twitts['Sentiment'],
    test_size=0.2, random_state=42
)

# -----------------------------
# STEP 6: TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# STEP 7: Model Training
# -----------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

# -----------------------------
# STEP 8: Model Evaluation Dashboard
# -----------------------------
print("\n--- MODEL EVALUATION DASHBOARD ---\n")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"✅ Accuracy: {accuracy:.2f}")
print(f"✅ Precision: {precision:.2f}")
print(f"✅ Recall: {recall:.2f}")
print(f"✅ F1 Score: {f1:.2f}")
print("\nDetailed Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Purples')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# STEP 9: Visualization Dashboard
# -----------------------------
print("\n--- VISUALIZATION DASHBOARD ---\n")

# 1️⃣ Sentiment Distribution
plt.figure(figsize=(6,5))
twitts['Sentiment'].value_counts().plot(kind='bar', color=['red','green','gray'])
plt.title("Overall Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Tweet Count")
plt.show()

# 2️⃣ Entity-wise Sentiment Distribution
plt.figure(figsize=(10,6))
sns.countplot(data=twitts, x='Entity', hue='Sentiment', palette='Set2')
plt.title("Entity-wise Sentiment Distribution")
plt.xticks(rotation=90)
plt.show()

# 3️⃣ WordCloud for Positive, Negative, Neutral Tweets
def plot_wordcloud(sentiment):
    text = ' '.join(twitts[twitts['Sentiment'] == sentiment]['Clean_Tweet'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(8,4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"WordCloud for {sentiment} Tweets", fontsize=16)
    plt.show()

for s in ['Positive','Negative','Neutral']:
    plot_wordcloud(s)

# 4️⃣ Feature Importance Visualization (Top Words)

feature_names = np.array(vectorizer.get_feature_names_out())
class_labels = model.classes_
top_n = 10

for i, label in enumerate(class_labels):
    top_indices = np.argsort(model.feature_log_prob_[i])[-top_n:]
    plt.figure(figsize=(8,4))
    plt.barh(feature_names[top_indices], model.feature_log_prob_[i][top_indices], color='skyblue')
    plt.title(f"Top Words for {label} Sentiment")
    plt.xlabel("Importance (Log Probability)")
    plt.tight_layout()
    plt.show()

feature_names = np.array(vectorizer.get_feature_names_out())
class_labels = model.classes_
top_n = 10


print("\nDashboard Visualization Complete ✅")


