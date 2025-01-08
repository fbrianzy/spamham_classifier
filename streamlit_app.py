import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
path = './data/enron_modelling.csv'
df = pd.read_csv(path)
df = df.dropna()
df['Spam/Ham'] = df['Spam/Ham'].map({'spam': 1, 'ham': 0})

# Vectorize the text data
vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, max_features=1500)
X = vectorizer.fit_transform(df['Stemm_Str']).toarray()
y = df['Spam/Ham']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the MultinomialNB class
class MultinomialNB():
    def __init__(self, alpha=1):
        self.alpha = alpha 

    def fit(self, X_train, y_train):
        m, n = X_train.shape
        self.classes = np.unique(y_train)
        self.class_count = np.zeros(len(self.classes))
        self.feature_count = np.zeros((len(self.classes), n))
        self.feature_prob = np.zeros((len(self.classes), n))
        
        for i, c in enumerate(self.classes):
            X_c = X_train[y_train == c]
            self.class_count[i] = X_c.shape[0]
            self.feature_count[i, :] = X_c.sum(axis=0)
        
        self.class_prob = self.class_count / self.class_count.sum()
        self.feature_prob = (self.feature_count + self.alpha) / (self.class_count[:, None] + self.alpha * n)

    def predict(self, X):
        log_prob = np.log(self.class_prob) + X @ np.log(self.feature_prob.T)
        return self.classes[np.argmax(log_prob, axis=1)]

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

# Streamlit app
st.title('Spam/Ham Classifier')

# Display accuracy using st.info
st.info(f'Model Accuracy: {accuracy:.2f}%')

# Center the image
st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://raw.githubusercontent.com/fbrianzy/spamham_classifier/main/assets/she_classifier_icon.png" alt="App Icon" style="width: 500px;">
    </div>
    """, 
    unsafe_allow_html=True
)

user_input = st.text_area('Enter the text to classify')

if st.button('Predict'):
    if user_input:
        user_input_vectorized = vectorizer.transform([user_input]).toarray()
        prediction = model.predict(user_input_vectorized)
        result = 'Spam' if prediction[0] == 1 else 'Ham'
        if result == 'Spam':
            st.error(f'The text is classified as: {result}')
        else:
            st.success(f'The text is classified as: {result}')
    else:
        st.write('Please enter some text to classify.')
