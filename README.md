# Women-cloth-review-
# Import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset (replace with your own data)
reviews = [...]  # list of reviews
labels = [...]  # list of corresponding labels (e.g., 0 for negative, 1 for positive)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)

# Create a bag-of-words representation using CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X_train_count = vectorizer.fit_transform(X_train)
X_test_count = vectorizer.transform(X_test)

# Train the Multinomial Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_count, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test_count)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
```

