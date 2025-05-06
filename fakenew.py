import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import re
import string
import joblib

# Load datasets
df_fake = pd.read_csv("dataset/Fake.csv")
df_true = pd.read_csv("dataset/True.csv")

# Assign class labels
df_fake["class"] = 0
df_true["class"] = 1

# Remove last 10 rows for manual testing
df_fake_manual_testing = df_fake.tail(10)
for i in range(23480, 23470, -1):
    df_fake.drop([i], axis=0, inplace=True)

df_true_manual_testing = df_true.tail(10)
for i in range(21416, 21406, -1):
    df_true.drop([i], axis=0, inplace=True)

# Assign class labels to manual testing data
df_fake_manual_testing["class"] = 0
df_true_manual_testing["class"] = 1

# Combine manual testing data and save to CSV
df_manual_testing = pd.concat([df_fake_manual_testing, df_true_manual_testing], axis=0)
df_manual_testing.to_csv("manual_testing.csv")

# Merge fake and true dataframes
df_merge = pd.concat([df_fake, df_true], axis=0)

# Drop unnecessary columns
df = df_merge.drop(["title", "subject", "date"], axis=1)

# Shuffle the dataframe
df = df.sample(frac=1).reset_index(drop=True)

# Text processing function
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Apply text processing
df["text"] = df["text"].apply(wordopt)

# Define features and target
x = df["text"]
y = df["class"]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Vectorize text data
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Logistic Regression
LR = LogisticRegression()
LR.fit(xv_train, y_train)
pred_lr = LR.predict(xv_test)
print("Logistic Regression Accuracy:", LR.score(xv_test, y_test))
print("Logistic Regression Classification Report:\n", classification_report(y_test, pred_lr))

# Decision Tree Classifier
DT = DecisionTreeClassifier(random_state=0)
DT.fit(xv_train, y_train)
pred_dt = DT.predict(xv_test)
print("Decision Tree Accuracy:", DT.score(xv_test, y_test))
print("Decision Tree Classification Report:\n", classification_report(y_test, pred_dt))

# Gradient Boosting Classifier
GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(xv_train, y_train)
pred_gbc = GBC.predict(xv_test)
print("Gradient Boosting Accuracy:", GBC.score(xv_test, y_test))
print("Gradient Boosting Classification Report:\n", classification_report(y_test, pred_gbc))

# Random Forest Classifier
RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)
pred_rfc = RFC.predict(xv_test)
print("Random Forest Accuracy:", RFC.score(xv_test, y_test))
print("Random Forest Classification Report:\n", classification_report(y_test, pred_rfc))

# Save models
joblib.dump(LR, 'Linear_Regression.pkl')
joblib.dump(DT, 'Decision_tree.pkl')
joblib.dump(GBC, 'GradientBoostingClassifier.pkl')
joblib.dump(RFC, 'RandomForestClassifier.pkl')

# Function to output label
def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

# Manual testing function
def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    print("\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(
        output_lable(pred_LR[0]),
        output_lable(pred_DT[0]),
        output_lable(pred_GBC[0]),
        output_lable(pred_RFC[0])
    ))

# Example usage
if __name__ == "__main__":
    news = input("Enter news article text: ")
    manual_testing(news)