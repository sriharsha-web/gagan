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
import os
import sys

print("Fake News Classification System")
print("=" * 40)

# Create directory for saving models if it doesn't exist
os.makedirs('models', exist_ok=True)

# ====== DATASET LOADING SECTION ======
# Function to find dataset files
def find_dataset_files(fake_filename="Fake.csv", true_filename="True.csv"):
    """
    Try to locate the dataset files in various possible locations
    """
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define possible locations to check
    possible_locations = [
        # Current directory
        (".", ".", fake_filename, true_filename),
        # Script directory
        (script_dir, script_dir, fake_filename, true_filename),
        # Dataset subdirectory in current location
        ("dataset", "dataset", fake_filename, true_filename),
        # Dataset subdirectory in script location
        (os.path.join(script_dir, "dataset"), os.path.join(script_dir, "dataset"), fake_filename, true_filename),
        # Data subdirectory
        ("data", "data", fake_filename, true_filename),
        # User's Desktop (common place for downloaded files)
        (os.path.expanduser("~/Desktop"), os.path.expanduser("~/Desktop"), fake_filename, true_filename),
        # User's Downloads folder
        (os.path.expanduser("~/Downloads"), os.path.expanduser("~/Downloads"), fake_filename, true_filename)
    ]
    
    # Check each location
    for fake_dir, true_dir, fake_file, true_file in possible_locations:
        fake_path = os.path.join(fake_dir, fake_file)
        true_path = os.path.join(true_dir, true_file)
        
        if os.path.isfile(fake_path) and os.path.isfile(true_path):
            print(f"Found dataset files at:")
            print(f"- Fake news: {fake_path}")
            print(f"- True news: {true_path}")
            return fake_path, true_path
            
    # If we get here, files weren't found
    return None, None

# Try to find and load the dataset files
fake_path, true_path = find_dataset_files()

# If files not found, ask user for paths
if fake_path is None or true_path is None:
    print("\nERROR: Could not automatically find the dataset files.")
    print("Please enter the full paths to your dataset files:")
    
    fake_path = input("Path to Fake.csv: ").strip()
    if not fake_path:
        fake_path = "/Users/sriharshas/Desktop/Fake.csv"  # Default from error message
        print(f"Using default path: {fake_path}")
        
    true_path = input("Path to True.csv: ").strip()
    if not true_path:
        true_path = "/Users/sriharshas/Desktop/True.csv"  # Default based on fake path
        print(f"Using default path: {true_path}")

# Load the datasets
try:
    print(f"\nLoading fake news data from: {fake_path}")
    df_fake = pd.read_csv(fake_path)
    print(f"Successfully loaded {len(df_fake)} fake news articles")
    
    print(f"Loading true news data from: {true_path}")
    df_true = pd.read_csv(true_path)
    print(f"Successfully loaded {len(df_true)} true news articles")
    
except FileNotFoundError as e:
    print(f"\nERROR: {e}")
    print("\nPlease make sure the files exist at the specified locations and try again.")
    print("You can download the dataset from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
    sys.exit(1)
except Exception as e:
    print(f"\nUnexpected error loading datasets: {e}")
    sys.exit(1)

# Display dataset information
print("\nDataset Information:")
print(f"Fake news dataset shape: {df_fake.shape}")
print(f"True news dataset shape: {df_true.shape}")
print(f"Fake news columns: {', '.join(df_fake.columns)}")
print(f"True news columns: {', '.join(df_true.columns)}")

# Add source column to track the origin of each article
df_fake["source"] = "fake"
df_true["source"] = "true"

# Assign class labels
df_fake["class"] = 0
df_true["class"] = 1

# Check if dataframes have enough rows before removing
if len(df_fake) >= 10 and len(df_true) >= 10:
    # Remove last 10 rows for manual testing
    df_fake_manual_testing = df_fake.tail(10).copy()
    df_fake = df_fake.iloc[:-10].reset_index(drop=True)
    
    df_true_manual_testing = df_true.tail(10).copy()
    df_true = df_true.iloc[:-10].reset_index(drop=True)
    
    print(f"\nSeparated {len(df_fake_manual_testing) + len(df_true_manual_testing)} articles for manual testing")
else:
    print("\nWarning: Dataframes don't have enough rows for proper manual testing")
    # Take a smaller sample if needed
    df_fake_manual_testing = df_fake.tail(min(5, len(df_fake))).copy()
    df_true_manual_testing = df_true.tail(min(5, len(df_true))).copy()

# Combine manual testing data and save to CSV
df_manual_testing = pd.concat([df_fake_manual_testing, df_true_manual_testing], axis=0)
manual_testing_path = "manual_testing.csv"
df_manual_testing.to_csv(manual_testing_path, index=False)
print(f"Manual testing dataset saved to: {os.path.abspath(manual_testing_path)}")

# Merge fake and true dataframes
df_merge = pd.concat([df_fake, df_true], axis=0)
print(f"\nCombined dataset shape: {df_merge.shape}")

# Keep a copy of titles for potential analysis
if 'title' in df_merge.columns:
    titles = df_merge['title'].copy()

# Check which columns exist before dropping
columns_to_drop = []
for col in ["title", "subject", "date"]:
    if col in df_merge.columns:
        columns_to_drop.append(col)

# Drop unnecessary columns - consider keeping date for temporal analysis
df = df_merge.drop(columns_to_drop, axis=1)
print(f"Dropped columns: {', '.join(columns_to_drop)}")

# Shuffle the dataframe
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print("Dataset shuffled for training")

# Text processing function
def wordopt(text):
    if not isinstance(text, str):
        return ""  # Handle non-string inputs
    
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)  # Use r prefix for raw strings
    text = re.sub(r'\W', " ", text)  # Non-word characters to space
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\n', ' ', text)  # Replace newlines with spaces
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing digits
    return text.strip()  # Remove leading/trailing whitespace

# Apply text processing
print("\nProcessing text data...")
df["text"] = df["text"].apply(wordopt)
print("Text processing complete")

# Check for and remove empty texts after processing
empty_texts = df["text"].str.strip().eq('')
if empty_texts.any():
    print(f"Removing {empty_texts.sum()} empty text entries after processing")
    df = df[~empty_texts]

# Define features and target
x = df["text"]
y = df["class"]

# Check if we have sufficient data
if len(x) < 100:  # Arbitrary minimum threshold
    print("\nWARNING: Dataset size is very small for ML training")
    print(f"Current size: {len(x)} samples")
    print("Results may not be reliable")

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

print(f"\nTraining set size: {len(x_train)}, Testing set size: {len(x_test)}")

# Vectorize text data with better parameters
print("Vectorizing text data...")
vectorization = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.85)
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)
print(f"Vocabulary size: {len(vectorization.vocabulary_)}")

# Save the vectorizer for future use
vectorizer_path = os.path.join('models', 'tfidf_vectorizer.pkl')
joblib.dump(vectorization, vectorizer_path)
print(f"Vectorizer saved to: {os.path.abspath(vectorizer_path)}")

# Model evaluation function
def evaluate_model(model, name, xv_train, y_train, xv_test, y_test):
    print(f"\nTraining {name}...")
    model.fit(xv_train, y_train)
    
    # Evaluate on test set
    pred = model.predict(xv_test)
    accuracy = accuracy_score(y_test, pred)
    report = classification_report(y_test, pred)
    
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"{name} Classification Report:")
    print(report)
    
    # Save model
    model_filename = f"{name.replace(' ', '_')}.pkl"
    model_path = os.path.join('models', model_filename)
    joblib.dump(model, model_path)
    print(f"{name} model saved to: {os.path.abspath(model_path)}")
    
    return model, pred

# Initialize models with better parameters
print("\n" + "=" * 40)
print("Training Machine Learning Models")
print("=" * 40)

models = {
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
}

# Train and evaluate models
trained_models = {}
for name, model in models.items():
    trained_model, predictions = evaluate_model(
        model, name, xv_train, y_train, xv_test, y_test
    )
    trained_models[name] = trained_model

# Function to output label
def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not Fake News"

# Function to predict with confidence scores
def predict_news(news, trained_models, vectorizer):
    if not isinstance(news, str) or not news.strip():
        return "Error: Please provide valid text for analysis"
    
    # Process the input text
    processed_news = wordopt(news)
    
    if not processed_news.strip():
        return "Error: Text is empty after processing"
    
    # Create DataFrame for consistency
    testing_news = pd.DataFrame({"text": [processed_news]})
    
    # Transform using the same vectorizer used during training
    new_xv_test = vectorizer.transform(testing_news["text"])
    
    # Get predictions from all models
    results = {}
    for name, model in trained_models.items():
        prediction = model.predict(new_xv_test)[0]
        
        # Get probability scores if the model supports it
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(new_xv_test)[0]
            confidence = proba[prediction]
            results[name] = {
                "prediction": output_label(prediction),
                "confidence": f"{confidence:.2f}"
            }
        else:
            results[name] = {
                "prediction": output_label(prediction)
            }
    
    return results

# Function for manual testing
def manual_testing(news):
    results = predict_news(news, trained_models, vectorization)
    
    print("\nPrediction Results:")
    for model_name, result in results.items():
        if "confidence" in result:
            print(f"{model_name}: {result['prediction']} (Confidence: {result['confidence']})")
        else:
            print(f"{model_name}: {result['prediction']}")
    
    # Determine consensus
    predictions = [result["prediction"] for result in results.values()]
    if len(set(predictions)) == 1:
        print(f"\nConsensus: All models predict this is {predictions[0]}")
    else:
        fake_count = predictions.count("Fake News")
        real_count = predictions.count("Not Fake News")
        if fake_count > real_count:
            print(f"\nMajority ({fake_count}/{len(predictions)}): This appears to be Fake News")
        elif real_count > fake_count:
            print(f"\nMajority ({real_count}/{len(predictions)}): This appears to be Not Fake News")
        else:
            print("\nNo consensus: Models are split on this news")

# Example usage
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Fake News Classifier - Interactive Testing")
    print("=" * 60)
    print("Enter 'quit' to exit the program")
    
    while True:
        print("\n" + "-" * 40)
        news = input("\nEnter news article text (or 'quit' to exit): ")
        if news.lower() in ['quit', 'exit', 'q']:
            print("\nExiting program. Thank you for using the Fake News Classifier!")
            break
        manual_testing(news)
