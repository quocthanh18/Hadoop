import os
import string
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')


def load_stopwords(file_path):
    with open(file_path, 'r') as f:
        stopwords = f.read().splitlines()
    return set(stopwords)


# Load the stopwords from your file
stop_words = load_stopwords("E:/stopwords.txt")


def preprocess_file(file_path):
    with open(file_path, 'r') as f:
        text = f.read()

    # Convert to lower case
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the text
    word_tokens = word_tokenize(text)
    # Remove stopwords
    filtered_text = [word for word in word_tokens if not word in stop_words]

    # Stem the words
    # You can use the PorterStemmer from the nltk library
    stemmer = nltk.PorterStemmer()
    filtered_text = [stemmer.stem(word) for word in filtered_text]
    return ' '.join(filtered_text)


def preprocess_directory(dir_path):
    for filename in os.listdir(dir_path):
        if filename.endswith(".txt"):
            print(f"Processing {filename}")
            preprocessed_text = preprocess_file(os.path.join(dir_path, filename))
            with open(os.path.join(dir_path, f"{filename}"), 'w', encoding="utf-8") as f:
                f.write(preprocessed_text)


# Call the function on your directory
preprocess_directory("E:/bbc/sport")
preprocess_directory("E:/bbc/tech")
preprocess_directory("E:/bbc/business")
preprocess_directory("E:/bbc/politics")
preprocess_directory("E:/bbc/entertainment")
