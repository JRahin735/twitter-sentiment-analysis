#%%
import numpy as np
import pandas as pd
import nltk
import sklearn 
import string
import re # helps you filter urls
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
#%%
#%%
#Whether to test your Q9 for not? Depends on correctness of all modules
def test_pipeline():
    return True # Make this true when all tests pass

# Convert part of speech tag from nltk.pos_tag to word net compatible format
# Simple mapping based on first letter of return tag to make grading consistent
# Everything else will be considered noun 'n'
posMapping = {
# "First_Letter by nltk.pos_tag":"POS_for_lemmatizer"
    "N":'n',
    "V":'v',
    "J":'a',
    "R":'r'
}


# Additional Downloads
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def get_wordnet_pos(treebank_tag):
    """ Map POS tag from pos_tag() to WordNet compatible POS tags. """
    return posMapping.get(treebank_tag[0].upper(), 'n')


#%%
def process(text, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    """ Normalizes case and handles punctuation
    Inputs:
        text: str: raw text
        lemmatizer: an instance of a class implementing the lemmatize() method
                    (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    Outputs:
        list(str): tokenized text
    """
    text = text.lower()
    text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)  # Improved URL removal
    text = re.sub(r"\'s", '', text)  # Remove possessive apostrophe
    text = re.sub(r"\'", '', text)   # Remove all remaining apostrophes
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    tokens = word_tokenize(text)
    processed_tokens = []
    for word, tag in pos_tag(tokens):
        try:
            lemmatized_word = lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag))
            processed_tokens.append(lemmatized_word)
        except Exception:
            pass

    processed_tokens = [token for token in processed_tokens if token not in string.punctuation or token == '…']

    return processed_tokens


#%%
def process_all(df, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    """ process all text in the dataframe using process function.
    Inputs
        df: pd.DataFrame: dataframe containing a column 'text' loaded from the CSV file
        lemmatizer: an instance of a class implementing the lemmatize() method
                    (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    Outputs
        pd.DataFrame: dataframe in which the values of text column have been changed from str to list(str),
                        the output from process_text() function. Other columns are unaffected.
    """
    df['text'] = df['text'].apply(lambda x: process(' '.join(x) if isinstance(x, list) else x, lemmatizer))
    return df    
#%%
def create_features(processed_tweets, stop_words):
    """ creates the feature matrix using the processed tweet text
    Inputs:
        processed_tweets: pd.DataFrame: processed tweets read from train/test csv file, containing the column 'text'
        stop_words: list(str): stop_words by nltk stopwords (after processing)
    Outputs:
        sklearn.feature_extraction.text.TfidfVectorizer: the TfidfVectorizer object used
            we need this to tranform test tweets in the same way as train tweets
        scipy.sparse.csr.csr_matrix: sparse bag-of-words TF-IDF feature matrix
    """
    # Convert to list
    stop_words_list = list(stop_words)
    # Initialize
    tfidf = TfidfVectorizer (lowercase=False, min_df=2, stop_words=stop_words_list, tokenizer=lambda x:x)
    # Convert to a list of lists
    processed_tweets_list = processed_tweets['text'].tolist()
    # Transform
    X = tfidf.fit_transform(processed_tweets_list)
        
    # Get feature names and sort them
    feature_names = np.array(tfidf.get_feature_names_out())
    sorted_indices = np. argsort (feature_names)
    sorted_feature_names = feature_names [sorted_indices]
    
    # Sort the TF-IDF matrix by the sorted feature names
    X_sorted = X[:, sorted_indices]
    
    return tfidf, X_sorted


#%%
def create_labels(processed_tweets):
    """ creates the class labels from screen_name
    Inputs:
        processed_tweets: pd.DataFrame: tweets read from train file, containing the column 'screen_name'
    Outputs:
        numpy.ndarray(int): dense binary numpy array of class labels
    """
    return np.where(processed_tweets['screen_name'].isin(['realDonaldTrump', 'mike_pence', 'GOP']), 0, 1)
    
#%%
class MajorityLabelClassifier():
    """
    A classifier that predicts the mode of training labels
    """
    def __init__(self):
        """
        Initialize your parameter here
        """
        self.mode_label = None
        
    def fit(self, X, y):
        """
        Implement fit by taking training data X and their labels y and finding the mode of y
        i.e. store your learned parameter
        """
        self.mode_label = np.bincount(y).argmax()
    
    def predict(self, X):
        """
        Implement to give the mode of training labels as a prediction for each data instance in X
        return labels
        """
        return np.full(X.shape[0], self.mode_label)

#%%
def learn_classifier(X_train, y_train, kernel):
    """ learns a classifier from the input features and labels using the kernel function supplied
    Inputs:
        X_train: scipy.sparse.csr.csr_matrix: sparse matrix of features, output of create_features()
        y_train: numpy.ndarray(int): dense binary vector of class labels, output of create_labels()
        kernel: str: kernel function to be used with classifier. [linear|poly|rbf|sigmoid]
    Outputs:
        sklearn.svm.SVC: classifier learnt from data
    """  
    clf = SVC(kernel=kernel)
    clf.fit(X_train, y_train)
    return clf
#%%
def evaluate_classifier(classifier, X_validation, y_validation):
    """ evaluates a classifier based on a supplied validation data
    Inputs:
        classifier: sklearn.svm.SVC: classifer to evaluate
        X_validation: scipy.sparse.csr.csr_matrix: sparse matrix of features
        y_validation: numpy.ndarray(int): dense binary vector of class labels
    Outputs:
        double: accuracy of classifier on the validation data
    """
    y_pred = classifier.predict(X_validation)
    return accuracy_score(y_validation, y_pred)

#%%
def classify_tweets(tfidf, classifier, unlabeled_tweets):
    """ predicts class labels for raw tweet text
    Inputs:
        tfidf: sklearn.feature_extraction.text.TfidfVectorizer: the TfidfVectorizer object used on training data
        classifier: sklearn.svm.SVC: classifier learned
        unlabeled_tweets: pd.DataFrame: tweets read from tweets_test.csv
    Outputs:
        numpy.ndarray(int): dense binary vector of class labels for unlabeled tweets
    """
    processed_unlabeled = process_all(unlabeled_tweets)
    X_unlabeled = tfidf.transform(processed_unlabeled['text'].tolist())
    return classifier.predict(X_unlabeled)