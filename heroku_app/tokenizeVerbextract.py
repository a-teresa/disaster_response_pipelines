import nltk; # nltk.download('punkt','averaged_perceptron_tagger','wordnet','stopwords')
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import pandas as pd


def tokenize(text):
    """
    INPUT
    text: text to be stemmed and lemmatized
    OUTPUT:
    clean_tokens: text after remove stopwords, reduce to their stems,
                    lematize and normalize
    """
    #Normalize text
    text = re.sub(r"[^a-zA-Z0-9]"," ", text.lower())
    #tokenize text
    tokens = word_tokenize(text)
    #remove stop words
    tokens_stop = [w for w in tokens if w not in stopwords.words("english")]
    #reduce words to their stems
    tokens_stem = [PorterStemmer().stem(w) for w in tokens_stop]
    #initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    #iterate through each token
    clean_tokens=[]
    for tok in tokens_stem:
        #lemmatize, normalize case and remove white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


class VerbExtractor(BaseEstimator, TransformerMixin):
    def verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            try:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
            except:
                return False
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.verb)
        return pd.DataFrame(X_tagged)