#import libraries
import sys
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag, ne_chunk
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer
import pandas as pd
from sqlalchemy import create_engine
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
    """
    INPUT:
    database_filepath - path to database
    OUTPUT
    X - pandas dataframe with messages
    y - pandas dataframes with repective categories
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * FROM disaster_response", engine)
    X = df.message.values
    #exclude id, message and genre from y variable
    y=df.iloc[:,3:]
    category_names = y.columns
    return X,y, category_names



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

#verb extraction
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


def build_model():
    pipeline = Pipeline([
    ('features', FeatureUnion([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
        ('verb', VerbExtractor())
    ])),
    ('clf',MultiOutputClassifier(RandomForestClassifier()))
])
    parameters = {
        'clf__estimator__n_estimators':[200],
        'features__text_pipeline__vect__max_df':(0.75,1.0)
    }
    #create grid_search
    cv = GridSearchCV(pipeline, param_grid = parameters, verbose = 3)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    model = build_model()
    y_pred=model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names = category_names))
    pass


def save_model(model, model_filepath):
    file = 'disaster_response.pkl'
    pickle.dump(model, open(file, 'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()