import nltk
nltk.download('punkt','averaged_perceptron_tagger','wordnet','stopwords')
import re
import sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib


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
    y=df.iloc[:,4:]
    y=y.replace(2,1)
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
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    #iterate through each token
    clean_tokens=[]
    for tok in tokens_stop:
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
    """
    build_model() creates a classifier model using the pipeline
    INPUT:
    none
    OUTPUT:
    classifier model
    """
    pipeline = Pipeline([
    ('features', FeatureUnion([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
        ('verb', VerbExtractor())
    ])),
    ('clf',MultiOutputClassifier(RandomForestClassifier(random_state=42)))
])
    parameters = {
        'clf__estimator__n_estimators':[250],
        'features__text_pipeline__vect__max_df':[1.0]
    }
    #create grid_search
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model
    INPUT:
    model: the trained model
    X_test: unseen data to test model
    Y_test: true values
    category_names: column names
    OUTPUT:
    print model prediction report
    """
    y_pred=model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names = category_names))
    pass


def save_model(model, model_filepath):
    """
    save model
    INPUT:
    model: the trained model
    model_filepath: path to store the model
    OUTPUT:
    none
    """
    joblib.dump(model.best_estimator_, model_filepath, compress=3)
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