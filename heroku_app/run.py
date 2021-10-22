import nltk; # nltk.download('punkt','averaged_perceptron_tagger','wordnet','stopwords')
import json
import plotly
import pandas as pd
from tokenizeVerbextract import tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
import joblib




app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../app/DisasterResponse.db')
df = pd.read_sql("SELECT * FROM disaster_response", engine)


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

@app.before_first_request
# load model
def load_models():
    global model
    model = joblib.load("classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    global df
    # find top 5 and least 5
    df= df.replace(2,1) #replace all 2 with 1
    df = df.iloc[:,4:] #select only wanted columns
    count_val = (df.apply(pd.value_counts).transpose()).iloc[:, 1:2] #count and select only true values
    count_val['count'] = count_val.iloc[:,0] #name column
    #find the top 5 occurrences
    top_five = count_val.sort_values(by=['count'], ascending=False)[0:5]
    top_freq = top_five['count']
    top_names = list(top_freq.index)
    
    #find the top 5 occurrences
    least_five = count_val.sort_values(by=['count'], ascending=True)[0:5]
    least_freq = least_five['count']
    least_names = list(least_freq.index)
    
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=top_names,
                    y=top_freq
                )
            ],

            'layout': {
                'title': 'Top 5 Common Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=least_names,
                    y=least_freq
                )
            ],

            'layout': {
                'title': 'Least 5 Common Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


#def main():
 #   app.run(host='0.0.0.0', port=3001, debug=True)


#if __name__ == '__main__':
 #   main()