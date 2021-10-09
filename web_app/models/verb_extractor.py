from sklearn.base import BaseEstimator, TransformerMixin
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