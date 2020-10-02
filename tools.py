from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from nltk.stem import SnowballStemmer
from stop_words import get_stop_words
from nltk.corpus import stopwords
import pandas as pd
import pickle

FR = SnowballStemmer('french')
MY_STOP_WORD_LIST = get_stop_words('french')
FINAL_STOPWORDS_LIST = stopwords.words('french')

S_W = list(set(FINAL_STOPWORDS_LIST + MY_STOP_WORD_LIST))
S_W = [elem.lower() for elem in S_W]

CLS = pickle.load(open("./data/model.pkl", "rb"))
LOADED_VEC = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open("./data/base.pkl", "rb")))

VECTORIZER = TfidfVectorizer()

DF=pd.read_csv('corpus.csv')


def vectorisation(text):
    VECTORIZER.fit(text)
    return VECTORIZER.transform(text)

def labelisation():
    DF.loc[(DF.rating < 3),'rating']=0
    DF.loc[(DF.rating > 3),'rating']=1
    return DF['rating']