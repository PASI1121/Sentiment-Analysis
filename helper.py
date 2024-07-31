import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
import sklearn


from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

import pickle
with open('static/model/model.pickle','rb')as f:
    model=pickle.load(f)

with open ('static/model/corpora/stopwords/english','r') as file:
    sw = file.read().splitlines()

vocab = pd.read_csv('static/model/vocabulary.txt',header=None)
tokens= vocab[0].tolist()


punctuation_pattern = r'[^\w\s]' 
number_pattern = r'\d+'

stop_words = set(stopwords.words('english'))
def preprocessing(text):
    data = pd.DataFrame([text],columns=['tweet'])
    data['tweet'] = data['tweet'].apply(lambda x: ' '.join(x.lower() for x in x.split()))
    data['tweet'] = data['tweet'].apply(lambda x: ' '.join(re.sub(r'^https?:\/\/.*[\r\n]*', '',x,flags=re.MULTILINE) for x in x.split()))
    data['tweet'] = data['tweet'].str.replace(punctuation_pattern, '', regex=True)
    data['tweet'] = data['tweet'].str.replace(number_pattern, '', regex=True)
    data['tweet'] = data['tweet'].apply(lambda x: ' '.join(x for x in x.split() if x not in sw))
    data['tweet'] = data['tweet'].apply(lambda x: ' '.join(stemmer.stem(x) for x in x.split()))
    return data['tweet']

def vectorizer(ds):
    vectorized_lst=[]
    for sentence in ds:
        sentence_lst = np.zeros(len(tokens))
        for i in range(len(tokens)):
            if tokens[i] in sentence.split():
                sentence_lst[i] =1
        vectorized_lst.append(sentence_lst)
    vectorized_lst_new = np.asarray(vectorized_lst,dtype=np.float32)
    if len(ds)==1:
        return vectorized_lst_new[0]
    return vectorized_lst_new

def get_prediction(vectorized_txt):
    prediction = model.predict([vectorized_txt])
    if prediction == 1:
        return 'negative'
    else:
        return 'Positive'

