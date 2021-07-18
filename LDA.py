# -*- coding: utf-8 -*-
"""
@author: NSK
"""

import nltk 
import re
import numpy as np
import pandas as pd
import pickle

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models.wrappers import LdaMallet

# spacy for lemmatization
import spacy

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import time

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])


# Mallet 
mallet_path = r'C:\Mallet\bin\mallet'
csv_file_path = "complaints.csv"

def read_data(csv_file_path):
    DF = pd.read_csv(csv_file_path, usecols = ["Consumer complaint narrative"])
    DF.rename(columns = {'Consumer complaint narrative':'COMPLAINTS'}, inplace = True)
    DF['COMPLAINTS'].dropna(inplace=True)
    DF['COMPLAINTS'] = DF['COMPLAINTS'].astype(str)
    
    return DF

    
def get_complaints(DF):
    
    raw_data = DF.copy()
    # raw_data = raw_data.sample(n=50000)
    raw_data = raw_data['COMPLAINTS'].values.tolist()
    
    # Clean the text
    data = [str(sent) for sent in raw_data]
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
    data = [re.sub('\s+', ' ', sent) for sent in data]
    data = [re.sub("\'", "", sent) for sent in data]
    data = [re.sub("X","",sent) for sent in data]
    data = [sent for sent in data if sent != "nan"]
    
    filename = 'data.pkl'
    outfile = open(filename,'wb')
    pickle.dump(data,outfile)
    outfile.close()
    
    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
            
            
    data_words = list(sent_to_words(data))
    data_words = [i for i in data_words if len(i)>=10 and len(i) < 300]
    
    doc_lens = [len(d) for d in data_words]
    print("Document word count")
    print("Min:",(round(np.min(doc_lens))))
    print("Mean:",(round(np.mean(doc_lens))))
    print("Median:",(round(np.median(doc_lens))))
    print("Max:",(round(np.max(doc_lens))))
                
    
            
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
    
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    
    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]
    
    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]
    
    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out
  
    data_words_nostops = remove_stopwords(data_words)
    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)
    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    data_lemmatized = [i for i in data_lemmatized if len(i)!=0]
    
    return data_lemmatized

def generate_corpus(cleaned_data):
    # Create Dictionary
    id2word = corpora.Dictionary(cleaned_data)
    texts = cleaned_data
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    
    return id2word,corpus


def build_mallet_model(mallet_path, id2word, corpus, data_lemmatized):
    ldamallet = LdaMallet(mallet_path, 
                      corpus=corpus, 
                      num_topics=15, 
                      id2word=id2word,
                      workers = 3)
    
    return ldamallet
    
def main():
    
    # Get all the complaints texts from input
    DF = read_data(csv_file_path)
    
    # Clean Text
    cleanedData = get_complaints(DF)
    
    # Generate corpus from cleaned data
    id2word, corpus = generate_corpus(cleanedData)
    
    # Build LDA using MALLET
    mallet_out = build_mallet_model(mallet_path, id2word, corpus, cleanedData)
    lda_model_from_mallet = LdaMallet.malletmodel2ldamodel(mallet_out,
                                                           gamma_threshold=0.001, 
                                                           iterations=50)
    lda_model_from_mallet.save("lda_model_from_mallet.model")
    
   
    filename = 'corpus.pkl'
    outfile = open(filename,'wb')
    pickle.dump(corpus,outfile)
    outfile.close()
       
    
if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Processing time: ", time.time() - start_time)
    
    
    

