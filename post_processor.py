# -*- coding: utf-8 -*-
"""

@author: NSK
"""
import pandas as pd
import pickle

# Gensim
import gensim



def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


def main():
    model_mallet = gensim.models.LdaModel.load("lda_model_from_mallet.model")
    infile = open("corpus.pkl",'rb')
    corpus = pickle.load(infile)
    infile.close()
    
    infile = open("data.pkl",'rb')
    data = pickle.load(infile)
    infile.close()
    
    DF = format_topics_sentences(lda_model = model_mallet, corpus = corpus, texts = data)
    DF.dropna(inplace = True)
    
    
    MAPPING = {8: 'DEBT',
                7: 'CUSTOMER CARE',
                13: 'VERIFICATION',
                14: 'SECURITY',
                9: 'NET BANKING',
                11: 'STATEMENT',
                2: 'TRANSACTION',
                0: 'CREDIT CARD',
                1: 'DISPUTE',
                12: 'THEFT/FRAUD',
                3: 'HOME LOAN',
                4: 'VEHICLE LOAN',
                5: 'E MAIL',
                6: 'LOANS',
                10: 'BANKING'}
    
    DF['Topics'] = DF["Dominant_Topic"].map(MAPPING)
    DF.to_csv("Final_Output.csv")
    
if __name__ == "__main__":
    main()
    