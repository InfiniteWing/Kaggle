import numpy as np
import pandas as pd
import xgboost as xgb
from nltk.tokenize import TweetTokenizer, word_tokenize
from sklearn.metrics import log_loss, f1_score
from sklearn.model_selection import KFold
from tqdm import tqdm
import nltk
import datetime

def get_pos_feature_df(df):
    pos_tags=[
        'UNKNOWN',
        'CC',
        'CD',
        'DT',
        'EX',
        'FW',
        'IN',
        'JJ',
        'JJR',
        'JJS',
        'LS',
        'MD',
        'NN',
        'NNS',
        'NNP',
        'NNPS',
        'PDT',
        'POS',
        'PRP',
        'PRP$',
        'RB',
        'RBR',
        'RBS',
        'RP',
        'SYM',
        'TO',
        'UH',
        'VB',
        'VBD',
        'VBG',
        'VBN',
        'VBP',
        'VBZ',
        'WDT',
        'WP',
        'WP$',
        'WRB'
    ]
    print(datetime.datetime.now())
    pos_tags_dict = {}
    for i, v in enumerate(pos_tags):
        pos_tags_dict[v] = i
    results = [[] for _ in range(len(pos_tags))]
    nltk_df = pd.DataFrame({'qid': df['qid'].values, 'question_text':df['question_text'].values})
    del df
    nltk_df['question_text'] = nltk_df['question_text'].apply(word_tokenize)
    nltk_df['question_text'] = nltk_df['question_text'].apply(nltk.pos_tag)
    for index, row in nltk_df.iterrows():
        tags = row['question_text']
        counts = [0 for _ in range(len(pos_tags))]
        for tag in tags:
            if(tag[1] in pos_tags_dict):
                counts[pos_tags_dict[tag[1]]] += 1
            else:
                counts[0] += 1
        total = sum(counts)
        counts = [a/total for a in counts]
        for i, r in enumerate(counts):
            results[i].append(r)
    nltk_pos_feature_df = pd.DataFrame({'qid': nltk_df['qid'].values})
    for i, v in enumerate(pos_tags):
        nltk_pos_feature_df[v] = results[i]
    print(nltk_pos_feature_df.head())
    print(datetime.datetime.now())
    return nltk_pos_feature_df

#nltk.download('averaged_perceptron_tagger')
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

nltk_pos_feature_df_train = get_pos_feature_df(train)
nltk_pos_feature_df_test = get_pos_feature_df(test)
nltk_pos_feature_df_train.to_csv('../input/nltk_pos_feature_df_train.csv', index=False, float_format = '%.6f')
nltk_pos_feature_df_test.to_csv('../input/nltk_pos_feature_df_test.csv', index=False, float_format = '%.6f')