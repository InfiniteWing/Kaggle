import numpy as np
import pandas as pd
import xgboost as xgb
from nltk.tokenize import TweetTokenizer, word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import log_loss, f1_score
from sklearn.model_selection import KFold
from tqdm import tqdm
import nltk
def main():
    #nltk.download('averaged_perceptron_tagger')
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    sub = pd.read_csv('../input/sample_submission.csv')
    #train = train.head(1000)
    #test = test.head(1000)
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
    pos_tags_dict = {}
    for i, v in enumerate(pos_tags):
        pos_tags_dict[v] = i
    results = [[] for _ in range(len(pos_tags))]
    for index, row in tqdm(train.iterrows(), total=train.shape[0]):
        question_text = row['question_text']
        text = word_tokenize(question_text)
        counts = [0 for _ in range(len(pos_tags))]
        for tag in nltk.pos_tag(text):
            if(tag[1] in pos_tags_dict):
                counts[pos_tags_dict[tag[1]]] += 1
            else:
                counts[0] += 1
        total = sum(counts)
        counts = [a/total for a in counts]
        for i, r in enumerate(counts):
            results[i].append(r)
    nltk_pos_features = pd.DataFrame({'qid': train['qid'].values})
    for i, v in enumerate(pos_tags):
        nltk_pos_features[v] = results[i]
    print(nltk_pos_features.head())
    nltk_pos_features.to_csv('../input/nltk_pos_features.csv', index=False, float_format = '%.6f')
    #input('Wait')
    max_features = 120000
    tk = Tokenizer(lower = True, filters='', num_words=max_features)
    full_text = list(train['question_text'].values) + list(test['question_text'].values)
    tk.fit_on_texts(full_text)
    train_tokenized = tk.texts_to_sequences(train['question_text'].fillna('ghiuhihiuhiy'))
    test_tokenized = tk.texts_to_sequences(test['question_text'].fillna('ghiuhihiuhiy'))



    max_len = 50
    maxlen = 50
    X_train = pad_sequences(train_tokenized, maxlen = max_len)
    #x_test = pad_sequences(test_tokenized, maxlen = max_len)

    for col_index in range(X_train.shape[1]):
        print(len(X_train[:,col_index]))
        nltk_pos_features['pad_sequences'+str(col_index)] = X_train[:,col_index]
    print(nltk_pos_features.head())
    y_train = train['target'].values


    X_train = np.array(nltk_pos_features.drop(['qid'], axis=1))

    K = 5
    kf = KFold(n_splits = K, random_state = 3228, shuffle = True)
    fold = 1
    y_preds = None
    valid_scores = 0.0
    for train_index, test_index in kf.split(X_train):
        print("Start fold", fold)
        train_X, valid_X = X_train[train_index], X_train[test_index]
        train_y, valid_y = y_train[train_index], y_train[test_index]
        
        d_train = xgb.DMatrix(train_X, train_y)
        d_valid = xgb.DMatrix(valid_X, valid_y)
        #d_test = xgb.DMatrix(x_test)
        
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        
        # xgboost params
        xgb_params = {
            'eta': 0.3,
            'max_depth': 6,
            'subsample': 0.8,
            'objective': 'reg:logistic',
            'eval_metric': 'logloss',
            'seed': 3228,
            'silent': 1
        }
        model = xgb.train(xgb_params, d_train, 200, watchlist, early_stopping_rounds=20, verbose_eval=20)
        valid_pred = model.predict(d_valid)
        #y_pred = model.predict(d_test)
        
        best_score = -1
        best_threshold = -1
        for t in range(100):
            threshold = t*0.01 + 0.01
            valid_score = f1_score(valid_y, valid_pred>threshold)
            if(best_score < valid_score):
                best_score = valid_score
                best_threshold = threshold
        print('Valid F1 = ', best_score, best_threshold)
        valid_scores += best_score
        #if(fold == 1):
        #    y_preds = y_pred
        #else:
        #    y_preds += y_pred
        #fold += 1
        
    valid_scores /= K
    #y_preds /= K
    print('Average valid F1 =', valid_scores)
    #output = pd.DataFrame({'target': y_preds})
    #output.to_csv('xgb_pred.csv', index=False, float_format = '%.5f')
main()
