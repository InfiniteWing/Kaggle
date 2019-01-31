import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, f1_score

def main():
    nltk_pos_feature_df_train = pd.read_csv("../input/nltk_pos_feature_df_train_all.csv")
    nltk_pos_feature_df_test = pd.read_csv("../input/nltk_pos_feature_df_test_all.csv")
    train = pd.read_csv("../input/train.csv")
    sub = pd.read_csv('../input/sample_submission.csv')
    
    # Origin NN Score
    '''
    valid_y = train['target'].values
    valid_pred = nltk_pos_feature_df_train['nn_preds']
    
    best_score = -1
    logloss_score = log_loss(valid_y, valid_pred)
    best_threshold = -1
    for t in range(100):
        threshold = t*0.01 + 0.01
        valid_score = f1_score(valid_y, valid_pred>threshold)
        if(best_score < valid_score):
            best_score = valid_score
            best_threshold = threshold
    print('Valid F1 = ', best_score, best_threshold)
    sub['prediction'] = nltk_pos_feature_df_test['nn_preds'] > best_threshold
    sub.to_csv("submission_nn.csv", index=False)
    return 
    '''
    drop_list = ['pad_sequences{}'.format(i) for i in range(72)]
    drop_list = []
    X_train = np.array(nltk_pos_feature_df_train.drop(['qid'] + drop_list, axis=1))
    X_test = np.array(nltk_pos_feature_df_test.drop(['qid'] + drop_list, axis=1))
    y_train = train['target'].values

    NFold = 4
    splits = list(StratifiedKFold(n_splits=NFold, shuffle=True, random_state=10).split(X_train, y_train))
    
    lgb_params = {
        'learning_rate': 0.05,
        'application': 'binary',
        'max_depth': 6,
        'num_leaves':2**6,
        'max_bin': 1024*2,
        'verbosity': -1,
        'metric': 'binary_logloss'
    }
    
    y_preds = None
    threshold_all = 0
    f1_all = 0
    for i, (train_index, test_index) in enumerate(splits):
        print("Start fold", i)
        train_X, valid_X = X_train[train_index], X_train[test_index]
        train_y, valid_y = y_train[train_index], y_train[test_index]
        
        d_train = lgb.Dataset(train_X, label=train_y)
        d_valid = lgb.Dataset(valid_X, label=valid_y)
        watchlist = [d_train, d_valid]

        model = lgb.train(lgb_params, train_set=d_train, num_boost_round=300, valid_sets=watchlist, early_stopping_rounds=30, verbose_eval=10) 
        
        valid_pred = model.predict(valid_X)
        y_pred = model.predict(X_test)
        
        best_score = -1
        logloss_score = log_loss(valid_y, valid_pred)
        best_threshold = -1
        for t in range(100):
            threshold = t*0.01 + 0.01
            valid_score = f1_score(valid_y, valid_pred>threshold)
            if(best_score < valid_score):
                best_score = valid_score
                best_threshold = threshold
        print('Valid F1 = ', best_score, best_threshold)
        f1_all += best_score
        threshold_all += best_threshold
        print('Valid logloss = ', logloss_score)
        #valid_scores += best_score
        if(i == 0):
            y_preds = y_pred
        else:
            y_preds += y_pred
    threshold_all /= NFold
    y_preds /= NFold
    f1_all /= NFold
    print('Mean valid F1 = ', f1_all)

    sub['prediction'] = y_preds > threshold_all
    sub.to_csv("submission_lgbm.csv", index=False)
    
    
def XGBModel():
    nltk_pos_feature_df_train = pd.read_csv("../input/nltk_pos_feature_df_train_all.csv")
    nltk_pos_feature_df_test = pd.read_csv("../input/nltk_pos_feature_df_test_all.csv")
    train = pd.read_csv("../input/train.csv")
    
    X_train = np.array(nltk_pos_feature_df_train.drop(['qid'], axis=1))
    X_test = np.array(nltk_pos_feature_df_test.drop(['qid'], axis=1))
    y_train = train['target'].values

    NFold = 4
    splits = list(StratifiedKFold(n_splits=NFold, shuffle=True, random_state=10).split(X_train, y_train))
    
    lgb_params = {
        'learning_rate': 0.05,
        'application': 'binary',
        'max_depth': 7,
        'num_leaves': 256,
        'verbosity': -1,
        'metric': 'binary_logloss'
    }
    
    y_preds = None
    threshold_all = 0
    for i, (train_index, test_index) in enumerate(splits):
        print("Start fold", i)
        train_X, valid_X = X_train[train_index], X_train[test_index]
        train_y, valid_y = y_train[train_index], y_train[test_index]
        
        d_train = xgb.DMatrix(train_X, train_y)
        d_valid = xgb.DMatrix(valid_X, valid_y)
        d_test = xgb.DMatrix(X_test)
        
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        
        # xgboost params
        xgb_params = {
            'eta': 0.03,
            'max_depth': 8,
            'subsample': 0.8,
            'objective': 'reg:logistic',
            'eval_metric': 'logloss',
            'seed': 3228,
            'silent': 1
        }
        model = xgb.train(xgb_params, d_train, 480, watchlist, early_stopping_rounds=30, verbose_eval=30)
        valid_pred = model.predict(d_valid)
        y_pred = model.predict(d_test)
        
        best_score = -1
        logloss_score = log_loss(valid_y, valid_pred)
        best_threshold = -1
        for t in range(100):
            threshold = t*0.01 + 0.01
            valid_score = f1_score(valid_y, valid_pred>threshold)
            if(best_score < valid_score):
                best_score = valid_score
                best_threshold = threshold
        print('Valid F1 = ', best_score, best_threshold)
        threshold_all += best_threshold
        print('Valid logloss = ', logloss_score)
        #valid_scores += best_score
        if(i == 0):
            y_preds = y_pred
        else:
            y_preds += y_pred
    threshold_all /= NFold
    y_preds /= NFold

    sub['prediction'] = y_preds > threshold_all
    sub.to_csv("submission.csv", index=False)
    
main()