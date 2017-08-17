# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import gc
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
nFold=10
# Any results you write to the current directory are saved as output.
def load_data(path_data,fold=0):
    '''
    --------------------------------order_product--------------------------------
    * Unique in order_id + product_id
    '''
    priors = pd.read_csv(path_data + 'order_products__prior.csv', 
                     dtype={
                            'order_id': np.int32,
                            'product_id': np.uint16,
                            'add_to_cart_order': np.int16,
                            'reordered': np.int8})
    train = pd.read_csv(path_data + 'order_products__train.csv', 
                    dtype={
                            'order_id': np.int32,
                            'product_id': np.uint16,
                            'add_to_cart_order': np.int16,
                            'reordered': np.int8})
    '''
    --------------------------------order--------------------------------
    * This file tells us which set (prior, train, test) an order belongs
    * Unique in order_id
    * order_id in train, prior, test has no intersection
    * this is the #order_number order of this user
    '''
    orders = pd.read_csv(path_data + 'orders_fold_{}.csv'.format(fold), 
                         dtype={
                                'order_id': np.int32,
                                'user_id': np.int64,
                                'eval_set': 'category',
                                'order_number': np.int16,
                                'order_dow': np.int8,
                                'order_hour_of_day': np.int8,
                                'days_since_prior_order': np.float32})
    #orders=orders.head(2000)
    #  order in prior, train, test has no duplicate
    #  order_ids_pri = priors.order_id.unique()
    #  order_ids_trn = train.order_id.unique()
    #  order_ids_tst = orders[orders.eval_set == 'test']['order_id'].unique()
    #  print(set(order_ids_pri).intersection(set(order_ids_trn)))
    #  print(set(order_ids_pri).intersection(set(order_ids_tst)))
    #  print(set(order_ids_trn).intersection(set(order_ids_tst)))

    '''
    --------------------------------product--------------------------------
    * Unique in product_id
    '''
    products = pd.read_csv(path_data + 'products.csv')
    aisles = pd.read_csv(path_data + "aisles.csv")
    departments = pd.read_csv(path_data + "departments.csv")
    sample_submission = pd.read_csv(path_data + "sample_submission.csv")
    
    return priors, train, orders, products, aisles, departments, sample_submission

class tick_tock:
    def __init__(self, process_name, verbose=1):
        self.process_name = process_name
        self.verbose = verbose
    def __enter__(self):
        if self.verbose:
            print(self.process_name + " begin ......")
            self.begin_time = time.time()
    def __exit__(self, type, value, traceback):
        if self.verbose:
            end_time = time.time()
            print(self.process_name + " end ......")
            print('time lapsing {0} s \n'.format(end_time - self.begin_time))
            
def ka_add_groupby_features_1_vs_n(df, group_columns_list, agg_dict, only_new_feature=True):
    '''Create statistical columns, group by [N columns] and compute stats on [N column]

       Parameters
       ----------
       df: pandas dataframe
          Features matrix
       group_columns_list: list_like
          List of columns you want to group with, could be multiple columns
       agg_dict: python dictionary

       Return
       ------
       new pandas dataframe with original columns and new added columns

       Example
       -------
       {real_column_name: {your_specified_new_column_name : method}}
       agg_dict = {'user_id':{'prod_tot_cnts':'count'},
                   'reordered':{'reorder_tot_cnts_of_this_prod':'sum'},
                   'user_buy_product_times': {'prod_order_once':lambda x: sum(x==1),
                                              'prod_order_more_than_once':lambda x: sum(x==2)}}
       ka_add_stats_features_1_vs_n(train, ['product_id'], agg_dict)
    '''
    with tick_tock("add stats features"):
        try:
            if type(group_columns_list) == list:
                pass
            else:
                raise TypeError(k + "should be a list")
        except TypeError as e:
            print(e)
            raise

        df_new = df.copy()
        grouped = df_new.groupby(group_columns_list)

        the_stats = grouped.agg(agg_dict)
        the_stats.columns = the_stats.columns.droplevel(0)
        the_stats.reset_index(inplace=True)
        if only_new_feature:
            df_new = the_stats
        else:
            df_new = pd.merge(left=df_new, right=the_stats, on=group_columns_list, how='left')

    return df_new

def ka_add_groupby_features_n_vs_1(df, group_columns_list, target_columns_list, methods_list, keep_only_stats=True, verbose=1):
    '''Create statistical columns, group by [N columns] and compute stats on [1 column]

       Parameters
       ----------
       df: pandas dataframe
          Features matrix
       group_columns_list: list_like
          List of columns you want to group with, could be multiple columns
       target_columns_list: list_like
          column you want to compute stats, need to be a list with only one element
       methods_list: list_like
          methods that you want to use, all methods that supported by groupby in Pandas

       Return
       ------
       new pandas dataframe with original columns and new added columns

       Example
       -------
       ka_add_stats_features_n_vs_1(train, group_columns_list=['x0'], target_columns_list=['x10'])
    '''
    with tick_tock("add stats features", verbose):
        dicts = {"group_columns_list": group_columns_list , "target_columns_list": target_columns_list, "methods_list" :methods_list}

        for k, v in dicts.items():
            try:
                if type(v) == list:
                    pass
                else:
                    raise TypeError(k + "should be a list")
            except TypeError as e:
                print(e)
                raise

        grouped_name = ''.join(group_columns_list)
        target_name = ''.join(target_columns_list)
        combine_name = [[grouped_name] + [method_name] + [target_name] for method_name in methods_list]

        df_new = df.copy()
        grouped = df_new.groupby(group_columns_list)

        the_stats = grouped[target_name].agg(methods_list).reset_index()
        the_stats.columns = [grouped_name] + \
                            ['_%s_%s_by_%s' % (grouped_name, method_name, target_name) \
                             for (grouped_name, method_name, target_name) in combine_name]
        if keep_only_stats:
            return the_stats
        else:
            df_new = pd.merge(left=df_new, right=the_stats, on=group_columns_list, how='left')
        return df_new
        
def ka_add_groupby_features_n_vs_1(df, group_columns_list, agg_dict, keep_only_stats=True, verbose=1):
    '''Create statistical columns, group by [N columns] and compute stats on [1 column]

       Parameters
       ----------
       df: pandas dataframe
          Features matrix
       group_columns_list: list_like
          List of columns you want to group with, could be multiple columns
       target_columns_list: list_like
          column you want to compute stats, need to be a list with only one element
       methods_list: list_like
          methods that you want to use, all methods that supported by groupby in Pandas

       Return
       ------
       new pandas dataframe with original columns and new added columns

       Example
       -------
       ka_add_stats_features_n_vs_1(train, group_columns_list=['x0'], target_columns_list=['x10'])
    '''
    with tick_tock("add stats features", verbose):
        df_new = df.copy()
        grouped = df_new.groupby(group_columns_list)

        the_stats = grouped.agg(agg_dict).reset_index()
        if keep_only_stats:
            return the_stats
        else:
            df_new = pd.merge(left=df_new, right=the_stats, on=group_columns_list, how='left')
        return df_new
        
        
path_data = '../input/'

for fold in [5,6,7,8,9]:
    print('Start fold {} / {}'.format(fold+1,nFold))
    priors, train, orders, products, aisles, departments, sample_submission = load_data(path_data,fold)

    # Products information ----------------------------------------------------------------
    # add order information to priors set
    priors_orders_detail = orders.merge(right=priors, how='inner', on='order_id')

    # create new variables
    # _user_buy_product_times: 用户是第几次购买该商品
    priors_orders_detail.loc[:,'_user_buy_product_times'] = priors_orders_detail.groupby(['user_id', 'product_id']).cumcount() + 1
    # _prod_tot_cnts: 该商品被购买的总次数,表明被喜欢的程度
    # _reorder_tot_cnts_of_this_prod: 这件商品被再次购买的总次数
    ### 我觉得下面两个很不好理解，考虑改变++++++++++++++++++++++++++
    # _prod_order_once: 该商品被购买一次的总次数
    # _prod_order_more_than_once: 该商品被购买一次以上的总次数
    agg_dict = {'user_id':{'_prod_tot_cnts':'count'}, 
                'reordered':{'_prod_reorder_tot_cnts':'sum'}, 
                '_user_buy_product_times': {'_prod_buy_first_time_total_cnt':lambda x: sum(x==1),
                                            '_prod_buy_second_time_total_cnt':lambda x: sum(x==2)}}
    prd = ka_add_groupby_features_1_vs_n(priors_orders_detail, ['product_id'], agg_dict)

    # _prod_reorder_prob: 这个指标不好理解
    # _prod_reorder_ratio: 商品复购率
    prd['_prod_reorder_prob'] = prd._prod_buy_second_time_total_cnt / prd._prod_buy_first_time_total_cnt
    prd['_prod_reorder_ratio'] = prd._prod_reorder_tot_cnts / prd._prod_tot_cnts
    prd['_prod_reorder_times'] = 1 + prd._prod_reorder_tot_cnts / prd._prod_buy_first_time_total_cnt

    # _user_total_orders: 用户的总订单数
    # 可以考虑加入其它统计指标++++++++++++++++++++++++++
    # _user_sum_days_since_prior_order: 距离上次购买时间(和),这个只能在orders表里面计算，priors_orders_detail不是在order level上面unique
    # _user_mean_days_since_prior_order: 距离上次购买时间(均值)
    agg_dict_2 = {'order_number':{'_user_total_orders':'max'},
                  'days_since_prior_order':{'_user_sum_days_since_prior_order':'sum', 
                                            '_user_mean_days_since_prior_order': 'mean'}}
    users = ka_add_groupby_features_1_vs_n(orders[orders.eval_set == 'prior'], ['user_id'], agg_dict_2)

    # _user_reorder_ratio: reorder的总次数 / 第一单后买后的总次数
    # _user_total_products: 用户购买的总商品数
    # _user_distinct_products: 用户购买的unique商品数
    agg_dict_3 = {'reordered':
                  {'_user_reorder_ratio': 
                   lambda x: sum(priors_orders_detail.ix[x.index,'reordered']==1)/
                             sum(priors_orders_detail.ix[x.index,'order_number'] > 1)},
                  'product_id':{'_user_total_products':'count', 
                                '_user_distinct_products': lambda x: x.nunique()}}
    us = ka_add_groupby_features_1_vs_n(priors_orders_detail, ['user_id'], agg_dict_3)
    users = users.merge(us, how='inner')

    # 平均每单的商品数
    # 每单中最多的商品数，最少的商品数++++++++++++++
    users['_user_average_basket'] = users._user_total_products / users._user_total_orders

    us = orders[orders.eval_set != "prior"][['user_id', 'order_id', 'eval_set', 'days_since_prior_order']]
    us.rename(index=str, columns={'days_since_prior_order': 'time_since_last_order'}, inplace=True)

    users = users.merge(us, how='inner')

    # 这里应该还有很多变量可以被添加
    # _up_order_count: 用户购买该商品的次数
    # _up_first_order_number: 用户第一次购买该商品所处的订单数
    # _up_last_order_number: 用户最后一次购买该商品所处的订单数
    # _up_average_cart_position: 该商品被添加到购物篮中的平均位置
    agg_dict_4 = {'order_number':{'_up_order_count': 'count', 
                                  '_up_first_order_number': 'min', 
                                  '_up_last_order_number':'max'}, 
                  'add_to_cart_order':{'_up_average_cart_position': 'mean'}}

    data = ka_add_groupby_features_1_vs_n(df=priors_orders_detail, 
                                                          group_columns_list=['user_id', 'product_id'], 
                                                          agg_dict=agg_dict_4)

    data = data.merge(prd, how='inner', on='product_id').merge(users, how='inner', on='user_id')
    # 该商品购买次数 / 总的订单数
    # 最近一次购买商品 - 最后一次购买该商品
    # 该商品购买次数 / 第一次购买该商品到最后一次购买商品的的订单数
    data['_up_order_rate'] = data._up_order_count / data._user_total_orders
    data['_up_order_since_last_order'] = data._user_total_orders - data._up_last_order_number
    data['_up_order_rate_since_first_order'] = data._up_order_count / (data._user_total_orders - data._up_first_order_number + 1)

    # add user_id to train set
    train = train.merge(right=orders[['order_id', 'user_id']], how='left', on='order_id')
    data = data.merge(train[['user_id', 'product_id', 'reordered']], on=['user_id', 'product_id'], how='left')

    data.to_csv('features/original_features_{}.csv'.format(fold),index=False)
    
    new_feature_df = pd.read_csv("features/departments_aisles_features.csv")
    data=data.merge(new_feature_df,on=['user_id', 'product_id'], how='left')
    
    # release Memory
    # del train, prd, users
    # gc.collect()
    # release Memory
    del priors_orders_detail, orders
    gc.collect()

    import xgboost

    train = data.loc[data.eval_set == "train",:]
    train.drop(['eval_set', 'user_id', 'product_id', 'order_id'], axis=1, inplace=True)
    train.loc[:, 'reordered'] = train.reordered.fillna(0)

    X_test = data.loc[data.eval_set == "test",:]

    # subsample 让training时间更短
    #X_train, X_val, y_train, y_val = train_test_split(train.drop('reordered', axis=1), train.reordered,
    #                                                    test_size=0.9, random_state=42)
    X_train=train.drop('reordered', axis=1)
    y_train=train.reordered
    d_train = xgboost.DMatrix(X_train, y_train)
    xgb_params = {
        "objective"         : "reg:logistic"
        ,"eval_metric"      : "logloss"
        ,"eta"              : 0.1
        ,"max_depth"        : 7
        ,"min_child_weight" :10
        ,"gamma"            :0.70
        ,"subsample"        :0.76
        ,"colsample_bytree" :0.95
        ,"alpha"            :2e-05
        ,"lambda"           :10
    }

    watchlist= [(d_train, "train")]
    bst = xgboost.train(params=xgb_params, dtrain=d_train, num_boost_round=800, evals=watchlist, verbose_eval=10)
    xgboost.plot_importance(bst)

    d_test = xgboost.DMatrix(X_test.drop(['eval_set', 'user_id', 'order_id', 'reordered', 'product_id'], axis=1))
    X_test.loc[:,'reordered'] = (bst.predict(d_test)).astype(float)
    preds=X_test['reordered'].values
    order_ids=X_test['order_id'].values
    product_ids=X_test['product_id'].values
    order2preds={}
    product_id_to_pred=[]
    
    for i in range(len(preds)):
        order_id=order_ids[i]
        product_id=product_ids[i]
        pred=preds[i]
        if(order_id not in order2preds):
            order2preds[order_id]={}
        order2preds[order_id][product_id]=pred
    final_preds=[]
    final_order_ids=list(order2preds.keys())
    for order_id in order2preds:
        product2preds=order2preds[order_id]
        product2preds=sorted(product2preds.items(), key=lambda x:x[1],reverse=True)
        probabilities=list(product2preds.values())
        products=list(product2preds.keys())

        opt=F1Optimizer.maximize_expectation(thresholds)
        best_k=opt[0]
        pred=products[:best_k]
        if(opt[1]):
            pred.append('None')
        final_preds.append(' '.join(pred))
    