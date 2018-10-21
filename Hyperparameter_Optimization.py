import numpy as np
import pandas as pd
import gc
import time
import sys
import itertools
from contextlib import contextmanager
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from hyperopt import STATUS_OK
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def missing(df):
    num = df.isnull().sum()
    percent = num / len(df) * 100
    table = pd.concat([num, percent], axis=1)
    table = table.rename({0: 'Missing Values', 1: "% of Total Values"}, axis=1)
    table = table[~(table["% of Total Values"] == 0)].sort_values("% of Total Values", ascending=False).round(1)
    return table

def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def applications_train_test(num_rows=None, nan_as_category=False):
    df = pd.read_csv("../input/application_train.csv", nrows=num_rows)

    for bin_feature in ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])

    df["REGION_RATING_CLIENT_W_CITY"] = df["REGION_RATING_CLIENT_W_CITY"].astype("object")
    df["REGION_RATING_CLIENT"] = df["REGION_RATING_CLIENT"].astype("object")  
    df["HOUR_APPR_PROCESS_START"] = df["HOUR_APPR_PROCESS_START"].astype("object")

    df, cat_cols = one_hot_encoder(df, nan_as_category)
    df["DAYS_EMPLOYED"].replace(365243, np.nan, inplace=True)
    df["AMT_INCOME_TOTAL"].replace(117000000.0, np.nan, inplace=True)
    df["OBS_30_CNT_SOCIAL_CIRCLE"].replace(348.0, np.nan, inplace=True)
    df["OBS_60_CNT_SOCIAL_CIRCLE"].replace(344.0, np.nan, inplace=True)
    df["AMT_REQ_CREDIT_BUREAU_QRT"].replace(261.0, np.nan, inplace=True)

    df["DAYS_EMPLOYED_PERC"] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df["AMT_CREDIT"]
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    df['ANNUITY_INCOME_PERC'] = df["AMT_ANNUITY"] / df['AMT_INCOME_TOTAL']
    df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    return df


def bureau_and_balance(num_rows=None, nan_as_category=True):
    bureau = pd.read_csv("../input/bureau.csv", nrows=num_rows)
    bb = pd.read_csv("../input/bureau_balance.csv", nrows=num_rows)
    
    bureau.loc[bureau.DAYS_CREDIT_ENDDATE < -20000, "DAYS_CREDIT_ENDDATE"] = np.nan
    bureau.loc[bureau.DAYS_ENDDATE_FACT < -4000, "DAYS_ENDDATE_FACT"] = np.nan
    bureau.loc[bureau.AMT_CREDIT_MAX_OVERDUE > 10000000, "AMT_CREDIT_MAX_OVERDUE"] = np.nan
    bureau.loc[bureau.AMT_CREDIT_SUM > 10000000, "AMT_CREDIT_SUM"] = np.nan
    bureau.loc[bureau.AMT_CREDIT_SUM_DEBT < 0, "AMT_CREDIT_SUM_DEBT"] = 0
    bureau.loc[bureau.AMT_CREDIT_SUM_DEBT > 50000000, "AMT_CREDIT_SUM_DEBT"] = np.nan
    bureau.loc[bureau.AMT_ANNUITY > 10000000, "AMT_ANNUITY"] = np.nan

    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby("SK_ID_BUREAU").agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }

    cat_aggregations = {}
    for cat in bureau_cat:
        cat_aggregations[cat] = ['mean']
    for cat in bb_cat:
        cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby("SK_ID_CURR").agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    active = bureau[bureau.CREDIT_ACTIVE_Active == 1]
    active_agg = active.groupby("SK_ID_CURR").agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')

    closed = bureau[bureau.CREDIT_ACTIVE_Closed == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del bureau, active, active_agg, closed, closed_agg
    gc.collect()
    return bureau_agg


def previous_applications(num_rows=None, nan_as_category=True):
    prev = pd.read_csv("../input/previous_application.csv", nrows=num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category)

    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    prev['APP_CREDIT_PERC'].replace(float('inf'), 100, inplace=True)
    
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'], 
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'], 
        'CNT_PAYMENT': ['mean', 'sum'], 
    
        'DAYS_LAST_DUE_1ST_VERSION': ['min', 'max', 'mean'], 
        'DAYS_LAST_DUE': ['min', 'max', 'mean'],
    }

    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    del prev
    gc.collect()
    return prev_agg


def pos_cash(num_rows=None, nan_as_category=True):
    pos = pd.read_csv("../input/POS_CASH_balance.csv", nrows=num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category)

    aggregations = {
    'MONTHS_BALANCE': ['max', 'min', 'size'],
    'SK_DPD': ['max', 'mean'],
    'SK_DPD_DEF': ['max', 'mean']
    }

    for cat in cat_cols:
        aggregations[cat] = ['mean']
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg


def installments_payments(num_rows=None, nan_as_category=True):
    ins = pd.read_csv("../input/installments_payments.csv", nrows=num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category)

    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_PERC'].replace(float('inf'), 30000, inplace=True)
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x : x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x : x if x > 0 else 0)

    aggregations = {
    'NUM_INSTALMENT_VERSION': ['nunique'],
    'DPD': ['max', 'mean', 'sum'],
    'DBD': ['max', 'mean', 'sum'],
    'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],  
    'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],  
    'AMT_INSTALMENT': ['max', 'mean', 'sum'],
    'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
    'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }

    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(["INSTAL_" + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg


def credit_card_balance(num_rows=None, nan_as_category=True):
    cc = pd.read_csv("../input/credit_card_balance.csv", nrows=num_rows)
    cc.loc[cc.AMT_DRAWINGS_ATM_CURRENT < 0, 'AMT_DRAWINGS_ATM_CURRENT'] = np.nan
    cc.loc[cc.AMT_DRAWINGS_CURRENT < 0, 'AMT_DRAWINGS_CURRENT'] = np.nan
    cc, cat_cols = one_hot_encoder(cc, nan_as_category)
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min','max','mean','sum','var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg


def join_tables():
    num_rows = 1000
    df = applications_train_test(num_rows)
    with timer("join all tables"):
        bureau = bureau_and_balance(num_rows)
        prev = previous_applications(num_rows)
        pos = pos_cash(num_rows)
        ins = installments_payments(num_rows)
        cc = credit_card_balance(num_rows)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        df = df.join(prev, how='left', on='SK_ID_CURR')
        df = df.join(pos, how='left', on='SK_ID_CURR')
        df = df.join(ins, how='left', on='SK_ID_CURR')
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del bureau, prev, pos, ins, cc
        gc.collect()
    print('data shape: ', df.shape)
    features = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    labels = df.TARGET
    df = lgb.Dataset(df.loc[:, features], label=labels)
    return df


def objective_grid_search(dataset, hyperparameters, iteration):
    if 'n_estimators' in hyperparameters.keys():
        del hyperparameters['n_estimators']
    
    cv_results = lgb.cv(hyperparameters, dataset, num_boost_round=10000, nfold=5, 
                        early_stopping_rounds=100, metrics='auc', seed=42)
    
    score = cv_results['auc-mean'][-1]
    estimators = len(cv_results['auc-mean'])
    hyperparameters['n_estimators'] = estimators
    return [score, hyperparameters, iteration]


def grid_search(dataset, param_grid, max_evals=5):
    results = pd.DataFrame(columns = ['score', 'params', 'iteration'], index = list(range(max_evals)))
    keys, values = zip(*param_grid.items())
    i = 0
    for v in itertools.product(*values):
        hyperparameters = dict(zip(keys, v))
        eval_results = objective_grid_search(dataset, hyperparameters, i)
        results.loc[i, :] = eval_results
        i += 1
        if i > max_evals:
            break
    results.sort_values('score', ascending=False, inplace=True)
    results.reset_index(inplace=True)
    print("best score: ", results.loc[0, 'score'])
    return results


def objective_hyperopt(hyperparameters):
    if 'n_estimators' in hyperparameters:
        del hyperparameters['n_estimators']
    
    for parameter_name in ['num_leaves', 'min_data_in_leaf']:  # these two parameters should be int
        hyperparameters[parameter_name] = int(hyperparameters[parameter_name])
    
    cv_results = lgb.cv(hyperparameters, df, num_boost_round=10000, nfold=5, stratified=False,
                        early_stopping_rounds=200, metrics='auc', seed=42)

    best_score = cv_results['auc-mean'][-1]
    loss = 1 - best_score
    n_estimators = len(cv_results['auc-mean'])
    hyperparameters['n_estimators'] = n_estimators
    return {'loss': loss, 'hyperparameters': hyperparameters, 'status': STATUS_OK}


def hyperopt_search(space, max_evals=5):
    trials = Trials()
    best = fmin(fn=objective_hyperopt, space=space, algo=tpe.suggest, trials=trials, max_evals=max_evals)
    trials_dict = sorted(trials.results, key=lambda x : x['loss'])

    whole_results = pd.Series()
    for model in trials_dict:
        single_result = pd.Series(model['hyperparameters'])
        single_result['auc_score'] = 1 - model['loss']
        whole_results = pd.concat([whole_results, single_result], axis=1, ignore_index=True)
    return whole_results.T.drop(0, axis=0)


def objective_bayesian(subsample, num_leaves, learning_rate, reg_alpha, reg_lambda, colsample_bytree, 
                 min_split_gain, min_child_weight, min_data_in_leaf):
    
    params = dict(boosting='gbdt', max_depth=8)
    params['subsample'] = max(min(subsample, 1), 0)
    params['num_leaves'] = int(round(num_leaves))
    params['learning_rate'] = max(min(subsample, 0.5), 0)
    params['reg_alpha'] = max(reg_alpha, 0)
    params['reg_lambda'] = max(reg_lambda, 0)
    params['colsample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['min_split_gain'] = max(min(min_split_gain, 0.2), 0)
    params['min_child_weight'] = max(min(min_child_weight, 50), 0)
    params['min_data_in_leaf'] = int(max(min(min_data_in_leaf, 500), 5))

    cv_result = lgb.cv(params, df, num_boost_round=10000, nfold=5, stratified=False,
                       early_stopping_rounds=200, metrics='auc', seed=42, verbose_eval=200)
    
    print(len(cv_result['auc-mean']))
    params['n_estimators'] = len(cv_result['auc-mean'])
    return max(cv_result['auc-mean'])


def bayesian_search(space, iteration=10):
    bayesian_opt = BayesianOptimization(space, {'num_leaves': (20, 150),
                                             'subsample': (0.5, 1),
                                             'learning_rate': (0.01, 0.1),
                                             'reg_alpha': (0.0, 1.0),
                                             'reg_lambda': (0.0, 1.0),
                                             'colsample_bytree': (0.6, 1.0),
                                             'min_split_gain': (0.0, 0.2),
                                             'min_child_weight': (0.0, 50.0),  
                                             'min_data_in_leaf': (5, 500)}, random_state=42)
    bayesian_opt.maximize(init_points=2, n_iter=iteration)
    result = pd.concat([pd.DataFrame(bayesian_opt.res['all']['params']), 
                        pd.DataFrame(bayesian_opt.res['all']['values'])], axis=1)
    result.rename({0: 'roc_score'}, axis=1, inplace=True)
    return result.sort_values('roc_score', ascending=False)


param_grid = {
    'num_leaves': list(range(20, 150)),
    'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base=10, num=1000)),
    'subsample': list(np.linspace(0.5, 1, 100)),
    'reg_alpha': list(np.linspace(0, 1)),
    'reg_lambda': list(np.linspace(0, 1)),
    'colsample_bytree': list(np.linspace(0.5, 1, 30)),
    'min_split_gain': list(np.linspace(0.0, 0.2, 20)),
    'min_child_weight': list(np.linspace(0.0, 50.0, 20)),
    'min_data_in_leaf': list(range(5, 500, 5)),
    }

space = {
    'boosting_type': hp.choice('boosting_type', ['gbdt']),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'num_leaves': hp.quniform('num_leaves', 20, 150, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.1)),
 #   'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
 #   'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
    'min_split_gain': hp.uniform('min_split_gain', 0.0, 0.2),
    'min_child_weight': hp.uniform('min_child_weight', 0.0, 50.0), 
    'min_data_in_leaf': hp.quniform('min_data_in_leaf', 5, 500, 1),
#    'min_data_in_leaf': hp.quniform('min_data_in_leaf', 5, 80, 1),
    }

    

if __name__ == "__main__":
    with timer("Data Preparing: "):
        df = join_tables()

#    with timer("Grid Search: "):
#        result = grid_search(df, param_grid, max_evals=5)

#    with timer("Hyperopt: "):
#        result = hyperopt_search(space, max_evals=5)

#    with timer("Bayesian Optimization: "):
#        result = bayesian_search(objective_bayesian, iteration=2)