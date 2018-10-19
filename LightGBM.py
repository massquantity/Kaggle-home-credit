import numpy as np
import pandas as pd
import gc
import time
import sys
from contextlib import contextmanager
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
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
    df_train = pd.read_csv("../input/application_train.csv", nrows=num_rows)
    df_test = pd.read_csv("../input/application_test.csv", nrows=num_rows)
    df_test.loc[df_test.REGION_RATING_CLIENT_W_CITY == -1, "REGION_RATING_CLIENT_W_CITY"] = 2
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

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
    del df_train, df_test
    gc.collect()
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


def model_lightgbm(df, convert=False, cv=False, stratified=True):
    if convert:
        print("data size before convert: ", np.round(sys.getsizeof(df) / 1024 / 1024, 2))
        for col in df.columns:
            c_min = df[col].min()
            c_max = df[col].max()

            if df[col].dtype == 'int64':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                else:
                    df[col] = df[col].astype(np.int32)
            elif df[col].dtype == 'float64':
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                else:
                    df[col] = df[col].astype(np.float32)
        print("data size after convert: ", np.round(sys.getsizeof(df) / 1024 / 1024, 2))

    features = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    train_df = df.loc[df.TARGET.notnull(), features]
    test_df = df.loc[df.TARGET.isnull(), features]
    y = df.loc[df.TARGET.notnull(), 'TARGET']
    result = df.loc[df.TARGET.isnull(), "SK_ID_CURR"]
    del df
    gc.collect()

    if cv:
        if stratified:
            folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        else:
            folds = KFold(n_splits=5, shuffle=True, random_state=42)

        oof_preds = np.zeros(train_df.shape[0])
        feature_importance_df = pd.DataFrame()
        for n_fold , (train_idx, valid_idx) in enumerate(folds.split(train_df, y)):
            train_x, train_y = train_df.iloc[train_idx], y.iloc[train_idx]
            valid_x, valid_y = train_df.iloc[valid_idx], y.iloc[valid_idx]

            clf = LGBMClassifier(
            #   boosting='goss',
                nthread=4,
                n_estimators=10000,
                learning_rate=0.02,
                num_leaves=34,
                colsample_bytree=0.9497036,
                subsample=0.8715623,
                max_depth=8,
                reg_alpha=0.041545473,
                reg_lambda=0.0735294,
                min_split_gain=0.0222415,
                min_child_weight=39.3259775,   # min_data_in_leaf
                silent=-1,
                verbose=-1, )

            with timer("model fit"):
                clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
                eval_metric='auc', verbose=500, early_stopping_rounds=200)
            
            oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = features
            fold_importance_df["importance"] = clf.feature_importances_
            fold_importance_df["fold"] = n_fold + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
            print("Fold %2d AUC: %.6f" % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))

        print("Full AUC score %.6f" % roc_auc_score(y, oof_preds))
        display_importances(feature_importance_df)
        return feature_importance_df

    else:
        sub_preds = np.zeros(test_df.shape[0])
        with timer('lgb_ensemble'):
            for seed in [1, 42, 100, 1024, 2018]:
                clf = LGBMClassifier(
                    nthread=4,
                    n_estimators=4600,
                    learning_rate=0.01,
                    num_leaves=21,
                    colsample_bytree=0.609317,
                    subsample=0.973193,
                    max_depth=8,
                    reg_alpha=0.283623,
                    reg_lambda=0.165442,
                    min_split_gain=0.134190,
                    min_child_weight=4.326922,
                    min_data_in_leaf=375,
                    silent=-1,
                    verbose=-1,
                    random_state=seed, )

                clf.fit(train_df, y, eval_metric='auc', verbose=500)
                sub_preds += clf.predict_proba(test_df, num_iteration=clf.n_estimators)[:, 1] / 5
    
        result['TARGET'] = sub_preds
        result.to_csv("submission_lgbm.csv", index=False)


def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(15, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title("LightGBM Features (avg over folds)")
    plt.tight_layout()


def main():
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

    with timer("model fitting"):
        model_lightgbm(df, convert=False, cv=False)


if __name__ == "__main__":
    with timer("Full model run"):
        main()
