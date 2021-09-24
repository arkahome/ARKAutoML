import pandas as pd
import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold
from loguru import logger

def create_folds(df,n,target_col,fold_method):
    '''
    Sample call:
    create_folds(df,5,'target','KFold')

    fold_method can be from any of these values ['KFold','StratifiedKFold']
    '''
    df = df.copy()
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    
    if fold_method=='KFold':
        kf = KFold(n_splits=n)
        for fold, (train_idx,val_idx) in enumerate(kf.split(X=df)):
            logger.info(f'Fold {fold+1} : {len(train_idx)}, {len(val_idx)}')
            df.loc[val_idx,'kfold'] = fold
    
    elif fold_method=='StratifiedKFold':
        skf = StratifiedKFold(n_splits=n)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X=df, y=df[target_col].values)):
            logger.info(f'Fold {fold+1} : {len(train_idx)}, {len(val_idx)}')
            df.loc[val_idx, 'kfold'] = fold
    
    elif fold_method=='RegressionStratifiedKFold':
        num_bins = np.floor(1 + np.log2(len(df)))
        # bin targets
        df.loc[:, "bins"] = pd.cut(
        df[target_col], bins=num_bins, labels=False
        )
        # initiate the kfold class from model_selection module
        kf = StratifiedKFold(n_splits=5)
        # fill the new kfold column
        # note that, instead of targets, we use bins!
        for f, (t_, v_) in enumerate(kf.split(X=df, y=df.bins.values)):
            df.loc[v_, 'kfold'] = f
        # drop the bins column
        df = df.drop("bins", axis=1)

    # df.to_csv('../input/df_folds.csv',index=False)
    return df

