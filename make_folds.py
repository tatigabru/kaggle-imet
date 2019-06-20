import configs
import argparse
from collections import defaultdict, Counter
import random
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import StratifiedKFold, KFold


def make_folds(df, n_folds: int) -> pd.DataFrame:
    """
    makes iterative stratification of multi label data
    
    Author:  Kostantin Lopuhin
    Source:  https://github.com/lopuhin/kaggle-imet-2019/tree/master/imet
        
    Input:  df dataframe with images and labels
            n_folds: number of folds           
    Output: df with folds    
    """
    cls_counts = Counter(cls for classes in df['attribute_ids'].str.split()
                         for cls in classes)
    print(cls_counts)
    fold_cls_counts = defaultdict(int)
    folds = [-1] * len(df)
    for item in tqdm.tqdm(df.sample(frac=1, random_state=42).itertuples(),
                          total=len(df)):
        cls = min(item.attribute_ids.split(), key=lambda cls: cls_counts[cls])
        print(cls)
        if cls < 5:
            continue
        fold_counts = [(f, fold_cls_counts[f, cls]) for f in range(n_folds)]
        min_count = min([count for _, count in fold_counts])
        random.seed(item.Index)
        fold = random.choice([f for f, count in fold_counts
                              if count == min_count])
        folds[item.Index] = fold
        for cls in item.attribute_ids.split():
            fold_cls_counts[fold, cls] += 1
    df['fold'] = folds
    return df


def make_strat_folds(df, n_folds: int) -> pd.DataFrame:
    """
    makes iterative stratification of multi label data
    Source: https://github.com/trent-b/iterative-stratification
    """
    msss = MultilabelStratifiedShuffleSplit(n_splits=n_folds, test_size=0.2, random_state=42)
    train_df_orig = df.copy()
    X = train_df_orig['ImageId'].tolist()
    cls_counts = Counter(cls for classes in df['attribute_ids'].str.split()
                         for cls in classes)
    y = train_df_orig['attribute_ids'].str.split().tolist()
    #print(X, y)
    for train_index, test_index in msss.split(X, y): 
        print("TRAIN:", train_index, "TEST:", test_index)
        train_df = train_df_orig.loc[train_df_orig.index.intersection(train_index)].copy()
        valid_df = train_df_orig.loc[train_df_orig.index.intersection(test_index)].copy()
    return train_df, valid_df


def split_images(df, X):
    """Splits by unique image names
       different images are in train and validation sets,
       no stratification
       Output: a dataframe with added folds values
    """
    df['folds'] = 0    
    folds = KFold(n_splits=5, shuffle=True, random_state=1111)    
    # make split
    for num, (train_index, test_index) in enumerate(folds.split(X)):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index] 
        print('X_train, X_test: ', X_train, X_test)         
        df['folds'].loc[df['ImageId'].isin(X_test)] = num        
    print(df.head(40))     
    return df


def check_folds(file_name):    
    # sanity checks
    folds = pd.read_csv(file_name)    
    for fold in range(1):  
        train_fold = folds[folds['Folds'] != fold]
        valid_fold = folds[folds['Folds'] == fold] 
        
        cls_counts = Counter(cls for classes in train_fold['attribute_ids'].str.split()
                                 for cls in classes)   
        print('train_fold counts :', cls_counts)
        cls_counts = Counter(cls for classes in valid_fold['attribute_ids'].str.split()
                                 for cls in classes) 
        print('valid_fold counts :', cls_counts)      
    
    
def main():
    DATA_ROOT = '../../../input/'
    train = pd.read_csv(DATA_ROOT+'train_with_attr.csv')    
    # select data with dress only  
    #train = train[train['Categories']==10] 
    #imgs = train.ImageId.unique() 
    #print(len(imgs)) 
    train_df = select_classes_with_attributes(train)
    # select data with attributes only  
    train_df = train[train["Class_att"] == 1]
    print(train_df.Categories.unique())   
    print(len(train_df))
    images = train_df.ImageId.unique()   
    print(len(images))        
    X = train.ImageId.unique()   
    print(len(X))
    cls_counts = Counter(cls for classes in train_df['Attributes'].str.split()
                         for cls in classes)
    print(cls_counts)
    #count unique attributes
    print('all attributes counts :', cls_counts)
    print(list(cls_counts), len(list(cls_counts)))
    
    df = stratify_by_categories(train_df)
    
    #df = split_unique_images(train_df, X)
    #df = split_unique_images(train_df, X)
    #df = make_folds(train_df, n_folds=5)
    #train_df, valid_df = make_strat_folds(train_df) 
  
    # save folds to scv file
    file_name = 'attributes_folds.csv'
    df.to_csv(file_name, index=None)  
    check_folds(file_name)
   

    
if __name__ == '__main__':
    main()
