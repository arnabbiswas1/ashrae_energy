import pandas as pd
import numpy as np
import random
import feather


def read_data(data_dir):
    print('Reading Data...')
    train_df = feather.read_dataframe(f'{data_dir}/train_merged.feather')
    test_df = feather.read_dataframe(f'{data_dir}/test_merged.feather')
    print(f'Shape of train_df : {train_df.shape}')
    print(f'Shape of test_df : {test_df.shape}')
    return train_df, test_df


def create_date_features(df, feature_name):
    '''
    Create new features related to dates
    
    df : The complete dataframe
    feature_name : Name of the feature of date type which needs to be decomposed.
    '''
    print('Creating date related fetaures...')
    df.loc[:, 'year'] = df.loc[:, feature_name].dt.year.astype('uint16')
    df.loc[:, 'month'] = df.loc[:, feature_name].dt.month.astype('uint8')
    df.loc[:, 'quarter'] = df.loc[:, feature_name].dt.quarter.astype('uint8')
    df.loc[:, 'weekofyear'] = df.loc[:, feature_name].dt.weekofyear.astype('uint8')
    
    df.loc[:, 'day'] = df.loc[:, feature_name].dt.day.astype('uint16')
    df.loc[:, 'dayofweek'] = df.loc[:, feature_name].dt.dayofweek.astype('uint8')
    df.loc[:, 'dayofyear'] = df.loc[:, feature_name].dt.dayofyear.astype('uint16')
    df.loc[:, 'is_month_start'] = df.loc[:, feature_name].dt.is_month_start
    df.loc[:, 'is_month_end'] = df.loc[:, feature_name].dt.is_month_end
    df.loc[:, 'is_quarter_start']= df.loc[:, feature_name].dt.is_quarter_start
    df.loc[:, 'is_quarter_end'] = df.loc[:, feature_name].dt.is_quarter_end
    df.loc[:, 'is_year_start'] = df.loc[:, feature_name].dt.is_year_start
    df.loc[:, 'is_year_end'] = df.loc[:, feature_name].dt.is_year_end
    
    df.loc[:, 'hour'] = df.loc[:, feature_name].dt.hour.astype('uint8')    
    return df


def concat_features(source_df, target_df, f1, f2):
    target_df[f'{f1}_{f2}'] =  source_df[f1].astype(str) + '_' + source_df[f2].astype(str)
    return target_df


def create_interaction_features(source_df, target_df):
    target_df = concat_features(source_df, target_df, 'site_id', 'building_id')
    target_df['site_building_meter_id'] = source_df.site_id.astype(str) + '_' + source_df.building_id.astype(str) + '_' + source_df.meter.astype(str)
    target_df['site_building_meter_id_usage'] = source_df.site_id.astype(str) + '_' + source_df.building_id.astype(str) + '_' + source_df.meter.astype(str) + '_' + source_df.primary_use

    target_df = concat_features(source_df, target_df, 'site_id', 'meter')
    target_df = concat_features(source_df, target_df, 'building_id', 'meter')
    
    target_df = concat_features(source_df, target_df, 'site_id', 'primary_use')
    target_df = concat_features(source_df, target_df, 'building_id', 'primary_use')
    target_df = concat_features(source_df, target_df, 'meter', 'primary_use')
    
    return target_df


def create_age(source_df, target_df, f):
    target_df['building_age'] = 2019 - source_df[f]
    return target_df


DATA_DIR = '/home/jupyter/kaggle/energy/data/read_only_feather/v2'
CREATED_FEATURE_DIR = '/home/jupyter/kaggle/energy/data/created_data'

train_df, test_df = read_data(DATA_DIR)

# Place Holder DataFrame for newly created train and test features
train_features_df = pd.DataFrame()
test_features_df = pd.DataFrame()


# As of now creating the following intercation features
# site_id + building_id
# site_id + building_id + meter
# site_id + building_id + meter + usage
# site_id + meter
# building_id + meter
# site_id + usage
# building_id + usage
# meter + usage
train_features_df = create_interaction_features(train_df, train_features_df)
test_features_df = create_interaction_features(test_df, test_features_df)


# Frequency Encoding
# Not sure how to do it at this momment?

# Create Age of the buildings
train_features_df = create_age(train_df, train_features_df, 'year_built')
test_features_df = create_age(test_df, test_features_df, 'year_built')

train_features_df.reset_index(drop=True).to_feather(f'{CREATED_FEATURE_DIR}/train_features.feather')
test_features_df.reset_index(drop=True).to_feather(f'{CREATED_FEATURE_DIR}/train_features.feather')