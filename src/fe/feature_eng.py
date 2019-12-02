import pandas as pd
import numpy as np
import random
import feather
import sys

# Load the utility script
sys.path.insert(0, "/home/jupyter/kaggle/energy/src")
import utility


WRITE_FEATHER = True

train_df, test_df, _, _, _ = utility.read_data(utility.CREATED_DATA_DIR, 
                                         train=True, 
                                         test=True, 
                                         weather_train=False, 
                                         weather_test=False, 
                                         building=False)

# Clean the training data for site_0 first
train_df = utility.clean_data_for_site_0(train_df)

# Write the cleaned data as a feather file
if WRITE_FEATHER:
    train_df.reset_index(drop=True).to_feather(f'{utility.CREATED_FEATURE_DIR}/train_data_cleaned_site_0.feather')

# Place Holder DataFrame for newly created train and test features
train_features_df = pd.DataFrame()
test_features_df = pd.DataFrame()


# First create date features
train_features_df = utility.create_date_features(train_df, train_features_df, 'timestamp')
test_features_df = utility.create_date_features(test_df, test_features_df, 'timestamp')

# As of now creating the following intercation features
# site_id + building_id
# site_id + building_id + meter
# site_id + building_id + meter + usage
# site_id + meter
# building_id + meter
# site_id + usage
# building_id + usage
# meter + usage
train_features_df = utility.create_interaction_features(train_df, train_features_df)
test_features_df = utility.create_interaction_features(test_df, test_features_df)


# Frequency Encoding
# Not sure how to do it at this momment?

# Create Age of the buildings
train_features_df = utility.create_age(train_df, train_features_df, 'year_built')
test_features_df = utility.create_age(test_df, test_features_df, 'year_built')

if WRITE_FEATHER:
    train_features_df.reset_index(drop=True).to_feather(f'{utility.CREATED_FEATURE_DIR}/train_features.feather')
    test_features_df.reset_index(drop=True).to_feather(f'{utility.CREATED_FEATURE_DIR}/test_features.feather')
    
