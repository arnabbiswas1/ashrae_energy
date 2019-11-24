"""
This script loads the raw csv files, converts few columns into int32.
And writes back as feather format.

TODO 
- Need to decide if float64 should be converted into float32.
- For weather_data there are variables which are categorical in nature.
But because of the presence of Null, I am not able to convert. Post Null 
Handling it should be converted.

"""

import pandas as pd
import numpy as np
import feather


DATA_DIR = '/home/jupyter/kaggle/energy/data/read_only'
CREATED_DATA_DIR = '/home/jupyter/kaggle/energy/data/read_only_feather/v2'

print(f'Loading Data from {DATA_DIR}..')

train_dtypes = {'building_id' : np.uint16, 
                'meter' : np.uint8, 
                'meter_reading' : np.float32}

test_dtypes = {'row_id' : np.uint32,
               'building_id' : np.uint16, 
               'meter' : np.uint8}

print('Loading Data from train.csv ..')
train = pd.read_csv(f'{DATA_DIR}/train.csv', parse_dates=['timestamp'], 
                    low_memory=False, dtype=train_dtypes)
print(f'Shape of train : {train.shape}')

print('Loading Data from test.csv..')
# row_id is the index?
test = pd.read_csv(f'{DATA_DIR}/test.csv', parse_dates=['timestamp'], 
                   low_memory=False, dtype=test_dtypes)
print(f'Shape of test : {test.shape}')

# I am converting the data types with NaNs into Int32Dtype.
# https://pandas.pydata.org/pandas-docs/stable/user_guide/integer_na.html
weather_train_dtypes = {'site_id' : np.uint8, 
                        'air_temperature' : np.float32,
                        'cloud_coverage' : np.float32, 
                        'dew_temperature' : np.float32,
                        'precip_depth_1_hr' : np.float32, 
                        'sea_level_pressure' : np.float32,
                        'wind_direction' : np.float32,
                        'wind_speed' : np.float32}

weather_test_dtypes = {'site_id' : np.uint8, 
                        'air_temperature' : np.float32,
                        'cloud_coverage' : np.float32, 
                        'dew_temperature' : np.float32,
                        'precip_depth_1_hr' : np.float32, 
                        'sea_level_pressure' : np.float32,
                        'wind_direction' : np.float32,
                        'wind_speed' : np.float32}

print('Loading Data from weather_train.csv..')
weather_train = pd.read_csv(f'{DATA_DIR}/weather_train.csv', 
                            parse_dates=['timestamp'], 
                            low_memory=False, 
                            dtype=weather_train_dtypes)
print(f'Shape of weather_train : {weather_train.shape}')

print('Loading Data from weather_test.csv..')
weather_test = pd.read_csv(f'{DATA_DIR}/weather_test.csv', 
                           parse_dates=['timestamp'], 
                           low_memory=False, 
                           dtype=weather_test_dtypes)

print(f'Shape of weather_test : {weather_test.shape}')

print('Loading Data from building_metadata.csv..')

# dtypes defined for buildind data
building_dtypes = {'site_id' : np.uint8, 
                  'building_id' : np.uint16, 
                  'square_feet' : np.uint32, 
                  'year_built' : np.float32,
                  'floor_count' : np.float32
                  }

building = pd.read_csv(f'{DATA_DIR}/building_metadata.csv', low_memory=False, dtype=building_dtypes)
print(f'Shape of building : {building.shape}')

# Load the submission data
submission_dtype = {'row_id' : np.uint32, 'meter_reading' : np.float32}
submission = pd.read_csv(f'{DATA_DIR}/sample_submission.csv')
print(f'Shape of submission data : {submission.shape}')

# Writing data in feather format
print(f'Writing in feather format to {CREATED_DATA_DIR}')
train.reset_index(drop=True).to_feather(f'{CREATED_DATA_DIR}/train.feather')
test.reset_index(drop=True).to_feather(f'{CREATED_DATA_DIR}/test.feather')
weather_train.reset_index(drop=True).to_feather(f'{CREATED_DATA_DIR}/weather_train.feather')
weather_test.reset_index(drop=True).to_feather(f'{CREATED_DATA_DIR}/weather_test.feather')
building.reset_index(drop=True).to_feather(f'{CREATED_DATA_DIR}/building.feather')
submission.reset_index(drop=True).to_feather(f'{CREATED_DATA_DIR}/submission.feather')
print('Saving original data in feather format has been completed...')


# Uptil this point filling with Null has not have happened.
# Let's do that now

# Filling all the missing values with year from the future
building.year_built.fillna(1000, inplace=True)
# Let's mark missing value of floor_count with 99.
building.floor_count.fillna(99, inplace=True)

# Converting data types (Only possible after filling values)
building.floor_count = building.floor_count.astype('uint8')
building.year_built = building.year_built.astype('uint16')

# In weather data, all the columns has Null Value apart 
# from site_id and timestamp. They can't be filled up now.

# site_id                   0
# timestamp                 0
# air_temperature          55
# cloud_coverage        69173
# dew_temperature         113
# precip_depth_1_hr     50289
# sea_level_pressure    10618
# wind_direction         6268
# wind_speed              304

# Create the merged train data set
print('Merging train with building')
print(f'Shape of train before merge {train.shape}')
train_building = pd.merge(train, building, how='left', on='building_id')
print(f'Shape of train after merge {train_building.shape}')
print('Merging train with weather')
print(f'Shape of weather_train before merge {weather_train.shape}')
train_df = pd.merge(train_building, weather_train, how='left', on=['site_id', 'timestamp'])
print(f'Shape of train after merge {train_df.shape}')

# Create the merged test data set
print('Merging test with building')
print(f'Shape of test before merge {test.shape}')
test_building = pd.merge(test, building, how='left', on='building_id')
print(f'Shape of test after merge {test_building.shape}')
print('Merging test with weather')
print(f'Shape of weather_test before merge {weather_test.shape}')
test_df = pd.merge(test_building, weather_test, how='left', on=['site_id', 'timestamp'])
print(f'Shape of test after merge {test_df.shape}')

# Order the column names
train_ordered_column_names = ['site_id', 'building_id', 'timestamp', 'meter',
       'primary_use', 'square_feet', 'year_built', 'floor_count',
       'air_temperature', 'cloud_coverage', 'dew_temperature',
       'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction',
       'wind_speed', 'meter_reading']

#Include row_id. All columns except meter_reading
test_ordered_column_names = ['row_id', 'site_id', 'building_id', 'timestamp', 'meter',
       'primary_use', 'square_feet', 'year_built', 'floor_count',
       'air_temperature', 'cloud_coverage', 'dew_temperature',
       'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction',
       'wind_speed']
# Order the column names in convenient order
train_df = train_df[train_ordered_column_names]
test_df = test_df[test_ordered_column_names]
print(f'Shape of train after ordering : {train_df.shape}')
print(f'Shape of test after ordering : {test_df.shape}')

# Sort train and test based on time
print('Sorting values based on timestamp, site_id, building_id & meter...')
train_df.sort_values(['timestamp', 'site_id', 'building_id', 'meter'], inplace=True)
test_df.sort_values(['timestamp', 'site_id', 'building_id', 'meter'], inplace=True)
print(f'Shape of train after sorting : {train_df.shape}')
print(f'Shape of test after sorting : {test_df.shape}')

print('Writing the merged data in feather format...')
train_df.reset_index(drop=True).to_feather(f'{CREATED_DATA_DIR}/train_merged.feather')
test_df.reset_index(drop=True).to_feather(f'{CREATED_DATA_DIR}/test_merged.feather')
building.reset_index(drop=True).to_feather(f'{CREATED_DATA_DIR}/building_filled.feather')
print('Writing the merged data in feather format : Completed!')