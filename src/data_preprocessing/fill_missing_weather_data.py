import feather
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, "/home/jupyter/kaggle/energy/src")
import utility

utility.set_seed(42)

# Read weather_train and weather_test data
_, _, weather_train_df, weather_test_df, building_df = utility.read_data(utility.CREATED_DATA_DIR, 
                                                                         train=False, test=False, 
                                                                         weather_train=True, weather_test=True, 
                                                                         building=True)

print(f'Shape of weather_train_df : {weather_train_df.shape}')
print(f'Shape of weather_test_df : {weather_test_df.shape}')

# columns_name = ['air_temperature', 'cloud_coverage', 'dew_temperature', 
#                 'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 
#                 'wind_speed']

# cloud_coverage, precip_depth_1_hr, sea_level_pressure, wind_direction
# are failing now for some reason. Hence, filling only for three weather
# attributes
columns_name = ['air_temperature', 'dew_temperature', 'wind_speed']

print('Null distribution before filling')
print(weather_train_df[columns_name].isna().sum())

weather_train_filled = pd.DataFrame(columns=['site_id', 'timestamp'])
for name in columns_name:
    print(f'Filling {name} values...')
    df_filled_mix = utility.fill_null_values_with_mix(weather_train_df, index_name='timestamp', columns_name='site_id', values_name=name)
    weather_train_filled = pd.merge(weather_train_filled, df_filled_mix, how='outer', on=['site_id', 'timestamp'])
    
print(weather_train_filled.shape)
print('Null distribution after filling')
print(weather_train_filled[columns_name].isna().sum())
