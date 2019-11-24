import numpy as np 
import pandas as pd
import feather
import plotly
import plotly.graph_objects as go
import gc

pd.options.display.max_rows = 200

from IPython.display import display


meter_dict = {0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'}
CREATED_DATA_DIR = '/home/jupyter/kaggle/energy/data/read_only_feather/v2'


def read_files(dir_path, train_file_name='train.csv', 
               test_file_name='test.csv', 
               submission_file_name='sample_submission.csv'):

    print('Loading Data...')
    train = pd.read_csv(f'{dir_path}/{train_file_name}')
    test = pd.read_csv(f'{dir_path}/{test_file_name}')
    submission = pd.read_csv(f'{dir_path}/{submission_file_name}')
    
    print(f'Shape of {train_file_name} : {train.shape}')
    print(f'Shape of {test_file_name} : {test.shape}')
    print(f'Shape of {submission_file_name} : {submission.shape}')
    
    print('Data Loaded...')
    
    return train, test, submission


def read_data(data_dir, train=True, test=True, weather_train=False, weather_test=False, building=False):
    print('Reading Data...')
    train_df = None
    test_df = None
    weather_train_df = None
    weather_test_df = None
    building_df = None
    if train:
        train_df = feather.read_dataframe(f'{data_dir}/train_merged.feather')
        print(f'Shape of train_df : {train_df.shape}')
    if test:
        test_df = feather.read_dataframe(f'{data_dir}/test_merged.feather')
        print(f'Shape of test_df : {test_df.shape}')
    if weather_train:
        weather_train_df = feather.read_dataframe(f'{data_dir}/weather_train.feather')
        print(f'Shape of weather_train_df : {weather_train_df.shape}')
    if weather_test:
        weather_test_df = feather.read_dataframe(f'{data_dir}/weather_test.feather')
        print(f'Shape of weather_test_df : {weather_test_df.shape}')
    if building:
        building_df = feather.read_dataframe(f'{data_dir}/building.feather')
        print(f'Shape of building_df : {building_df.shape}')
    return train_df, test_df, weather_train_df, weather_test_df, building_df


def read_data(data_dir, train=True, test=True, weather_train=False, weather_test=False, building=False):
    print('Reading Data...')
    train_df = None
    test_df = None
    weather_train_df = None
    weather_test_df = None
    building_df = None
    if train:
        train_df = feather.read_dataframe(f'{data_dir}/train_merged.feather')
        print(f'Shape of train_df : {train_df.shape}')
    if test:
        test_df = feather.read_dataframe(f'{data_dir}/test_merged.feather')
        print(f'Shape of test_df : {test_df.shape}')
    if weather_train:
        weather_train_df = feather.read_dataframe(f'{data_dir}/weather_train.feather')
        print(f'Shape of weather_train_df : {weather_train_df.shape}')
    if weather_test:
        weather_test_df = feather.read_dataframe(f'{data_dir}/weather_test.feather')
        print(f'Shape of weather_test_df : {weather_test_df.shape}')
    if building:
        building_df = feather.read_dataframe(f'{data_dir}/building.feather')
        print(f'Shape of building_df : {building_df.shape}')
    return train_df, test_df, weather_train_df, weather_test_df, building_df


############################################## Utility ##############################################

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)

############################################## Visualization ##############################################


def plot_meter_reading_for_site(df, site_id, meter_name):
    """
    Plot meter_reading for an entire site for all buildings
    """
    df = df.set_index('timestamp')
    building_id_list = df.columns
    for building_id in building_id_list:
        fig = go.Figure()
        df_subset = df.loc[:, building_id]
        fig.add_trace(go.Scatter(
             x=df_subset.index,
             y=df_subset.values,
             name=f"{meter_name}",
             hoverinfo=f'x+y+name',
             opacity=0.7))

        fig.update_layout(width=1000,
                        height=500,
                        title_text=f"Meter Reading for Site [{site_id}] Building [{building_id}]",
                        xaxis_title="timestamp",
                        yaxis_title="meter_reading",)
        fig.show()
        

def plot_meter_reading_for_building(df, site_id, building_id, meter_name):
    """
    Plot meter_reading for an entire site for all buildings
    """
    df = df.set_index('timestamp')
    building_id_list = df.columns
    for building_id in building_id_list:
        fig = go.Figure()
        df_subset = df.loc[:, building_id]
        fig.add_trace(go.Scatter(
             x=df_subset.index,
             y=df_subset.values,
             name=f"{meter_name}",
             hoverinfo=f'x+y+name',
             opacity=0.7))

        fig.update_layout(width=1000,
                        height=500,
                        title_text=f"Meter Reading for Site [{site_id}] Building [{building_id}]",
                        xaxis_title="timestamp",
                        yaxis_title="meter_reading",)
        fig.show()
        

def display_all_site_meter_reading(df, site_id, meter):
    train_df_meter_subset = df[(df.site_id == site_id) & (train_df.meter == 0)]
    train_df_meter_subset = train_df_meter_subset.pivot(index='timestamp', columns='building_id', values='meter_reading')

    column_names = train_df_meter_subset.reset_index().columns.values
    train_df_meter_subset.reset_index(inplace=True)
    train_df_meter_subset.columns = column_names
    
    print(f'Missing Values for {site_id}')
    display(train_df_meter_subset.isna().sum())
    
    plot_meter_reading_for_site(train_df_meter_subset, 0, meter_dict[meter])


def plot_hist_train_test_overlapping(df_train, df_test, feature_name, kind='hist'):
    """
    Plot histogram for a particular feature both for train and test.
    
    kind : Type of the plot
    
    """
    df_train[feature_name].plot(kind=kind, figsize=(15, 5), label='train', 
                         bins=50, alpha=0.4, 
                         title=f'Train vs Test {feature_name} distribution')
    df_test[feature_name].plot(kind='hist',label='test', bins=50, alpha=0.4)
    plt.legend()
    plt.show()
    
    

def plot_barh_train_test_side_by_side(df_train, df_test, feature_name, normalize=True, sort_index=False):
    """
    Plot histogram for a particular feature both for train and test.
    
    kind : Type of the plot
    
    """
    print(f'Number of unique values in train : {count_unique_values(df_train, feature_name)}')
    print(f'Number of unique values in test : {count_unique_values(df_test, feature_name)}')
    
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 8))
    
    if sort_index == True:
            df_train[feature_name].value_counts(
                normalize=normalize, dropna=False).sort_index().plot(
                kind='barh', figsize=(15, 5), 
                ax=ax1,
                grid=True,
                title=f'Bar plot for {feature_name} for train')
    
            df_test[feature_name].value_counts(
                    normalize=normalize, dropna=False).sort_index().plot(
                    kind='barh', figsize=(15, 5), 
                    ax=ax2,
                    grid=True,
                    title=f'Bar plot for {feature_name} for test')
    else:
        df_train[feature_name].value_counts(
                normalize=normalize, dropna=False).sort_values().plot(
                kind='barh', figsize=(15, 5), 
                ax=ax1,
                grid=True,
                title=f'Bar plot for {feature_name} for train')

        df_test[feature_name].value_counts(
                normalize=normalize, dropna=False).sort_values().plot(
                kind='barh', figsize=(15, 5), 
                ax=ax2,
                grid=True,
                title=f'Bar plot for {feature_name} for test')

    
    plt.legend()
    plt.show()
    
    
def plot_line_train_test_overlapping(df_train, df_test, feature_name):
    """
    Plot line for a particular feature both for train and test
    """
    df_train[feature_name].plot(kind='line', figsize=(10, 5), label='train', 
                          alpha=0.4, 
                         title=f'Train vs Test {feature_name} distribution')
    df_test[feature_name].plot(kind='line',label='test', alpha=0.4)
    plt.ylabel(f'Value of {feature_name}')
    plt.legend()
    plt.show()
    
    
def plot_hist(df, feature_name, kind='hist', bins=100, log=True):
    """
    Plot either for train or test
    """
    if log:
        df[feature_name].apply(np.log1p).plot(kind='hist', 
                                              bins=bins, 
                                              figsize=(15, 5), 
                                              title=f'Distribution of log1p[{feature_name}]')
    else:
        df[feature_name].plot(kind='hist', 
                              bins=bins, 
                              figsize=(15, 5), 
                              title=f'Distribution of {feature_name}')
    plt.show()


def plot_barh(df, feature_name, normalize=True, kind='barh', figsize=(15,5), sort_index=False):
    """
    Plot barh for a particular feature both for train and test.
    
    kind : Type of the plot
    
    """
    if sort_index==True:
        df[feature_name].value_counts(
                normalize=normalize, dropna=False).sort_index().plot(
                kind=kind, figsize=figsize, grid=True,
                title=f'Bar plot for {feature_name}')
    else:   
        df[feature_name].value_counts(
                normalize=normalize, dropna=False).sort_values().plot(
                kind=kind, figsize=figsize, grid=True,
                title=f'Bar plot for {feature_name}')
    
    plt.legend()
    plt.show()
    

def plot_boxh(df, feature_name, kind='box', log=True):
    """
    Box plot either for train or test
    """
    if log:
        df[feature_name].apply(np.log1p).plot(kind='box', vert=False, 
                                                  figsize=(10, 6), 
                                                  title=f'Distribution of log1p[{feature_name}]')
    else:
        df[feature_name].plot(kind='box', vert=False, 
                              figsize=(10, 6), 
                              title=f'Distribution of {feature_name}')
    plt.show()
    
    
def plot_boxh_groupby(df, feature_name, by):
    """
    Box plot with groupby feature
    """
    df.boxplot(column=feature_name, by=by, vert=False, 
                              figsize=(10, 6))
    plt.title(f'Distribution of {feature_name} by {by}')
    plt.show()


def plot_meter_reading_for_site(df, site_id, meter_name):
    """
    Plot meter_reading for an entire site for all buildings
    """
    df = df.set_index('timestamp')
    building_id_list = df.columns
    for building_id in building_id_list:
        fig = go.Figure()
        df_subset = df.loc[:, building_id]
        fig.add_trace(go.Scatter(
             x=df_subset.index,
             y=df_subset.values,
             name=f"{meter_name}",
             hoverinfo=f'x+y+name',
             opacity=0.7))

        fig.update_layout(width=1000,
                        height=500,
                        title_text=f"Meter Reading for Site [{site_id}] Building [{building_id}]",
                        xaxis_title="timestamp",
                        yaxis_title="meter_reading",)
        fig.show()
        

def plot_meter_reading_for_building(df, site_id, building_id, meter_name):
    """
    Plot meter_reading for an entire site for all buildings
    """
    df = df.set_index('timestamp')
    building_id_list = df.columns
    for building_id in building_id_list:
        fig = go.Figure()
        df_subset = df.loc[:, building_id]
        fig.add_trace(go.Scatter(
             x=df_subset.index,
             y=df_subset.values,
             name=f"{meter_name}",
             hoverinfo=f'x+y+name',
             opacity=0.7))

        fig.update_layout(width=1000,
                        height=500,
                        title_text=f"Meter Reading for Site [{site_id}] Building [{building_id}]",
                        xaxis_title="timestamp",
                        yaxis_title="meter_reading",)
        fig.show()      


    
########################################################### EDA ###########################################################

def display_head(df):
    display(df.head(2))

    
def check_null(df):
    print('Checking Null Percentage..')
    return df.isna().sum() * 100/len(df)


def check_duplicate(df, subset):
    print(f'Number of duplicate rows considering {len(subset)} features..')
    if subset is not None: 
        return df.duplicated(subset=subset, keep=False).sum()
    else:
        return df.duplicated(keep=False).sum()

    
def count_unique_values(df, feature_name):
    return df[feature_name].nunique()


def do_value_counts(df, feature_name):
    return df[feature_name].value_counts(normalize=True, dropna=False).sort_values(ascending=False) * 100


def check_id(df, column_name, data_set_name):
    '''
    Check if the identifier column is continous and monotonically increasing
    '''
    print(f'Is the {column_name} monotonic : {df[column_name].is_monotonic}')
    # Plot the column
    ax = df[column_name].plot(title=data_set_name)
    plt.show()
    
    
def get_fetaure_names(df, feature_name_substring) :
    """
    Returns the list of features with name matching 'feature_name_substring'
    """
    return [col_name for col_name in df.columns if col_name.find(feature_name_substring) != -1]


def check_value_counts_across_train_test(train_df, test_df, feature_name, normalize=True):
    """
    Create a DF consisting of value_counts of a particular feature for 
    train and test
    """
    train_counts = train_df[feature_name].sort_index().value_counts(normalize=normalize, dropna=True) * 100
    test_counts = test_df[feature_name].sort_index().value_counts(normalize=normalize, dropna=True) * 100
    count_df = pd.concat([train_counts, test_counts], axis=1).reset_index(drop=True)
    count_df.columns = [feature_name, 'train', 'test']
    return count_df


###################################################### Pre-Processing ######################################################


def fill_with_gauss(df, w=12):
    return df.fillna(df.rolling(window=w, win_type='gaussian', center=True, min_periods=1).mean(std=2))


def fill_with_po3(df):
    df = df.fillna(df.interpolate(method='polynomial', order=3))
    assert df.count().min() >= len(df) - 1 
    # fill the first item with second item
    return df.fillna(df.iloc[1])         


def fill_with_lin(df):
    df =  df.fillna(df.interpolate(method='linear'))
    assert df.count().min() >= len(df) - 1 
    # fill the first item with second item
    return df.fillna(df.iloc[1])         


def fill_with_mix(df):
    df = (df.fillna(df.interpolate(method='linear', limit_direction='both')) +
               df.fillna(df.interpolate(method='polynomial', order=3, limit_direction='both'))
              ) * 0.5
    assert df.count().min() >= len(df) - 1 
    # fill the first item with second item
    return df.fillna(df.iloc[1])


############################################ Feature Engineering ############################################

def create_date_features(df, feature_name):
    '''
    Create new features related to dates
    
    df : The complete dataframe
    feature_name : Name of the feature of date type which needs to be decomposed.
    '''
    df.loc[:, 'year'] = df.loc[:, feature_name].dt.year.astype('uint32')
    df.loc[:, 'month'] = df.loc[:, feature_name].dt.month.astype('uint32')
    df.loc[:, 'quarter'] = df.loc[:, feature_name].dt.quarter.astype('uint32')
    df.loc[:, 'weekofyear'] = df.loc[:, feature_name].dt.weekofyear.astype('uint32')
    
    df.loc[:, 'day'] = df.loc[:, feature_name].dt.day.astype('uint32')
    df.loc[:, 'dayofweek'] = df.loc[:, feature_name].dt.dayofweek.astype('uint32')
    df.loc[:, 'dayofyear'] = df.loc[:, feature_name].dt.dayofyear.astype('uint32')
    df.loc[:, 'is_month_start'] = df.loc[:, feature_name].dt.is_month_start
    df.loc[:, 'is_month_end'] = df.loc[:, feature_name].dt.is_month_end
    df.loc[:, 'is_quarter_start']= df.loc[:, feature_name].dt.is_quarter_start
    df.loc[:, 'is_quarter_end'] = df.loc[:, feature_name].dt.is_quarter_end
    df.loc[:, 'is_year_start'] = df.loc[:, feature_name].dt.is_year_start
    df.loc[:, 'is_year_end'] = df.loc[:, feature_name].dt.is_year_end
    
    df.loc[:, 'hour'] = df.loc[:, feature_name].dt.hour.astype('uint32')
    df.loc[:, 'minute'] = df.loc[:, feature_name].dt.minute.astype('uint32')
    df.loc[:, 'second'] = df.loc[:, feature_name].dt.second.astype('uint32')
    # This is of type object
    df.loc[:, 'month_year'] = df.loc[:, feature_name].dt.to_period('M')
    
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


############################################ Modeling ############################################

def get_data_splits_by_fraction(dataframe, valid_fraction=0.1):
    """
    Creating holdout set from the train data based on fraction
    """
    print(f'Splitting the data into train and holdout with validation fraction {valid_fraction}...')
    valid_size = int(len(dataframe) * valid_fraction)
    train = dataframe[:valid_size]
    validation = dataframe[valid_size:]
    print(f'Shape of the training data {train.shape} ')
    print(f'Shape of the validation data {validation.shape}')
    return train, validation


def get_data_splits_by_month(dataframe, train_months, validation_months):
    """
    Creating holdout set from the train data based on months
    """
    print(f'Splitting the data into train and holdout based on months...')
    print(f'Training months {train_months}')
    print(f'Validation months {validation_months}')
    training = dataframe[dataframe.month.isin(train_months)]
    validation = dataframe[dataframe.month.isin(validation_months)]
    print(f'Shape of the training data {training.shape} ')
    print(f'Shape of the validation data {validation.shape}')
    return training, validation


def train_model(training, validation,predictors, taget,  params, test_X=None):
    
    train_X = training[predictors]
    train_Y = np.log1p(training[target])
    validation_X = validation[predictors]
    validation_Y = np.log1p(validation[target])

    print(f'Shape of train_X : {train_X.shape}')
    print(f'Shape of train_Y : {train_Y.shape}')
    print(f'Shape of validation_X : {validation_X.shape}')
    print(f'Shape of validation_Y : {validation_Y.shape}')
    
    dtrain = lgb.Dataset(train_X, label=train_Y)
    dvalid = lgb.Dataset(validation_X, validation_Y)
    
    print("Training model!")
    bst = lgb.train(params, dtrain, valid_sets=[dvalid], verbose_eval=100)
    
    valid_prediction = bst.predict(validation_X)
    valid_score = np.sqrt(metrics.mean_squared_error(validation_Y, valid_prediction))
    print(f'Validation Score {valid_score}')
    
    if test_X is not None:
        print('Do Nothing')
    else:
        return bst, valid_score


def make_prediction(df_train_X, df_train_Y, df_test_X, params, n_splits=5):
    yoof = np.zeros(len(df_train_X))
    yhat = np.zeros(len(df_test_X))
    cv_scores = []
    result_dict = {}
    
    kf = KFold(n_splits=n_splits, random_state=SEED, shuffle=False)

    fold = 0
    for in_index, oof_index in kf.split(df_train_X, df_train_Y):
        fold += 1
        print(f'fold {fold} of {n_splits}')
        X_in, X_oof = df_train_X.iloc[in_index].values, df_train_X.iloc[oof_index].values
        y_in, y_oof = df_train_Y.iloc[in_index].values, df_train_Y.iloc[oof_index].values
        
        lgb_train = lgb.Dataset(X_in, y_in)
        lgb_eval = lgb.Dataset(X_oof, y_oof, reference=lgb_train)
        
        model = lgb.train(
            params,
            lgb_train,
            valid_sets = [lgb_train, lgb_eval],
            verbose_eval = 50,
        )   
        
        del lgb_train, lgb_eval, in_index, X_in, y_in 
        gc.collect()
        
        print('Training completed')
        yoof[oof_index] = model.predict(X_oof)
        print('OOF Prediction completed.')
        prediction = model.predict(df_test_X.values)
        print('Shape of prediction', prediction.shape)
        yhat += np.expm1(prediction)
        print('Prediction completed')
        cv_oof_score = np.sqrt(metrics.mean_squared_error(y_oof, yoof[oof_index]))
        print(f'CV OOF Score for fold {fold} is {cv_oof_score}')
        cv_scores.append(cv_oof_score)
        
        del oof_index, X_oof, y_oof
        gc.collect()

    yhat /= n_splits

    oof_score = round(np.sqrt(metrics.mean_squared_error(df_train_Y, yoof)), 5)
    avg_cv_scores = round(sum(cv_scores)/len(cv_scores), 5)
    std_cv_scores = round(np.array(cv_scores).std(), 5)

    print(f'Combined OOF score : {oof_score}')
    print(f'Average of {fold} folds OOF score {avg_cv_scores}')
    print(f'std of {fold} folds OOF score {std_cv_scores}')
    
    result_dict['yoof'] = yoof
    result_dict['prediction'] = yhat
    result_dict['oof_score'] = oof_score
    result_dict['cv_scores'] = cv_scores
    result_dict['avg_cv_scores'] = avg_cv_scores
    result_dict['std_cv_scores'] = std_cv_scores
    
    return result_dict

