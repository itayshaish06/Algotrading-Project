import pandas as pd
import numpy as np
import logging
import os
from catboost import CatBoostClassifier, Pool
from catboost.utils import get_fpr_curve
from sklearn.model_selection import cross_val_score
from time import sleep
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_curve, precision_recall_curve, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import optuna
import joblib

# -- logger functions --
def define_logger(year):
    """Define a logger for the given year."""
    logger = logging.getLogger(f'logger_{year}')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f'logs/{year}.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f'---------------------------------------- Starting year: {year} ----------------------------------------')
    return logger, file_handler

def close_logger(logger, file_handler, year):
    """Close the logger for the given year."""
    logger.info(f'---------------------------------------- Finished year: {year} ----------------------------------------')
    file_handler.close()
    logger.removeHandler(file_handler)
    logging.shutdown()

# -- scaling functions --
def scale_columns_train_data(data, columns, scaler, year, save_scaler = True):
    """Scale the columns of the training data. 
        Return the scaled data. 
            Save the scaler to a file."""
    if len(columns) > 0:
        data[columns] = scaler.fit_transform(data[columns])
        if save_scaler:
            joblib.dump(scaler, f'models/scaler{year}.pkl')
        return data
    return data

def scale_columns_test_data(data, columns, scaler):
    """Scale the columns of the validation data using the pre-trained scaler."""
    if len(columns) > 0:
        data[columns] = scaler.transform(data[columns])
        return data
    return data

# -- training functions --
def calc_metrics(y_test, y_pred, explicit_msg = '', logger = None):
    """Calculate and log the metrics for the given test data."""
    if logger is not None:
        logger.info(f'{explicit_msg}Classification Report:\n {classification_report(y_test, y_pred)}')
        logger.info(f'{explicit_msg}Acuracy Score: {accuracy_score(y_test, y_pred)}\n')

def optuna_optimization(data, category_columns, columns_to_scale, scaler, logger, file_handler, train_end = 2016, valid_start = 2017, valid_end = 2019, number_of_trials=100):
    """Optimize the hyperparameters of the model using Optuna.
        Training data is for years 2005-2016.
        Validation period is 2017-2019."""
    logger.info(f'Optuna optimization for years: 2005-{train_end}')
    try:
        # -- prepare train data --
        train_data = data[data['Year'] <= train_end]
        train_data = train_data.drop(columns=['Year']).sort_values(by='date').reset_index(drop=True)
        train_data = scale_columns_train_data(train_data, columns_to_scale, scaler, train_end, save_scaler=False)
        if train_data.isnull().sum().sum() > 0:
            print(f'null values in data: {train_data.isnull().sum()}')
            close_logger(logger, file_handler, train_end)
            return train_end
        logger.info(f'Training data shape: {train_data.shape}')
        featurs_start_col = 2 # columns 0,1 are date and symbol and are not needed for the model
        X_train = train_data.iloc[:, featurs_start_col:-1]
        y_train = train_data.iloc[:, -1]
        logger.info(f'\nexample: 2 rows of training data:\n {X_train.head(2)}\n')
        train_pool = Pool(data=X_train, label=y_train, cat_features=category_columns)

        # -- prepare validation data --
        validation_data = data[(data['Year'] >= valid_start) & (data['Year'] <= valid_end)]
        validation_data = validation_data.drop(columns=['Year']).sort_values(by='date').reset_index(drop=True)
        validation_data = scale_columns_test_data(validation_data, columns_to_scale, scaler)
        if validation_data.isnull().sum().sum() > 0:
            print(f'null values in data: {validation_data.isnull().sum()}')
            close_logger(logger, file_handler, train_end)
            return train_end
        logger.info(f'Test data shape: {validation_data.shape}')
        X_test = validation_data.iloc[:, featurs_start_col:-1]
        y_test = validation_data.iloc[:, -1]
        logger.info(f'\nexample: 2 rows of test data:\n {X_test.head(2)}\n')
        test_pool = Pool(data=X_test, label=y_test, cat_features=category_columns)

        def objective(trial):
            param = {
                'iterations': trial.suggest_int('iterations', 100, 1200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'depth': trial.suggest_int('depth', 4, 8),
                'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 3, 9),
                'border_count': trial.suggest_int('border_count', 32, 254),
                'loss_function': trial.suggest_categorical('loss_function', ['Logloss', 'MultiClass']),
                'cat_features': category_columns,
                'random_strength': trial.suggest_float('random_strength', 0.1, 10.0),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'feature_border_type': trial.suggest_categorical('feature_border_type', ['GreedyLogSum', 'Median', 'Uniform', 'MinEntropy']),
                'leaf_estimation_method': trial.suggest_categorical('leaf_estimation_method', ['Newton', 'Gradient']),
                'od_pval': trial.suggest_float('od_pval', 0.001, 0.5),
                'class_weights': [1, 1.2]  # Adjust weights as needed
            }
            classifier = CatBoostClassifier(**param, random_seed=42,silent=True)
            classifier.fit(train_pool)
            y_pred = classifier.predict(X_test)

            # -- filter the type of metric to return
            return balanced_accuracy_score(y_test, y_pred)
            # return recall_score(y_test, y_pred, average='micro')  
            # return recall_score(y_test, y_pred, average='weighted')  
            # return f1_score(y_test, y_pred, average='weighted')  
            # return recall_score(y_test, y_pred, average='macro')  
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=number_of_trials)
        logger.info(f'\tBest parameters: {study.best_params}')
        logger.info(f'\tBest value: {study.best_value}')
        classifier = CatBoostClassifier(**study.best_params, random_seed=42,silent=True)
        classifier.fit(train_pool)
        y_pred = classifier.predict(test_pool)
        perform_log_metrics(valid_start, valid_end, classifier, test_pool, validation_data, y_test, logger)
        classifier = None
    except Exception as e:
        logger.info(f'failed to optimize - {e}')

def cross_validation(classifier, X_train, y_train, logger, year):
    """Perform cross validation for the train data."""
    logger.info(f'\tstarting cross validation for year: {year}')
    accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
    logger.info(f'\tAccuracy: {(accuracies.mean()*100):.2f} %"')
    logger.info(f'\tStandard Deviation: {(accuracies.std()*100):.2f} %\n')

def create_fpr_curve(classifier, X_test, y_test, logger, year):
    """Create and Saves the FPR curve for the test data."""
    # Get FPR curve
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    fpr_curve, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr_curve, tpr, label='FPR Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('FPR Curve')
    plt.legend()
    plt.savefig(f'plots\\fpr_curve\\fpr_curve_{year}.png')
    logger.info(f'FPR curve saved for year: {year}')
    plt.close()

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(f'plots\\precision_recall\\precision_recall_curve_{year}.png')
    logger.info(f'Precision-Recall curve saved for year: {year}')
    plt.close()

def feature_importance(classifier, train_pool, X_train, logger):
    """Calculate and log the feature importance of the model."""
    logger.info(f'Feature importances:')
    feature_importances = classifier.get_feature_importance(train_pool)
    for feature, importance in zip(X_train.columns, feature_importances):
        logger.info(f'\t\t{feature}: {importance:.4f}%')

def perform_log_metrics(start_year, end_year, classifier, test_pool, test_data, y_test, logger):
    """Perform metrics for test data."""
    logger.info(f'\tstarting test prediction for {start_year}-{end_year} \n')
    #report for the test data
    y_pred = classifier.predict(test_pool)
    y_prob = classifier.predict_proba(test_pool)
    
    test_data['predicted_y'] = y_pred
    test_data['prob_y_0'] = y_prob[:,0]
    test_data['prob_y_1'] = y_prob[:,1]

    test_data.to_excel(f'plots\\test_excels\\test_{start_year}_{end_year}.xlsx')
        
    logger.info(f'Validation period: {start_year}-{end_year} ')

    #calc metrics for the whole test data
    calc_metrics(y_test, y_pred, 'All gaps:: ', logger)

    #calc metrics for the up gaps
    up_gaps_y = test_data[test_data['gap_direction'] == 'Up']['y']
    up_gaps_y_pred = test_data[test_data['gap_direction'] == 'Up']['predicted_y']
    calc_metrics(up_gaps_y, up_gaps_y_pred, 'Up gaps:: ', logger)

    #calc metrics for the down gaps
    down_gaps_y = test_data[test_data['gap_direction'] == 'Down']['y']
    down_gaps_y_pred = test_data[test_data['gap_direction'] == 'Down']['predicted_y']
    calc_metrics(down_gaps_y, down_gaps_y_pred, 'Down gaps:: ', logger)

def execute(year, data, dd, perform_cross_val=False, perform_train=False, perform_test_metrics=False, columns_to_drop = None, fpr_curve = False, optuna_opt = False, gap_pct_limit = None):
    try:
        # -- filter data if needed --
        if gap_pct_limit is not None:
            data = data[data['gap_pct'].abs() <= gap_pct_limit]
            print(f'gap_pct limit: {gap_pct_limit}')
        if columns_to_drop is not None and len(columns_to_drop) > 0:
            data.drop(columns=columns_to_drop, inplace=True)

        # -- define logger --
        if optuna_opt:
            logger, file_handler = define_logger('optuna_logger')
        else:
            logger, file_handler = define_logger(year)

        # -- define category columns and scale columns --
        category_columns = ['gap_direction','industry','sub_industry','gap_volume_direction','direction of last day']
        category_columns = [col for col in category_columns if col not in columns_to_drop] # if col not in columns_to_drop than it stays in the list
        data[category_columns] = data[category_columns].astype('category')
        scaler = StandardScaler()
        columns_to_scale = ['RSI','ATR','VIX','distance_from_prev_gap','day','month','SMA 5','SMA 10','SMA 20','distance_from_prev_gap','day','week','month']
        columns_to_scale = [col for col in columns_to_scale if col not in columns_to_drop] # if col not in columns_to_drop than it stays in the list

        # -- Optuna optimization --
        if optuna_opt:
            train_end = 2016
            if year != train_end:
                logger.info(f'Optuna optimization is only for {train_end}')
                close_logger(logger, file_handler, year)
                return year
            
            # define the validation period
            validation_start = train_end + 1
            validation_end = train_end + 3 

            number_of_trials = 200

            # perform the optimization
            optuna_optimization(data, category_columns, columns_to_scale, scaler, logger, file_handler, train_end, valid_start=validation_start,valid_end=validation_end, number_of_trials=number_of_trials)
            close_logger(logger, file_handler, year)
            return year

        # -- prepare train data --
        train_end = year
        train_data = data[data['Year'] <= train_end]
        train_data = train_data.drop(columns=['Year']).sort_values(by='date').reset_index(drop=True)
        train_data = scale_columns_train_data(train_data, columns_to_scale, scaler, year)
        if train_data.isnull().sum().sum() > 0:
            print(f'null values in data: {train_data.isnull().sum()}')
            close_logger(logger, file_handler, year)
            return year
        logger.info(f'Training data shape: {train_data.shape}')
        featurs_start_col = 2 # columns 0,1 are date and symbol and are not needed for the model
        X_train = train_data.iloc[:, featurs_start_col:-1]
        y_train = train_data.iloc[:, -1]
        logger.info(f'\nexample: 2 rows of training data:\n {X_train.head(2)}\n')
        train_pool = Pool(data=X_train, label=y_train, cat_features=category_columns)

        # -- prepare test data --
        valid_start = train_end + 1 # if train_end < 2019 else 2020
        valid_end = train_end + 3 if train_end < 2019 else 2023
        test_data = data[(data['Year'] >= valid_start) & (data['Year'] <= valid_end)]
        test_data = test_data.drop(columns=['Year']).sort_values(by='date').reset_index(drop=True)
        test_data = scale_columns_test_data(test_data, columns_to_scale, scaler)
        if test_data.isnull().sum().sum() > 0:
            print(f'null values in data: {test_data.isnull().sum()}')
            close_logger(logger, file_handler, year)
            return year
        logger.info(f'Test data shape: {test_data.shape}')
        X_test = test_data.iloc[:, featurs_start_col:-1]
        y_test = test_data.iloc[:, -1]
        logger.info(f'\nexample: 2 rows of test data:\n {X_test.head(2)}\n')
        test_pool = Pool(data=X_test, label=y_test, cat_features=category_columns)
        
        # -- define classifier --
        params = {'iterations': 1099, 'learning_rate': 0.10609132448935872, 'depth': 4, 'l2_leaf_reg': 7, 'border_count': 188, 'loss_function': 'MultiClass', 'random_strength': 5.886893143014055, 'bagging_temperature': 0.5673309707166335, 'feature_border_type': 'GreedyLogSum', 'leaf_estimation_method': 'Newton', 'od_pval': 0.2614023227455215}
        classifier = CatBoostClassifier(**params, cat_features=category_columns, verbose=100)

        # -- Cross Validation --
        if perform_cross_val:
            cross_validation(classifier, X_train, y_train, logger, year)
        
        # -- Train Model --
        if perform_train:
            logger.info(f'\tstarting fitting model for years: 2005-{year}\n')
            classifier.fit(train_pool, eval_set=test_pool, plot=False)
            classifier.save_model(f'models/catboost_model_{year}.cbm', format='cbm')

        # -- FPR curve --
        if fpr_curve:
            create_fpr_curve(classifier, X_test, y_test, logger, year)

        # -- Feature Importance --
        feature_importance(classifier, train_pool, X_train, logger)

        # -- Validation --
        if perform_test_metrics:
            perform_log_metrics(valid_start, valid_end, classifier, test_pool, test_data, y_test, logger)

        #close logger
        close_logger(logger, file_handler, year)

        return year
    except Exception as e:
        logger.info(f'failed to execute - {e}')
        close_logger(logger, file_handler, year)
        return year

if __name__ == '__main__':
    # --- Ensure directories exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs(f'plots\\precision_recall', exist_ok=True)
    os.makedirs(f'plots\\fpr_curve', exist_ok=True)
    os.makedirs(f'plots\\test_excels', exist_ok=True)
    dd_threshold = 0.05

    # --- Boolean flags for the different tasks
    optuna_opt = False
    perform_cross_val = False

    perform_train = True
    perform_test_metrics = perform_train
    fpr_curve = perform_train
    gap_pct_limit = None #0.1 #None # -> if the user wants to limit the gap_pct for the training session 

    columns_to_drop = ['gap_volume_change','gap_volume_direction','direction of last day','ATR', 'SMA 5', 'SMA 10', 'SMA 20', 'distance_from_prev_gap', 'month', 'industry', 'sub_industry'] # -> columns that was not helpful for the model
    # columns_to_drop = [] # uncomment if you want to keep all columns

    counter = 1

    data = pd.read_pickle(f'examples producing\\catBoost Data\\catboost_data{dd_threshold}.pickle')
    data['Year'] = data['date'].dt.year
    data = data.dropna()

    if optuna_opt:
        execute(2016, data.copy(), dd_threshold, perform_cross_val, perform_train, perform_test_metrics, columns_to_drop, fpr_curve, optuna_opt, gap_pct_limit)
        exit()

    start_year_model = 2010
    end_year_model = 2023
    for year in range(start_year_model, end_year_model):
        execute(year, data.copy(), dd_threshold, perform_cross_val, perform_train, perform_test_metrics, columns_to_drop, fpr_curve, optuna_opt, gap_pct_limit)
        print(f'finished task {counter}/{end_year_model-start_year_model} - year: {year}')
        counter += 1