
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor

# Rename all column labels to replace spaces with underscores and convert everything to lowercase. (Underscores can be much easier to work with in Python than spaces.)
def column_name(df1):
    df2 = df1.copy()
    df2 = df2.rename(columns=lambda x: x.strip().lower().replace(" ", "_").replace("(", "").replace(")", ""))
    return df2

# The columns start with 'unrounded' are similar with others
def unro(df1):
    df2 = df1.copy()
    df2 = df2[df2.columns.drop(list(df2.filter(regex='unrounded')))]
    return df2

# drive column has many nulls
def cleanDrive(df1_col):
    df2_col = df1_col.fillna('unknown')
    return df2_col

# transmission, engine_displacement and engine_cylinders columns have few rows with nulls
# Because they are categorical I have to delete them
def dropNull(df1):
    df2 = df1.copy()
    df2 = df2[df2.transmission.notna()] 
    df2 = df2[df2.engine_displacement.notna()]
    df2 = df2[df2.engine_cylinders.notna()]
    df2.engine_cylinders = df2.engine_cylinders.apply(np.int64)
    return df2

# start_stop_technology has yes, no and null values
# it will replace the nul values with U
def cleanStart(df1_col):
    df2_col = df1_col.fillna('U')
    return df2_col

# Turbocharger and Supercharger columns have only  T or S and null values
def cleanCharger(df1_col):
    df2_col = df1_col.fillna('Not')
    return df2_col

# It seems there is no duplicated rows but when look carefully you will see the model variables are duplicated and 
# the attributes are similar
def deDupe(df1):
    df2 = df1.copy()
    df3 = df2.drop_duplicates(subset=['model'], keep='first')
    return df3

# When the dataset is investigated detailed you can eliminate some columns though
def dropRedundant(df1):
    df2 = df1.copy()
    df3 = df2[['year', 'make', 'model', 'class', 'drive', 'transmission', 'engine_cylinders', 
       'engine_displacement', 'turbocharger', 'supercharger', 'fuel_type_1', 'city_mpg_ft1', 'highway_mpg_ft1', 
       'unadjusted_city_mpg_ft1', 'unadjusted_highway_mpg_ft1', 'combined_mpg_ft1', 'tailpipe_co2_in_grams/mile_ft1', 
       'fuel_economy_score', 'ghg_score', 'start_stop_technology']]
    
    return df3

# There are 2 columns stand for fuel type and the columns that are related to calculations. Only some cars are hybrid. 
# So I decided to work on only the cars using 1 fuel type
def fueltype(df1):
    df2 = df1.copy()
    df3 = df2[df2['fuel_type_2'].isna()]
    df4 = df3.drop(['fuel_type', 'fuel_type_2', 'city_mpg_ft2', 'highway_mpg_ft2', 'unadjusted_city_mpg_ft2',
                   'unadjusted_highway_mpg_ft2', 'combined_mpg_ft2', 'annual_fuel_cost_ft2', 
                   'annual_consumption_in_barrels_ft2', 'tailpipe_co2_ft2', 'tailpipe_co2_in_grams/mile_ft2',
                   'composite_city_mpg', 'composite_highway_mpg', 'composite_combined_mpg'], axis=1)
    return df4

# mpg stands for miles per galoon. That gives us the distance a car can drive with a galoon fuel
# But we need the fuel consume. In order to get it I used a transformation
def convertMpg(df1):
    df2 = df1.copy()
    df2['city_lkm'] = 235.214583 / df2['city_mpg_ft1']
    df2['highway_lkm'] = 235.214583 / df2['highway_mpg_ft1']
    df2['unadjusted_city_lkm'] = 235.214583 / df2['unadjusted_city_mpg_ft1']
    df2['unadjusted_highway_lkm'] = 235.214583 / df2['unadjusted_highway_mpg_ft1']
    df2['combined_lkm'] = 235.214583 / df2['combined_mpg_ft1']
    df3 = df2.drop(['city_mpg_ft1', 'highway_mpg_ft1', 'unadjusted_city_mpg_ft1', 'unadjusted_highway_mpg_ft1',
                   'combined_mpg_ft1'], axis=1)
    return df3


# main cleaning function
def cleanDataset(data):
    """
        Cleans the complete fuel economy dataframe.
        Input -> Dataframe to clean
        Output -> The cleaned dataframe
    """
    df1 = data.copy()
    df1 = column_name(df1)
    df1 = unro(df1)
    df1['drive'] = cleanDrive(df1['drive'])
    df1['start_stop_technology'] = cleanStart(df1['start_stop_technology'])
    df1['turbocharger'] = cleanCharger(df1['turbocharger'])
    df1['supercharger'] = cleanCharger(df1['supercharger'])
    df1 = deDupe(df1)
    df1 = fueltype(df1)
    df1 = dropNull(df1)
    df1 = dropRedundant(df1)
    df1 = convertMpg(df1)
    
    return df1

# Transmission column has many variables. To reduce the number of variables I applied a mask
def standardizeTrans(df1):
    df2 = df1.copy()
    df2['transmission'] = df2['transmission'].mask(df2['transmission'].str.startswith('Auto') == True, 'Auto')
    df2['transmission'] = df2['transmission'].mask(df2['transmission'].str.startswith('Man') == True, 'Manual')
    return df2

# Class column has many variables. To reduce the number of variables I applied a mapping
def mapClass(df, col):
    df2 = df.copy()
    mappings = {
    'Compact Cars': 'compact', 'Subcompact Cars': 'compact', 'Sport Utility Vehicle - 4WD': 'sport',
    'Midsize Cars': 'compact', 'Two Seaters': 'compact', 'Minicompact Cars': 'compact',
    'Large Cars': 'compact', 'Special Purpose Vehicles': 'special', 'Sport Utility Vehicle - 2WD': 'sport',
    'Standard Pickup Trucks 2WD': 'van-truck', 'Special Purpose Vehicle 2WD': 'special', 'Small Station Wagons': 'station',
    'Standard Pickup Trucks 4WD': 'van-truck', 'Small Sport Utility Vehicle 4WD': 'sport', 'Midsize-Large Station Wagons': 'station',
    'Midsize Station Wagons': 'station', 'Vans': 'van-truck', 'Standard Pickup Trucks': 'van-truck',
    'Standard Sport Utility Vehicle 4WD': 'sport', 'Small Sport Utility Vehicle 2WD': 'sport', 
    'Special Purpose Vehicle 4WD': 'special', 'Vans, Cargo Type': 'van-truck', 'Vans, Passenger Type': 'van-truck',
    'Minivan - 2WD': 'van-truck', 'Small Pickup Trucks 2WD': 'van-truck', 'Small Pickup Trucks 4WD': 'van-truck',
    'Minivan - 4WD': 'van-truck', 'Small Pickup Trucks': 'van-truck', 'Standard Sport Utility Vehicle 2WD': 'sport',
    'Special Purpose Vehicles/4wd': 'special', 'Special Purpose Vehicles/2wd': 'special', 'Vans Passenger': 'van-truck'}
    df2[col] = df2[col].replace(mappings)
    return df2

# Replacing the categorical values of cylinders, engine_displacement, ghg_score and fuel_economy_score columns with numerical values
def map_values(df, col, col_to_aggregate):
    df2=df.copy()
    col_aggregated = df2.groupby(col)[col_to_aggregate].agg("median").sort_values()
    col_median = dict(col_aggregated)
    col_median_dict = {key: value/col_median[col_aggregated.index[0]] for key, value in col_median.items()}
    df2[col].replace(col_median_dict, inplace=True)

    return df2


# Building linear regression and KNN regressor models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor

def buildModel(X, y, test_size, random_state, neigh_value=3, log=False):
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # split both Train and Test in numericals and categoricals
    numericals_train = X_train.select_dtypes(np.number)
    numericals_test = X_test.select_dtypes(np.number)
    numericals_train.head()
    categoricals_train= X_train.select_dtypes(object)
    categoricals_test= X_test.select_dtypes(object)
    categoricals_train.head()
    
    # transformations on numericals:
    #     fit ONLY on numercals_train
    #     transform BOTH numericals_train and numericals_test
    transformer = StandardScaler()
    transformer.fit(numericals_train)
    numericals_train_standardized = transformer.transform(numericals_train)
    numericals_test_standardized = transformer.transform(numericals_test)
    
    # encoding categoricals
    #     fit ONLY on categricals_train
    #     encode BOTH categoricals_train and categoricals_test
    encoder = OneHotEncoder(handle_unknown='error', drop='first')
    encoder.fit(categoricals_train)
    categoricals_train_encoded = encoder.transform(categoricals_train).toarray()
    categoricals_test_encoded  = encoder.transform(categoricals_test).toarray()
    categoricals_train_encoded = pd.DataFrame(categoricals_train_encoded, columns = encoder.get_feature_names_out())
    categoricals_test_encoded  = pd.DataFrame(categoricals_test_encoded, columns = encoder.get_feature_names_out())
        
    # combine numericals_train and categoricals_train into train_processed
    # combine numericals_test and categoricals_test into test_processed
    X_train = np.concatenate([numericals_train_standardized,categoricals_train_encoded],axis=1)
    X_test = np.concatenate([numericals_test_standardized,categoricals_test_encoded],axis=1)
    
    # define model
    # fit model on train_processed
    # evaluate (score) model on test_processed
    lm = LinearRegression()
    knn = KNeighborsRegressor(n_neighbors=neigh_value, weights="distance")
    if ( log == False ):
        
        lm.fit(X_train,y_train)
        knn.fit(X_train, y_train)
        
        y_pred_train_lm = lm.predict(X_train)
        y_pred_test_lm  = lm.predict(X_test)
        
        y_pred_train_knn = knn.predict(X_train)
        y_pred_test_knn  = knn.predict(X_test)
        
        r2_train_lm  = r2_score(y_train, y_pred_train_lm)
        r2_train_knn = r2_score(y_train, y_pred_train_knn)                                       
        print(f"r2 score train of the Linear Model = {r2_train_lm}")
        print(f"r2 score train of the KNN Model = {r2_train_knn}")  
                                       
        r2_test_lm = r2_score(y_test, y_pred_test_lm)
        r2_test_knn = r2_score(y_test, y_pred_test_knn)                               
        print(f"r2 score test of the Linear Model = {r2_test_lm}")
        print(f"r2 score test of the KNN Model = {r2_test_knn}")
                                       
        mse_lm = mean_squared_error(y_test, y_pred_test_lm)
        mse_knn = mean_squared_error(y_test, y_pred_test_knn)
        print(f'MSE (mean squared error) test of the Linear Model = {mse_lm}')
        print(f'MSE (mean squared error) test of the KNN Model = {mse_lm}')
        
        rmse_lm = np.sqrt(mse_lm)
        rmse_knn = np.sqrt(mse_knn)
        print(f'RMSE (rooted mean squared error) test of the Linear Model = {rmse_lm}')
        print(f'RMSE (rooted mean squared error) test of the KNN Model = {rmse_knn}')
        
        mae_lm = mean_absolute_error(y_test, y_pred_test_lm)
        mae_knn = mean_absolute_error(y_test, y_pred_test_knn)
        print(f'MAE (mean absolute error) test of the Linear Model = {mae_lm}')
        print(f'MAE (mean absolute error) test of the KNN Model = {mae_knn}')
        
        
    elif ( log == True ):
        
        lm.fit(X_train,y_train)
        knn.fit(X_train, y_train)
        
        y_pred_train_log_lm = lm.predict(X_train)
        y_pred_test_log_lm  = lm.predict(X_test)
        y_pred_train_lm = np.exp(y_pred_train_log_lm)
        y_pred_test_lm  = np.exp(y_pred_test_log_lm)
                                       
        y_pred_train_log_knn = knn.predict(X_train)
        y_pred_test_log_knn  = knn.predict(X_test)
        y_pred_train_knn = np.exp(y_pred_train_log_knn)
        y_pred_test_knn  = np.exp(y_pred_test_log_knn)
        
        r2_train_lm  = r2_score(y_train, y_pred_train_lm)
        r2_train_knn = r2_score(y_train, y_pred_train_knn)                                       
        print(f"r2 score train of the Linear Model = {r2_train_lm}")
        print(f"r2 score train of the KNN Model = {r2_train_knn}")  
                                       
        r2_test_lm = r2_score(y_test, y_pred_test_lm)
        r2_test_knn = r2_score(y_test, y_pred_test_knn)                               
        print(f"r2 score test of the Linear Model = {r2_test_lm}")
        print(f"r2 score test of the KNN Model = {r2_test_knn}")
                                       
        mse_lm = mean_squared_error(y_test, y_pred_test_lm)
        mse_knn = mean_squared_error(y_test, y_pred_test_knn)
        print(f'MSE (mean squared error) test of the Linear Model = {mse_lm}')
        print(f'MSE (mean squared error) test of the KNN Model = {mse_lm}')
        
        rmse_lm = np.sqrt(mse_lm)
        rmse_knn = np.sqrt(mse_knn)
        print(f'RMSE (rooted mean squared error) test of the Linear Model = {rmse_lm}')
        print(f'RMSE (rooted mean squared error) test of the KNN Model = {rmse_knn}')
        
        mae_lm = mean_absolute_error(y_test, y_pred_test_lm)
        mae_knn = mean_absolute_error(y_test, y_pred_test_knn)
        print(f'MAE (mean absolute error) test of the Linear Model = {mae_lm}')
        print(f'MAE (mean absolute error) test of the KNN Model = {mae_knn}')
    
    return y_pred_train_lm, y_pred_train_knn, y_pred_test_lm, y_pred_test_knn, y_train, y_test, lm, knn


# Visualising the results
def plot_results(y_train, y_pred_train_lm, y_pred_train_knn, y_test, y_pred_test_lm, y_pred_test_knn):
    '''
    Function to plot sctatterplots of real values against predicted ones. It also
    plots a histogram of the residuals
    
    Inputs:
    y_train -> np.array with the real values in the train set
    y_pred_train -> np.array/pd.Series with the predicted values in the train set
    y_test -> np.array with the real values in the test set
    y_pred_test -> np.array/pd.Series with the predicted values in the test set
    
    Output:
    Display a 4x4 grid of plots with the scatter plots and the histogram of the residuals.
    '''

    fig, ax = plt.subplots(2,2,figsize=(10,5))
    ax[0,0].scatter(x=y_train, y=y_pred_train_lm)
    ax[0,0].set_xlabel("Real fuel consume")
    ax[0,0].set_ylabel("Predicted fuel consume LR")
    ax[0,0].plot(y_train,y_train, color="black")
    ax[0,0].set_title("Train set")
    ax[1,0].scatter(x=y_test, y=y_pred_test_lm)
    ax[1,0].plot(y_test,y_test, color="black")
    ax[1,0].set_xlabel("Real fuel consume")
    ax[1,0].set_ylabel("Predicted fuel consume LR")
    ax[1,0].set_title("Test set")
    ax[0,1].scatter(x=y_train, y=y_pred_train_knn)
    ax[0,1].set_xlabel("Real fuel consume")
    ax[0,1].set_ylabel("Predicted fuel consume KNN")
    ax[0,1].plot(y_train,y_train, color="black")
    ax[0,1].set_title("Train set")
    ax[1,1].scatter(x=y_test, y=y_pred_test_knn)
    ax[1,1].plot(y_test,y_test, color="black")
    ax[1,1].set_xlabel("Real fuel consume")
    ax[1,1].set_ylabel("Predicted fuel consume KNN")
    ax[1,1].set_title("Test set")
    plt.tight_layout()
    plt.show()
