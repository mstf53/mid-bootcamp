
import pandas as pd
import numpy as np

# Rename all column labels to replace spaces with underscores and convert everything to lowercase. (Underscores can be much easier to work with in Python than spaces.)
def column_name(df1):
    df2 = df1.copy()
    df2 = df2.rename(columns=lambda x: x.strip().lower().replace(" ", "_").replace("(", "").replace(")", ""))
    return df2

# The columns start with unrounded are similar with others
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
