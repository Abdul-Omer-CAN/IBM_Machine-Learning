# Feature Engineering is creating new features from existing ones to improve ML model performance.
# New concepts in this lesson: 
# pd.get_dummies(one-hot coding) | np.log1p(log transform) 
# PolynomialFeatures(from sklearn) | groupby().transform(category level statistics)

## Import section ##

import numpy as np # For math operations & log transform
import pandas as pd # DataFrames
import matplotlib.pyplot as plt # plotting
import seaborn as sns # statistical plot & pairplot
sns.set() # applies seaborn default styling to all plots

## Load & Explore Data ##

# Load Ames Housing data

df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/Ames_Housing_Data.tsv", sep='\t')

# Examine columns and missing data

df.info()

# Remove outliers recommended by dataset author

df=df.loc[df['Gr Liv Area'] <= 4000, :] # removes houses above 4000 sqft(outliers) | df.loc means select rows and columns by condition. | After comma keep all columns using this ':'
print("Number of rows:", df.shape[0])
print("Number of columns:", df,shape[1])

# Keep a copy of the original data

data=df.copy() # saves original data before we start modifying it.

# Quick look at data

df.head() 

## One Hot Coding ##

# Get all string/categorical columns.

one_hot_encode_cols = df.dtypes[df.dtypes == np.object_] # filter by string categoricals | df.dtypes -> data type of every column. Filter for only object type(strings & text columns). Will return only the categorical columns.
one_hot_encode_cols = one_hot_encode_cols.index.tolist() # convert column names to list

# Preview categorical columns
df[one_hot_encode_cols].head().T # .T means transpose - flips rows and column. Makes it easier to read  when there are many column.

# Apply one hot encoding
df = pd.get_dummies(df, columns=one_hot_encode_cols, drop_first=True) # converts all  categorical columns to 0s and 1s. drop_first removes the first category to avoid multicollinearity( when 2 or more features are highly correlated or contain the same information)
df.describe().T

## Log Transformation ##

# Create list of float columns to check for skewing
mask = data.dtypes == np.float64 # Creates a True or False for each column. True means column is a float & False means not a float.
float_cols = data.columns[mask] # Gets only the column names where mask is True, so float_cols = list of all float columns.


# Define skew limit
skew_limit = 0.75 # If skewness is greater than 0.75 it will apply log transform.
skew_vals = data[float_cols].skew() # Calculates the skewness for each float column. Higher number = more skewed.

# Show skewed columns
skew_cols = (skew_vals
             .sort_values(ascending=False) # sort from skewed to least skewed.
             .to_frame() # Convert series to DataFrame
             .rename(columns={0: 'Skew'}) # Rename column from 0 to skew.
             .query('abs(Skew) > {}'.format(skew_limit))) # Keep only columnns where skewness is > 0.75. abs() catches both + and - skew. .format inserts 0.75 into the query string.

print(skew_cols)

# Apply log transform to skewed columns (skip SalePrice)
for col in skew_cols.index.values: # loops thru all skewed columns
    if col == "SalePrice": # Skips SalePrice(our target we dont transform it)
        continue
    df[col] = df[col].apply(np.log1p) # Apply np.log1p to fix the skew. 1p means 1 plus x -> log(1+x) to prevent zero because log(0) is error!

## Select Key Features ##

smaller_df = df.loc[:, ['Lot Area', 'Overall Qual', 'Overall Cond', 'Year Built', 'Year Remod/Add', 'Gr Liv Area', 'Full Bath', 'Bedroom AbvGr', 'Fireplaces', 'Garage Cars', 'SalePrice']]

# Summary statistics
smaller_df.describe().T

# Check data types and missing values
smaller_df.info()

# Fill missing values with 0
smaller_df = smaller_df.fillna(0)

# Confirm no more missing Values
smaller_df.info()

# We have 80+ columns in the fulldataset. Select only 10 meaningful features + SalePrice
# fillna(0) -> one missing value in Garage Cars, fills with 0
# 2 info() calls -> before and after filling to confirm fix.

## Pairplot ##

sns.pairplot(smaller_df, plot_kws=dict(alpha=.1, edgecolor='none')) # creates a grid of plots for all 11 columns. Diagonal = histograms & Off-Diagonal = scatter plots.
plt.show() # '.1' 10% opacity per dot. Making them transparent shows the density better. edgecolor removes black border around each dot, cleaner look.
# Does SalePrice go up as Overall Qual increases?
# Does SalePrice go up as Year Built increases.