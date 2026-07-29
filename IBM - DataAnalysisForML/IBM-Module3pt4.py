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

## Polynomial Features ##

# Set up X(features) & Y(Target)
y = smaller_df['SalePrice'] # Seperate the target what we want to predict
x = smaller_df.drop['SalePrice', axis=1] # Drop SalePrice from features. drop a column(axis=1) 0 would be rows. X now has only 10 feature columns.

# Copy X for polynomial features
X2 = X.copy() # Make a copy before adding new features.

# Add squared terms
X2['OQ2'] = X2['Overall Qual'] ** 2 # Create new feature = Overall Quality squared
X2['GLA2'] = X2['Gr Liv Area'] ** 2 # Create new features = Living Area squared
# The reason why we square it is because we want to amplify the difference between low and high quality. Helps the model capture non linear relationships.

## Interaction Terms ##

X3 = X2.copy() # copies X2 which has the squared features

# Multiplicative interaction - quality  x year built
X3['OQ_x_YB'] = X3['Overall Qual'] * X3['Year Built'] # Multiply quality x Year Built | Captures "A high quality new house is worth  more than a high quality OLD House"

# Division interaction - quality / lot area
X3['OQ_/_LA']= X3['Overall Qual'] / X3['Lot Area'] # Divide Quality / Lot Area | Captures "quality per sqaure foot"

## Category Features ##

# Check house style categories and counts. Counts how many houses of each style exist.
data['House Style'].value_counts()

# Preview one-hot encoding on House-Style. Each house becomes its own 0/1 column.
pd.get_dummies(df['House Style'], drop_first=True).head()

# Check neighborhood counts. 
nbh_counts = df.Neighborhood.value_counts()
print(nbh_counts)

# Find neighborhood with 8 or fewer houses (rare categories). Too few examples = unreliable category.
other_nbhs = list(nbh_counts[nbh_counts <= 8].index)
print(other_nbhs)

# Copy X3 and add Neighborhood feature
X4 = X3.copy()

# Replace rare neighborhoods with 'Other'. Replace all rare neighborhoods with Other. Groups them together so the model has enough data to learn.
X4['Neighborhood'] = df['Neighborhood'].replace(other_nbhs, 'Other')

## One-Hot Coding & Drop First Rule $$

# ML models cant understand text. one hot coding converts categories to 0s and 1s.
# Each row gets a 1 for its category and 0 for everything else.
# drop_first=True removes one column because its always predictable from the others. 
# Example: if 1Story=0 and 1.5=0 -> It has to be a 2 Story. The 2 story column is redunant.
# This is called a dummy variable trap = keeping all columns causes multicollinearity.
# Rule: always n-1 columns where n=number of categories.

## Deviation Features ##

def add_deviation_feature(X, feature, category): # Define a reusable fxn. X is the DataFrame. feature is the column we want to measure (e.g Year Built). category is the group we compare with within(e.g House Style   )

    # Group by category
    category_gb = X.groupby(category)[feature] # Group data by category(e.g house style) & Focus on one feature(e.g Year Built)

    # Calculate mean and std for each category
    category_mean = category_gb.transform(lambda x: x.mean()) # Calculate mean Year built for each House Style group.
    category_std = category_gb.transform(lambda x: x.std()) # Calculates standard deviation per group. How spread out are the Year Built values within each style.

    # Calculate how far each value is from its category mean
    deviation_feature = (X[feature] - category_mean) / category_std # How many standard deviations is this house from its group's mean?
    X[feature + '_Dev_' + category] = deviation_feature # Creates a new column name with a descriptive name e.g like Year Built_Dev_House Style. Stores the Deviation values there.

    # Create X5and add deviation features
    X5 = X4.copy()
    X5['House Style'] = df['House Style'] # Add house style column back. we need it for grouping.
    add_deviation_feature(X5, 'Year Built', 'House Style') # Call our function. How old is this house compared to other houses of the same style.
    add_deviation_feature(X5, 'Overall Qual', 'Neighborhood') # How good is this house compared to other houses in the same neighborhood.

    ## Sklearn Polynomial Features ##

    from sklearn.preprocessing import PolynomialFeatures

    # Create polynomial features object - degree 2. degree 2 includes original features, squared features AND cross terms
    pf = PolynomialFeatures(degree=2)

    # Select features to apply polynomial to
    features= ['Lot Area', 'Overal Qual']

    # Fit the polynomial features. Learn the feature names and structure.
    pf.fit(df[features])

    # See what features were created
    print(pf.get_feature_names_out(input_features=features)) # Show all features it will create: 1, Lot Area, Overall Qual, Lot Area², Lot Area×Overall Qual, Overall Qual²

    # Transform and create DataFrame
    feat_array= pf.transform(df[features]) # Actually creates all those new features.
    pd.DataFrame(feat_array, columns=pf.get_feature_names_out(input_features=features) # Converts numpy array into a proper DataFrame. feat_array= the numbers. columns=... gives us the  column names instead of numeric labels.



