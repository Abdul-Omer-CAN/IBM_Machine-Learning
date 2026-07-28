## LAB ##

## Exercise 1: Exploring Categorical Data.

housing["Sale Condition"].value_counts()

# Shows how many times each category appears in a column. Use '.value_counts()' to explore categorical(text) columns. 
# describe() only works on numerical columns for text columns we use value_counts().

## Exercise 2: Log Transformation.

# First Visualize

housing['Lot Area'].hist()
plt.show()

# Check skewness 

print("Skewness before:", housing['Lot Area'].skew())

# Apply log transformation

la_log = np.log(housing['Lot Area'])
la_log.hist()
plt.show()

# Check skewness after log transformation

print("Skewness after:", la_log.skew())

# Fixes skewed data by applying log transformation. 
# Skewed data is where most values are bunched up on one side of the distribution.
# Log transform pulls extreme values closer and makes a bell curve. That is the function of log to reduce large unusual values.
# Skewedness will be close to zero. Many ML algo perform better with normally distributed data. 

## Exercise 3: Handling Duplicates.

removed_sub = housing.drop_duplicates(subset=['Order'])

# Removes duplicate rows based on the "Order" column. .drop_duplicates() removes them. 
# subset=['Order'] only checks for duplicates in order column.
# without subset it would check for duplicates in all columns.

## Exercise 4: Handling Missing Values.

mean_value = housing['Mas Vnr Area'].mean()
housing['Mas Vnr Area'].fillna(mean_value, inplace=True)

# Fills missing values e.g NaN with the mean of the column.
# Real world data always has missing values & we cant train ML with missing values. It will crash. 
# Common strategies for missing values is.. Fill with mean, median, mode or drop the row.
# fillna() -> Fills NaN values.
# inplace=True -> modifies the DataFrame directly. Wont make a copy of it but will edit the original code.

## Exercise 5: Standardization.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
housing['SalePrice'] = scaler.fit_transform(housing[['SalePrice']])

# This will scale the data so the mean is zero and std is 1.
# ML Model gets confused when features have very different scales. 
# StandardScaler fixes this by putting everything on the same scale. 
# After Scale:mean=0, std=1. The scale will hover around 0 and  values will be negative and positive around zero.
# fit_transform() -> Learns the mean/std then scales in one step.
# housing[['SalePrice']] -> double brackets to keep it as a DataFrame and StandardScaler expects a 2D array.

## Exercise 6: Outlier Detection and Removal.

from scipy import stats

z_score = stats.zscore(housing['Lot Area'])
housing = housing[abs(z_score) < 3]

print("Shape after removing outliers:", housing.shape)

# Finds and removes outliers using Z score.
# Outliers can mislead ML models because they have extreme values that can skew the results.
# Rule is if Z score is greater than 3 or less than -3, it is considered an outlier.
# abs() catches both positive and negative outliers.
# housing[abs(z_score) < 3] -> keeps only non outlier rows in the DataFrame.

## 3 Types of Residual Outliers ##

# 1. Unstandardized Residuals: Raw difference between predicted and actual values. No scaling. Hardest to compare.
# 2. Standardized Residuals: Residuals divided by std. Tells us how many std away from the avg or zero.
# 3. Externally Studentized Residuals: Drops one observation at a time & runs the model without it & compares it to the full model.
# If removing that model changes the model a lot, that observation is an outlier aka asking if this data point didn't exist.