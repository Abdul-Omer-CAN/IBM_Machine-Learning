# Detecting Outliers we plot either a histogram and density plot  OR a boxplot
# Plot a Histogram and Density Plot -> sns.distplot(data, bins=20);
# Plot a Boxplot -> sns.boxplot(data);

# Import Section

import numpy as np
import pandas as pd
import sqlite3 as sq3

# Load Data

path = 'data/classic_rock.db' # Path to database file   
con = sq3.Connection(path) # open connection to database
query = 'SELECT * FROM rock_songs;' # SQL Query - give me all columns from the rock_songs table. ; end of query. 
data = pd.read_sql(query, con) # run query, return as DataFrame. pd.read_sql is a panda fxn that talks to the database.


# Calculate the interquartile range

q25, q50, q75 = np.percentile(data['PlayCount'].dropna(), [25, 50, 75]) # np.percentile calculates the percentiles of PlayCount column. dropna removes the missing values. Calculates the 25, 50 and 75th percentile. store each percentile in its own variable(q25, q50,q75)
iqr = q75 - q25 # IQR stands for interquartile range. distance between the 25th and 75th percentile. 

# Calculate the min / max limits to be considered an outlier

min = q25 - 1.5*(iqr) # Anything below this value is an outlier.
max = q75 + 1.5*(iqr) # Anything above this value is an outlier.

print(min, q25, q50, q75, max)


