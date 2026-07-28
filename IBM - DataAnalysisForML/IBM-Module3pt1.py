## EDA Lab - Iris Dataset ##

# Import Section #

import os # interact with opearting system such as file paths and directories.
import numpy as np # contains math operations such as arrays, percentiles, log etc
import pandas as pd  # contains data manupulation such as DataFrame, read_csv etc
import matplotlib.pyplot as plt # plotting such as histograms, scatter plots etc
import seaborn as sns # Advanced plotting - built on matplotlib, but gives us prettier graphs.

# Question 1 - Load Data and Explore #

data = pd.read_csv("URL")

# View first five rows 

data.head()

# Number of rows 

print(data.shape[0])

# Column names

print(data.columns.tolist())

# Data types of each column

print(data.dtypes)

# Question 2 - Clean Species Name #

data['species']  = data.species.str.replace('Iris-', '') # data.species.str accesses string methods on the species column & .replace will replace Iris with nothing an empty string.
data.head() # Show first  5 rows to confirm the change worked.

# Question 3 - Count each species and calculate the stats

# Count each species
print(data.species.value_counts()) # Count how many times each species appears.

# Calculate statistics
stats_df = data.describe() # Gives us a summary of these stats -> count, mean, std, min, 25%, 50%, 75%, max. Stores it in stats_df.

# Add range row (max - min)
stats_df.loc['range'] = stats_df.loc['max'] - stats_df.loc['min'] #  subtracts the max and min and adds a new row called the 'range'.

# Select specific rows
out_fields = ['mean', '25%', '50%', '75%', 'range'] # A list of which row we want to keep.
stats_df = stats_df.loc[out_fields] # Keep only the rows in our list. Filters out the rows we dont need.

# Rename 50% to median
stats_df.rename({'50%': 'median'}, inplace=True) # Rename the 50% row to median. Makes it more readable.
print(stats_df)

# Question 4 - Calculate mean and median per species.

# Mean per species
print(data.groupby('species').mean()) # groups the daa by species. Then calculates the mean of each measurement for species.

# Median per species
print(data.groupby('species').median()) # Same as above.

# Both mean and median in one table
print(data.groupby('species').agg(['mean', 'median'])) # .agg means aggregate. Apply multiple fxns at once. Returns both mean and median in one table. Better and optimizes everything.

# Question 5 - Scatter plot

ax = plt.axes() # creates a blank plot/canvas to draw on.

ax.scatter(data.sepal_length, data.sepal_width) # Draws a scatterplot. X axis sepal_length & Y axis sepal_width.

ax.set(xlabel='Sepal Length (cm)' # Sets label and title all in one line.
       ylabel='Sepal Width (cm)'
       title='Sepal Length vs Width')

plt.show() # Displays the plot.

# Question 6 - Histogram 

ax = plt.axes()
ax.hist(data.petal_length, bins=25) # draws a histogram of petal_length, splits data into 25 bars. More bins = more details.

ax.set(xlabel='Petal Length (cm)',
       ylabel='Frequency', # ylabel will show how many flowers fall in each bin
       title='Distribution of Petal Lengths')

plt.show()

# Question 7 - Multiple Histograms Overlayed

ax = data.plot.hist(bins=25, alpha=0.5) # Plots ALL numeric columns as histograms on ONE Chart.
ax.set_xlabel('Size (cm)') # 50% transparency so overlapping bars are visible.
plt.show()

# 4 seperate plots in one figure

axList = data.hist(bins=25) # Creates 4 seperate histograms - one per feature(We have 4 numeric columns). Returns a list of axes called axList.
# We have 4 numeric columns - sepal width and length. Petal width and length.

for ax in axList.flatten(): # Converts 2D grid of plots into a flat list so we can loop through them.
    if ax.is_last_row(): #  Add label to last row
        ax.set_xlabel('Size (cm)')
    if ax.is_first_col(): # Add label to first column
        aax.set_ylabel('Frequency')

plt.show()

## Question 8 - Boxplot ##

data.boxplot(by='species') # creates a boxplot for each measurement & groups it by species so you can compare(setosa, versicolor, virginica)
plt.show()

## Question 9 - Seaborn Boxplot ##

# Reshape the data first
plot_data = (data
             .set_index('species') # make species the index
             .stack() # collapse all  measurement columns into one column
             .to_frame() # convert to DataFrame
             .reset_index() # bring species back as a column
             .rename(columns={0:'size', 'level_1':'measurement'}) # give columns proper columns. pandas automatically name the columns 0 and level_1 so we are renaming them here.
             )

# Now plot
sns.set_style('white')
sns.set_context('notebook')
sns.set_palette('dark')

f = plt.figure(figsize=(6,4))
sns.boxplot(x='measurement', y='size', hue='species', data=plot_data) # hue means color each species differently. 
plt.show()

# Question 10 - Pairplot

sns.set_context('talk') # Makes the plot bigger and text larger
sns.pairplot(data, hue='species') # Oneline that creates a grid of plots. Every feature plotted against every other feature. hue='species' means different color for each species.
plt.show()

# In a pairplot features are listed as both rows and columns creating a grid. Where a feature meets itself on the diagonal. It shows
# a histogram(diistribution of that one feature). Where 2 different features meet off the diagonal & it will show a scatter plot.