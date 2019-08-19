
<h1 style='text-align:center'> Module 1 Assessment</h1>

This assessment is designed to test your understanding of the Mod 1 material. It covers:

* Python Fundamentals
* Pandas and Numpy
* Data Visualization
* Exploring Statistical Data


Read the instructions carefully. You will be asked both to write code and respond to a few short answer questions.

#### Note on the short answer questions
For the short answer questions _please use your own words_. The expectation is that you have **not** copied and pasted from an external source, even if you consult another source to help craft your response. While the short answer questions are not necessarily being assessed on grammatical correctness or sentence structure, do your best to communicate yourself clearly.

## Python Fundamentals

In the first section, we will work with various Python data types and try to accomplish certain tasks using some Python fundamentals. Below, we've defined a dictionary with soccer player names as keys for nested dictionaries containing information about each players age, nationality, and a list of teams they have played for.   


```python
players = {
	'L. Messi': {
		'age': 31,
		'nationality': 'Argentina',
		'teams': ['Barcelona']
	},
	'Cristiano Ronaldo': {
		'age': 33,
		'nationality': 'Portugal',
		'teams': ['Juventus', 'Real Madrid', 'Manchester United']
	},
	'Neymar Jr': {
		'age': 26,
		'nationality': 'Brazil',
		'teams': ['Santos', 'Barcelona', 'Paris Saint-German']
	},
	'De Gea': {
		'age': 27,
		'nationality': 'Spain',
		'teams': ['Atletico Madrid', 'Manchester United']
	},
	'K. De Bruyne': {
		'age': 27,
		'nationality': 'Belgium',
		'teams': ['Chelsea', 'Manchester City']
	}
}
```

**1) Create a `list` of all the keys in the `players` dictionary. Use python's documentation on dictionaries for help if needed. Store the list of player names in a variable called `player_names` to use in the next question.**


```python
# Get the list of all player names from the dictionary
player_names = None
```

**2) Great! Now that we have each players name, let's use that information to create a `list` of `tuples` containing each player's name along with their nationality. Store the list in a variable called `player_nationalities`**


```python
# Generate list of tuples such that the first element in the tuple is 
# a players name and the second is their nationality 
# Ex: [('L. Messi', 'Argentina'), ('Christiano Ronaldo', 'Portugal'), ...]
player_nationalities = None
```

**3) Now, define a function called `get_players_on_team` that returns a `list` of the names of all the players who have played on a given team.** 

Your function should take two arguments: 
* a dictionary of player information
* a `string` of the team you are trying to find the players for 

**Be sure that your function has a `return` statement.**


```python
# Define your get_players_on_team function here.
```


```python
players_on_manchester_united = get_players_on_team(players,'Manchester United')
```

## Pandas and Numpy

In this section you will be doing some preprocessing and exploratory data analysis for a dataset for the videogame FIFA19 (https://www.kaggle.com/karangadiya/fifa19).  The dataset contains both data for the game as well as information about the players' real life careers. 

**1) Read the CSV file into a pandas dataframe**

The data you'll be working with is found in a file called './data/fifa.csv'.  Use your knowledge of pandas to create a new dataframe using the csv data. 

Check the contents of your dataframe with `df.head()`</b>


```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
```


```python
df = None
df.head()
```

**2) Check for duplicates**
    
**First, check how many columns and rows are in the dataset, then check how many unique values are in the "ID" column.**


```python
# code here to see the size of the dataframe

```


```python
# code here to check number of unique ids

```

<b> 3) Drop Duplicates
    
It looks like there are duplicates.  Get rid of them by dropping duplicate rows. After you have dropped them, see how many rows are remaining.</b>


```python
# code here

```


```python
# now see how many rows there are

```

<b> 4. Drop n/a rows for "Release Clause"
    
Drop rows for which "Release Clause" is none or not given. This is part of a soccer player's contract dealing with being bought out by another team. Release Clause will be the target variable for our regression model.  After you have dropped them, see how many rows are remaining.</b>


```python
# code here to drop n/a rows

```


```python
# now check how many rows are left 

```

<b> 5) Convert players' heights to inches. Replace the original height column.
First create a function, then use it on your dataframe. Create a function that convert a string into a integer and then apply that function to all of the height column.</b>


```python
# code here to write a helper function
def convert_height(height):
    '''
    inputs: height (string)
    ----
    returns: height in inches (int)
    '''
    pass
```


```python
# test here
convert_height("5'7")
```


```python
# code here to use the function on the height column

```

## Data Visualization

<b> 1) Make a histogram of players age
    
_Add a title and x axis label._ Use whichever plotting library you are most comfortable with. </b>


```python
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
# histogram

```

<b> 2) Make a bar chart for the 10 countries with the most players (by nationality)</b>

Make sure to include x labels on your chart!


```python
# code here to get the top 10 countries with the most players

```


```python
# code here to plot a bar chart
plt.subplots(figsize=(10,6))

```

<b> 3) Make a scatter plot for the player stats StandingTackle and SlidingTackle

What can we say about these two features? </b>


```python
# code here to plot a scatterplot


```


```python
# Your written answer here
```

### Exploring Statistical Data

We'll continue using the same FIFA 2019 dataset.  This section will assess your ability to use numpy and work with summary statistics.

<b>1) Convert the Release Clause Price from Euros to Dollars
    
Create a new column that has the 'release_clause' in dollars.

1.2 Dollars = 1 Euro.</b>


```python
# code here to convert the column of euros to dollarss

```

<b>2) Get summary statistics for all numeric columns
    
(Please don't do each column individually!)</b>


```python
# code here

```

<b>3) What is the mean age and the median age for the players in this dataset?  How are the mean and median related to each other?</b>


```python
# code here
```


```python
# Your written answer here
```

#### 4) Who is the oldest player in Argentina and how old is he?  


```python
# code here
```


```python
# Your written answer here
```


