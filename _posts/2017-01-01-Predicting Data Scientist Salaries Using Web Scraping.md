---
layout: post
title: Predicting Data Scientist Salaries Using Logistic Regression
---

This project began as a web scraping exercise for our Data Science Immersive course which turned into a project in which we use Logistic Regression to predict whether a Data Science job listing in Boston will pay over $90,000.

The class brainstormed what predictors might be useful for a model and decided on four:

- Years of experience
- Whether the term 'scientist' is in the job title
- Whether the term 'start-up' appears in the job description
- Whether the term 'Phd' appears in the job description

The class split into teams to perform the web scraping, each team responsible for one of the four features.  Once the data was gathered we individually cleaned and merged the data and built models to attempt to predict whether listings had a salary of over $90K.

## Importing and Cleaning the CSVs

Each job's feature was stored in seperate CSV files with a job ID and the datapoint.  Each CSV had to be imported and cleaned, then they were merged using the job ID in order to conduct further analysis using a single dataframe.

See Appendix A below to for the code that performed the clean and merge.

## Importing necessary libraries


```python
import pandas as pd
import random
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
```

# Establish baselines to compare model against

To determine how well our model is performing I will need to compare how it does vs a random guess whether a job listing pays over 90K.

Guessing gives an accuracy of 51.9%.  Almost a perfect 50/50.

I also want to see how accurate taking the assumption that every job is listed over 90K, or that every job is listed under 90K is:

All jobs are over90K: 53.9% accuracy
All jobs are under90K: 46.1% accuracy


```python
random_guess = []
all_over_90K = []
all_under_90K = []
#y is the prediction target set
for i in y:
    random_guess.append(random.randint(0,1))
    all_over_90K.append(1)
    all_under_90K.append(0)
print("Random Guess: {0:.2%}".format(accuracy_score(y, random_guess)))
print("All Over90K: {0:.2%}".format(accuracy_score(y, all_over_90K)))
print("All Under90K: {0:.2%}".format(accuracy_score(y, all_under_90K)))
#{percent:.2%}'.format(percent=1.0/3.0)
```

    Random Guess: 51.88%
    All Over90K: 53.90%
    All Under90K: 46.10%


## Making a Logistic RegressionCV Model

Now that the data is ready and the baseline performance is established it's time to build a model and see if it accurately predicts whether a job pays over 90K.

Here I chose sci-kit learn's Logistic RegressionCV and train and evaluate based on the four features the team originally brainstormed for predictors:
    - Years of experience
    - Whether the term 'scientist' is in the job title
    - Whether the term 'start-up' appears in the job description
    - Whether the term 'Phd' appears in the job description


```python
from sklearn.linear_model import LogisticRegressionCV
X = df[[
        'years',
        'scientist_in_title',
        'has_startup',
        'has_phd'
    ]]
y = df.over_90k

model = LogisticRegressionCV(cv=3)
model.fit(X, y) # This trains the model using 3 cross validated sets from the data

for metric in ['accuracy', 'precision', 'recall', 'roc_auc']:
    scores = cross_val_score(model, X, y, scoring=metric)
    print("mean {}: {}, all: {}".format(metric, scores.mean(), scores))
```

    mean accuracy: 0.5671291451994911, all: [ 0.59437751  0.56854839  0.53846154]
    mean precision: 0.6269932389335374, all: [ 0.64102564  0.70149254  0.53846154]
    mean recall: 0.6368159203980099, all: [ 0.55970149  0.35074627  1.        ]
    mean roc_auc: 0.6059093151666236, all: [ 0.62255029  0.53698612  0.65819153]


### Model Accuracy: 56.7%

The model's results are better than random guessing, but fall quite short of inspiring confidence.

While this model is doing better than a coin flip it edges out the best baseline - assuming all jobs listings pay over 90K - by less than 3%.

## Thoughts and Conclusions

At the project outset, features the team thought would help determine whether a position had a salary over 90K turned out to be poor predictors when put into a model.  Significant accuracy improvements would likely be made by gathering more data points from the job descriptions, for example if 'SQL' is in the description, or if the title contains the term 'Senior' or 'Sr' or 'intern' etc.

As an overarching caveat, the data used was incomplete and inaccurate at best.  The web scraping effort put forth for data collection did not lend itself to a clean dataset provided us with less than 1000 job listings to use as data.

It's also worth mentioning that any job that did not provide a salary range on its listing was not used in this prediction which biases the dataset. For example, it's possible that listings that disclose a salary range are inherently higher paying.

#### Appendix A: Importing and Merging data


```python
#   This code imports each csv and creates a dataframe with the job ID as the index
#   which is then user later to merge the datapoints across jobs
python = pd.read_csv("./id_and_Python.csv",index_col='id')
del python['Unnamed: 0']
python = python.dropna()
python = python.reset_index().drop_duplicates(subset='id', keep='last').set_index('id')
years = pd.read_csv("./id_years - removed_dupes.csv", index_col='id')
del years['Unnamed: 0']
years = years.reset_index().drop_duplicates(subset='id', keep='last').set_index('id')
salary = pd.read_csv("./id_salary.csv", index_col='id')
del salary['Unnamed: 0']
phd = pd.read_csv("./phd_df.csv",index_col='job_id',encoding='ISO-8859-1')
del phd['Unnamed: 0']
phd = phd.reset_index().drop_duplicates(subset='job_id', keep='last').set_index('job_id')
startup = pd.read_csv("./startup_df.csv",index_col='id')
del startup['Unnamed: 0']
startup = startup.reset_index().drop_duplicates(subset='id', keep='last').set_index('id')
```


```python
#   This code merges each dataframe
df = pd.merge(python,years,how='outer', left_index=True, right_index=True, indicator='py_year')
df = pd.merge(df,scientist,how='outer', left_index=True, right_index=True, indicator='scientist_merge')
df.rename(columns={'Classifier':'scientist_in_title'},inplace=True)
df = pd.merge(df,salary,how='left',left_index=True,right_index=True,indicator='salary_merge')
df = pd.merge(df,startup,how='left',left_index=True,right_index=True,indicator='startup_merge')
df = pd.merge(df,phd,how='left',left_index=True,right_index=True,indicator='phd_merge')
```


```python
#   Fills any 'NaN' values with 0s
df['Python'] = df['Python'].fillna(0)
df['years'] = df['years'].fillna(0)
df['has_phd'] = df['has_phd'].fillna(0)
df['has_startup'] = df['has_startup'].fillna(0)
df['scientist_in_title'] = df['scientist_in_title'].fillna(0)
```


```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Python</th>
      <th>years</th>
      <th>py_year</th>
      <th>Title</th>
      <th>scientist_in_title</th>
      <th>scientist_merge</th>
      <th>over_90k</th>
      <th>salary_merge</th>
      <th>has_startup</th>
      <th>startup_merge</th>
      <th>has_phd</th>
      <th>title</th>
      <th>phd_merge</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>jl_f43cd8061406b3d7</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>left_only</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>left_only</td>
      <td>1.0</td>
      <td>both</td>
      <td>0.0</td>
      <td>left_only</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>left_only</td>
    </tr>
    <tr>
      <th>jl_b59a64d3298e7fe8</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>left_only</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>left_only</td>
      <td>1.0</td>
      <td>both</td>
      <td>0.0</td>
      <td>left_only</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>left_only</td>
    </tr>
    <tr>
      <th>jl_fbadb59b5f73dc18</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>both</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>left_only</td>
      <td>1.0</td>
      <td>both</td>
      <td>0.0</td>
      <td>left_only</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>left_only</td>
    </tr>
    <tr>
      <th>jl_90603c7f1f0af480</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>both</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>left_only</td>
      <td>1.0</td>
      <td>both</td>
      <td>0.0</td>
      <td>left_only</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>left_only</td>
    </tr>
    <tr>
      <th>jl_70351b5092814475</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>left_only</td>
      <td>biomedical data scientist</td>
      <td>1.0</td>
      <td>both</td>
      <td>1.0</td>
      <td>both</td>
      <td>0.0</td>
      <td>left_only</td>
      <td>1.0</td>
      <td>Biomedical Data Scientist</td>
      <td>both</td>
    </tr>
  </tbody>
</table>
</div>




```python
def examine_coefficients(model, df):
    df = pd.DataFrame(
        { 'Coefficient' : model.coef_[0] , 'Feature' : df.columns}
    ).sort_values(by='Coefficient')
    return df[df.Coefficient !=0 ]

examine_coefficients(model, X)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coefficient</th>
      <th>Feature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>-0.458136</td>
      <td>scientist_in_title</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.154580</td>
      <td>years</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.708252</td>
      <td>has_phd</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.580908</td>
      <td>has_startup</td>
    </tr>
  </tbody>
</table>
</div>
