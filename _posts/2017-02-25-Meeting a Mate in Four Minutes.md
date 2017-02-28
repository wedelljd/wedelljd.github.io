---
layout: post
title: Meeting a Mate in Four Minutes
---

# Meeting a Mate in Four Minutes

This project aims to draw conclusions from speed dating data collected at Columbia University and build a model to predict whether two participants will match.

The dataset, sourced from Kaggle, comes as a single table.  I designed and created an SQLite database to segment the data into a more useful and malleable structure.  Using this database I explored the difference in correlation with participants’ decisions to pursue a second date, whether placing limitations on how many match decisions were allowed affected males and females differently, how a participants number of matches vary with self-rated attractiveness and partner rated attractiveness and how 'yes' decisions trend with the sequential order of a date throughout the event.

## Evaluating a Logistic Regression model against the baseline

After creating the database I implemented a logistic regression model to predict a participants' decision.  To test my model I first built a baseline model which randomly guessed whether a participant's partner would give a 'yes' decision.  Random guessing produces an accuracy 50%.


```python
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from random import randint
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib
from IPython.display import display
matplotlib.style.use('ggplot')

speed_data = pd.read_csv('./Speed Dating Data2.csv', encoding = 'iso-8859-1')
features = ['attr_o']
X = speed_data[features]
y = speed_data['dec_o']

random_match_predictions = []
for i in range(len(X)):
    random_match_predictions.append(randint(0,1))

print("Random Matches Accuracy: " + str(accuracy_score(y,random_match_predictions)))
```

    Random Matches Accuracy: 0.497612795417


It turns out that the decision made by a participant's partner could be predicted well above the baseline with just a single feature - how attractive the partner rated the participant.  Adding other features to this model only marginally increased performance, and in some cases detracted from the model's accuracy.  This simple logistic regression gives us roughly 72% accuracy.

This discovery led me to focus on many other aspects of the attractiveness attribute within the dataset which will be discussed later on.


```python
lr_data = speed_data.dropna(axis=0,subset=['attr_o','intel_o','shar_o','int_corr','intel_o',])
#features = ['attr_o','shar_o','int_corr','intel_o']
features = ['attr_o']
model = LogisticRegressionCV(Cs=5)

X_train, X_test, y_train, y_test = train_test_split(lr_data[features], lr_data['dec_o'],test_size = 0.3)

model.fit(X_train,y_train)
model.score(X_test,y_test)

```




    0.7289325842696629



## Does the order of when you meet your partner of a have an impact on their decision?

Before diving into the results found looking into the attributes of the participants, I wanted to establish whether simply the order of a date during the event would influence the participants' decisions.

Related research on sequential decision-making from the University of Negev in Isreal shows that when judges are presiding in court they are far more lenient at the beginning of the day, and likewise after taking a break.

"They found that at the beginning of a court session about 65 per cent of the rulings tended to be in favour of the prisoner, but the chance of a favourable ruling declined to near zero by the end of the session" - theglobeandmail.com

The graph below shows the percentage of 'yes' decisions by date order.  It appears speed dating behavior is consistent with the results found in the study of judge leniency.  Participants tend to match more towards the beginning of the event with spikes as the date order is preceeded by a break, signalling a renewed willingness to match.  It's also notable that during the event there is an overall downtrend as the night goes on.


```python
aggregated_order = pd.pivot_table(speed_data,values='dec_o',columns='order',aggfunc='mean')
aggregated_order.to_csv('order_of_date_data.csv')
sns.set_style('dark',{'axes.grid' : False})
aggregated_order.plot()
plt.xticks(range(len(aggregated_order)))
plt.title('Mean of Yes Decisions vs Order of Date')
plt.ylabel('Mean of Yes Decisions')
plt.xlabel('Date Sequence Number')
plt.show()
```

[!png](/Meeting%20a%20Mate%20in%20Four%20Minutes_files/Meeting%20a%20Mate%20in%20Four%20Minutes_9_0.png)

![png](Meeting%20a%20Mate%20in%20Four%20Minutes_files/Meeting%20a%20Mate%20in%20Four%20Minutes_12_0.png)

## Observed differences in attribute importance under limitations
*See Appendix A for dataframe construction and manipulation

During the speed dating study there were two scenarios a speed-date event could fall under.  In scenario 1 participants were able to give a 'yes' decision to as many dates as they wanted.  In scenario 2 participants were limited to only giving a 'yes' response to half of their dates.

I wanted to determine how a participant's attribute ratings would vary with their decision on a partner when there were constraints placed on their 'yes' responses.

I looked at how well the ratings of attractiveness, intelligence and fun correlated with a participant's decision for both scenarios and also segmented the results by gender.  The result was rather striking.  

Female decisions correlated less with each attribute as their 'yes' decisions were limited.  This might suggest that as a woman's options expand they become more selective for attractiveness, fun and intelligence, however, as their options become limited they place greater consideration on other factors.


```python
sns.set_style('dark',{'axes.grid' : True})
extensive_female_dec_corrmat.head(1).iloc[:,1:].plot(kind='bar',rot=0)
extensive_female_dec_corrmat.head(1).iloc[:,1:].to_csv('extended female decision corrmat.csv')

plt.ylabel('Correlation')
plt.title('Female Decision Correlation with Extensive Choice')
plt.ylim([0,0.6])
plt.show()
display(extensive_female_dec_corrmat.head(1).iloc[:,1:])
limited_female_dec_corrmat.head(1).iloc[:,1:].plot(kind='bar',rot=0)
limited_female_dec_corrmat.head(1).iloc[:,1:].to_csv('limited female decision corrmat.csv')
plt.ylabel('Correlation')
plt.title('Female Decision Correlation with Limited Choice')
plt.ylim([0,0.6])
plt.show()
display(limited_female_dec_corrmat.head(1).iloc[:,1:])
```


![png](Meeting%20a%20Mate%20in%20Four%20Minutes_files/Meeting%20a%20Mate%20in%20Four%20Minutes_12_0.png)



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>F rating of partner attractiveness</th>
      <th>F rating of partner intelligence</th>
      <th>F rating of partner fun</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Female Decision</th>
      <td>0.461007</td>
      <td>0.250342</td>
      <td>0.43168</td>
    </tr>
  </tbody>
</table>
</div>



![png](Meeting%20a%20Mate%20in%20Four%20Minutes_files/Meeting%20a%20Mate%20in%20Four%20Minutes_12_2.png)



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>F rating of partner attractiveness</th>
      <th>F rating of partner intelligence</th>
      <th>F rating of partner fun</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Female Decision</th>
      <td>0.412462</td>
      <td>0.218562</td>
      <td>0.406078</td>
    </tr>
  </tbody>
</table>
</div>


## Men's Inverse Reaction to Limitation

The unexpected finding here was that men exhibited the exact opposite behavior when limiting their number of 'yes' decisions.  As their options became limited, they were more responsive to partners they percieved to be more attractive, intelligent and fun.  The greatest disparity in attribute correlation between limited and extensive choice is the jump in how men's rating of how fun they perceive their partner affected their 'yes' decisions.


```python
sns.set_style('dark',{'axes.grid' : True})
extensive_male_dec_corrmat.head(1).iloc[:,1:].plot(kind='bar',rot=0)
extensive_male_dec_corrmat.head(1).iloc[:,1:].to_csv('extended male decision corrmat.csv')
plt.ylabel('Correlation')
plt.title('Male Decision Correlation with Extensive Choice')
plt.ylim([0,0.6])
plt.show()
display(extensive_male_dec_corrmat.head(1).iloc[:,1:])
limited_male_dec_corrmat.head(1).iloc[:,1:].plot(kind='bar',rot=0)
limited_male_dec_corrmat.head(1).iloc[:,1:].to_csv('limied male decision corrmat.csv')
plt.ylabel('Correlation')
plt.title('Male Decision Correlation with Limited Choice')
plt.ylim([0,0.6])
plt.show()
display(limited_male_dec_corrmat.head(1).iloc[:,1:])
```


![png](Meeting%20a%20Mate%20in%20Four%20Minutes_files/Meeting%20a%20Mate%20in%20Four%20Minutes_15_0.png)



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>M rating of partner attractiveness</th>
      <th>M rating of partner intelligence</th>
      <th>M rating of partner fun</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Male Decision</th>
      <td>0.49525</td>
      <td>0.207197</td>
      <td>0.383823</td>
    </tr>
  </tbody>
</table>
</div>



![png](Meeting%20a%20Mate%20in%20Four%20Minutes_files/Meeting%20a%20Mate%20in%20Four%20Minutes_15_2.png)



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>M rating of partner attractiveness</th>
      <th>M rating of partner intelligence</th>
      <th>M rating of partner fun</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Male Decision</th>
      <td>0.528877</td>
      <td>0.230865</td>
      <td>0.448878</td>
    </tr>
  </tbody>
</table>
</div>


## Observed differences in match results based on self-rated attractiveness

Since attractiveness seemed to be a rather accuate predictor of whether participants give a 'yes' decision, I wanted to explore the relationship between people's rating of their own attractiveness and the matches they receive.

I grouped participants using KMeans clustering based on how highly they rated their own attractiveness the night of the event.  I performed the classification using both 5 and 8 clusters, with similar findings.  

The results for 5 clusters gave three groups who rate their attractiveness between a 5 and 7 with two outlier groups, those who rated themselves extremely high - averaging over 9 - and extremely low - averaging 3.5.

On the aggregate those who rate themselves extremely high do in fact receive more matches than those who rate themselves extremely low in attractiveness, but not by much.  This begged a follow-up question, though. Do results for a high self-rated attractive male differ from that of a female?  This breakdown revealed a dramatic separation.

The results were much different when looking at these groups broekn out by gender.  Men who rated themselves highly attractive did tend to do significantly better than their low self-rating couterparts.  The disparity is nearly 2 matches on average.  Even the next highest group receives almost a full match less.

The same breakdown for females, however, tells a much different story.  The difference in average number of matches between those who rated themselves highest versus those who rated themselves lowest was only 0.2.  Both groups receive roughly the same number of matches.  Even more profound is that the group rating themselves the highest don't even receive the most matches, in fact all the groups rating themselves more conservatively actually received more matches than those who rated themselves the highest!  Very strange.


```python

self_rated_five_class_pivot
#self_rated_five_class_pivot.to_csv('self_rated_five_class_pivot.csv')
#Gender 0 represents females, 1 represents males
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>gender</th>
      <th colspan="5" halign="left">0</th>
      <th colspan="5" halign="left">1</th>
    </tr>
    <tr>
      <th>self_rated_class</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>number of matches</th>
      <td>2.972117</td>
      <td>2.892996</td>
      <td>2.501487</td>
      <td>2.223684</td>
      <td>2.410468</td>
      <td>2.985677</td>
      <td>2.026582</td>
      <td>2.793905</td>
      <td>2.018100</td>
      <td>3.803625</td>
    </tr>
    <tr>
      <th>self-rated attractiveness at sign-up</th>
      <td>5.709759</td>
      <td>8.000000</td>
      <td>7.000000</td>
      <td>3.401316</td>
      <td>9.179063</td>
      <td>5.623698</td>
      <td>8.000000</td>
      <td>7.000000</td>
      <td>3.574661</td>
      <td>9.413897</td>
    </tr>
    <tr>
      <th>avg_attractiveness_rating</th>
      <td>6.188312</td>
      <td>6.481016</td>
      <td>6.517800</td>
      <td>5.652548</td>
      <td>6.805537</td>
      <td>5.662459</td>
      <td>6.028166</td>
      <td>5.834683</td>
      <td>4.972652</td>
      <td>6.793591</td>
    </tr>
    <tr>
      <th>self-rated intelligence at sign-up</th>
      <td>7.790875</td>
      <td>8.353113</td>
      <td>8.155600</td>
      <td>8.000000</td>
      <td>8.834711</td>
      <td>8.227865</td>
      <td>8.802532</td>
      <td>8.263031</td>
      <td>7.782805</td>
      <td>9.501511</td>
    </tr>
    <tr>
      <th>avg_intelligence_rating</th>
      <td>7.284892</td>
      <td>7.289691</td>
      <td>7.283637</td>
      <td>7.349559</td>
      <td>7.197458</td>
      <td>7.547812</td>
      <td>7.270740</td>
      <td>7.273910</td>
      <td>7.358702</td>
      <td>7.579472</td>
    </tr>
    <tr>
      <th>avg_fun_rating</th>
      <td>6.509194</td>
      <td>6.509474</td>
      <td>6.513414</td>
      <td>6.251391</td>
      <td>6.679665</td>
      <td>6.205716</td>
      <td>6.152609</td>
      <td>6.201890</td>
      <td>5.814118</td>
      <td>6.703393</td>
    </tr>
  </tbody>
</table>
</div>



## Observed differences in match results based on partner-rated attractiveness

The above results were rather striking, so I wanted to see if they would be repeated when grouping participants based on how their partners rated their attractiveness.  This time the clustering was performed based on the average attractiveness rating a participant received from their partners, not themselves.  The outcome supported the results of the earlier logistic regression.

Using the parner's rating as the clustering feature participants, for both genders, who were rated as more attractive literally outmatch their peers, and in order nonetheless.  The disparity between number of matches, however, is far greater between highest vs lowest rated males than it is for the same groups in females


```python
partner_rated_five_class_pivot
#partner_rated_five_class_pivot.to_csv('partner_rated_five_class_pivot.csv')
#Gender 0 represents females, 1 represents males
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>gender</th>
      <th colspan="5" halign="left">0</th>
      <th colspan="5" halign="left">1</th>
    </tr>
    <tr>
      <th>partner_rated_class</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>number of matches</th>
      <td>3.501220</td>
      <td>1.727592</td>
      <td>1.365957</td>
      <td>3.824666</td>
      <td>2.209770</td>
      <td>3.434985</td>
      <td>2.190972</td>
      <td>0.915563</td>
      <td>4.877888</td>
      <td>3.127660</td>
    </tr>
    <tr>
      <th>self-rated attractiveness at sign-up</th>
      <td>7.190244</td>
      <td>6.966608</td>
      <td>6.200000</td>
      <td>7.515602</td>
      <td>6.959770</td>
      <td>7.654799</td>
      <td>6.560185</td>
      <td>6.423841</td>
      <td>7.584158</td>
      <td>6.896809</td>
    </tr>
    <tr>
      <th>avg_attractiveness_rating</th>
      <td>7.035599</td>
      <td>5.227267</td>
      <td>4.140736</td>
      <td>7.886355</td>
      <td>6.156153</td>
      <td>7.027031</td>
      <td>5.230206</td>
      <td>4.117198</td>
      <td>7.903055</td>
      <td>6.164608</td>
    </tr>
    <tr>
      <th>self-rated intelligence at sign-up</th>
      <td>7.982927</td>
      <td>8.557118</td>
      <td>8.480851</td>
      <td>8.014859</td>
      <td>8.222222</td>
      <td>8.441176</td>
      <td>8.435185</td>
      <td>8.430464</td>
      <td>8.551155</td>
      <td>8.529787</td>
    </tr>
    <tr>
      <th>avg_intelligence_rating</th>
      <td>7.348845</td>
      <td>7.109719</td>
      <td>6.736120</td>
      <td>7.616685</td>
      <td>7.222301</td>
      <td>7.484498</td>
      <td>7.311059</td>
      <td>7.051675</td>
      <td>7.867794</td>
      <td>7.395059</td>
    </tr>
    <tr>
      <th>avg_fun_rating</th>
      <td>6.955656</td>
      <td>5.923385</td>
      <td>5.505711</td>
      <td>7.237465</td>
      <td>6.260304</td>
      <td>6.947432</td>
      <td>5.750463</td>
      <td>4.924612</td>
      <td>7.605218</td>
      <td>6.519961</td>
    </tr>
  </tbody>
</table>
</div>



## Building a Better Model

After exploring the ins and outs of how attractiveness impacts out participants the next goal was to build a model that takes all attribute ratings into account to predict the outcome of the speed date.

For this a Random Forest classifier was used, and to find the optimal parameters the model was put through a gridsearch.


```python
rfc = RandomForestClassifier()
features = ['condtn', 'Male Age', 'round','Female Age','F rating of partner attractiveness','M rating of partner attractiveness', 'F rating of partner fun','M rating of partner fun', 'F rating of partner intelligence','M rating of partner intelligence', 'F rating of partner sincerity','M rating of partner sincerity', 'F rating of partner shared interests','M rating of partner shared interests', 'F rating of partner ambition','M rating of partner ambition', 'order','age_difference']
X = dates[features]
y = dates['match']
param_grid = {
    'n_estimators': [50, 100,200,500],#, 200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10,20,100,None]
}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X, y)
print(CV_rfc.best_params_)
print(CV_rfc.best_score_)
```

    {'max_features': 'auto', 'max_depth': 100, 'n_estimators': 200}
    0.8484472049689441


The Random Forest model taking into account all attributes and a few other features about the date scored nearly 84.8%.  A gain of nearly 15% over the Logistic Regression which only took into account how attractive someone's partner rated them.


```python
feature_importance_dictionary = {}
for index, i in enumerate(features):
    feature_importance_dictionary[i] = CV_rfc.best_estimator_.feature_importances_[index]

feature_importance_dictionary
x =sorted(feature_importance_dictionary.items(), key=lambda x: x[1],reverse=True)
#print(x)

feature_importance_list = []
feature_importance_labels = []
for i in x:
    feature_importance_list.append(i[1])
    feature_importance_labels.append(i[0])

plt.bar(range(len(feature_importance_list)), feature_importance_list, align='center')
plt.xticks(range(len(feature_importance_labels)), feature_importance_labels,rotation=90)
plt.ylabel('Feature Importance')
plt.tight_layout()
plt.show()
```


![png](Meeting%20a%20Mate%20in%20Four%20Minutes_files/Meeting%20a%20Mate%20in%20Four%20Minutes_26_0.png)



```python
rfc = RandomForestClassifier()
features = ['condtn', 'Male Age', 'round','Female Age','F rating of partner attractiveness','M rating of partner attractiveness', 'F rating of partner fun','M rating of partner fun', 'F rating of partner intelligence','M rating of partner intelligence', 'F rating of partner sincerity','M rating of partner sincerity', 'F rating of partner shared interests','M rating of partner shared interests', 'F rating of partner ambition','M rating of partner ambition', 'order','age_difference']
X = dates[features]
y = dates['Male Decision']
param_grid = {
    'n_estimators': [50, 100,200,500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10,20,100,None]
}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X, y)
print(CV_rfc.best_params_)
print(CV_rfc.best_score_)

feature_importance_dictionary = {}
for index, i in enumerate(features):
    feature_importance_dictionary[i] = CV_rfc.best_estimator_.feature_importances_[index]

feature_importance_dictionary
x =sorted(feature_importance_dictionary.items(), key=lambda x: x[1],reverse=True)
#print(x)

feature_importance_list = []
feature_importance_labels = []
for i in x:
    feature_importance_list.append(i[1])
    feature_importance_labels.append(i[0])

plt.bar(range(len(feature_importance_list)), feature_importance_list, align='center')
plt.xticks(range(len(feature_importance_labels)), feature_importance_labels,rotation=90)
plt.ylabel('Feature Importance')
plt.tight_layout()
plt.show()
```

    {'max_features': 'sqrt', 'max_depth': 10, 'n_estimators': 200}
    0.7370600414078675



![png](Meeting%20a%20Mate%20in%20Four%20Minutes_files/Meeting%20a%20Mate%20in%20Four%20Minutes_27_1.png)



```python
rfc = RandomForestClassifier()
features = ['condtn', 'Male Age', 'round','Female Age','F rating of partner attractiveness','M rating of partner attractiveness', 'F rating of partner fun','M rating of partner fun', 'F rating of partner intelligence','M rating of partner intelligence', 'F rating of partner sincerity','M rating of partner sincerity', 'F rating of partner shared interests','M rating of partner shared interests', 'F rating of partner ambition','M rating of partner ambition', 'order','age_difference']
X = dates[features]
y = dates['Female Decision']
param_grid = {
    'n_estimators': [50, 100,200,500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10,20,100,None]
}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X, y)
print(CV_rfc.best_params_)
print(CV_rfc.best_score_)

feature_importance_dictionary = {}
for index, i in enumerate(features):
    feature_importance_dictionary[i] = CV_rfc.best_estimator_.feature_importances_[index]

feature_importance_dictionary
x =sorted(feature_importance_dictionary.items(), key=lambda x: x[1],reverse=True)
#print(x)

feature_importance_list = []
feature_importance_labels = []
for i in x:
    feature_importance_list.append(i[1])
    feature_importance_labels.append(i[0])

plt.bar(range(len(feature_importance_list)), feature_importance_list, align='center')
plt.xticks(range(len(feature_importance_labels)), feature_importance_labels,rotation=90)
plt.ylabel('Feature Importance')
plt.tight_layout()
plt.show()
```

    {'max_features': 'log2', 'max_depth': 10, 'n_estimators': 100}
    0.7325051759834369



![png](Meeting%20a%20Mate%20in%20Four%20Minutes_files/Meeting%20a%20Mate%20in%20Four%20Minutes_28_1.png)


## Appendix A


```python
conn = sqlite3.connect('SpeedDatingDB.sqlite')

dates = pd.read_sql_query\
("""
SELECT m.date_id
      ,m.iid as 'Male IID'
      ,m.pid as 'Female IID'
      ,m.wave
      ,m.condtn
      ,m.round
      ,m.age as 'Male Age'
      ,f.age as 'Female Age'

      ,m.dec_o as 'Female Decision'
      ,f.dec_o as 'Male Decision'

      ,m.attr_o as 'F rating of partner attractiveness'
      ,f.attr_o as 'M rating of partner attractiveness'

      ,m.fun_o as 'F rating of partner fun'
      ,f.fun_o as 'M rating of partner fun'

      ,m.intel_o as 'F rating of partner intelligence'
      ,f.intel_o as 'M rating of partner intelligence'

      ,m.sinc_o as 'F rating of partner sincerity'
      ,f.sinc_o as 'M rating of partner sincerity'

      ,m.shar_o as 'F rating of partner shared interests'
      ,f.shar_o as 'M rating of partner shared interests'

      ,m.amb_o as 'F rating of partner ambition'
      ,f.amb_o as 'M rating of partner ambition'

      ,m.shar_o as 'F rating of shared interests'
      ,f.shar_o as 'M rating of shared interests'

      ,m.match
      ,m.[order]

      from male_side_of_dates m
      join female_side_of_dates f on f.date_id = m.date_id


""", conn)
dates['age_difference'] = dates['Male Age'] - dates['Female Age']
dates = dates.dropna()
```


```python
limited_dates = dates[dates['condtn']==1]
extensive_dates = dates[dates['condtn']==2]
limited_female_dec_corrmat = limited_dates[['Female Decision','F rating of partner attractiveness', 'F rating of partner intelligence', 'F rating of partner fun']].corr()
extensive_female_dec_corrmat = extensive_dates[['Female Decision','F rating of partner attractiveness', 'F rating of partner intelligence', 'F rating of partner fun']].corr()
limited_male_dec_corrmat = limited_dates[['Male Decision','M rating of partner attractiveness','M rating of partner intelligence', 'M rating of partner fun']].corr()
extensive_male_dec_corrmat = extensive_dates[['Male Decision','M rating of partner attractiveness','M rating of partner intelligence', 'M rating of partner fun']].corr()
```

## Appendix B


```python
query = """SELECT p.iid
        , go.go_out_description [goes_out_desc]
        , fd.go_out_description [freq_date_desc]
        , r.avg_attractiveness_rating
        , r.avg_fun_rating
        , r.avg_intelligence_rating
        , r.avg_ambition_rating
        , p.[number of matches]
        , sra.[self-rated attractiveness at sign-up]
        , sra.[self-rated intelligence at sign-up]
        , p.gender

    from participants p
    left join ratings r on p.iid = r.iid
    left join Go_Out_and_Frequency_of_Dates_Description go on go.go_out_id = p.go_out
    left join Go_Out_and_Frequency_of_Dates_Description fd on fd.go_out_id = p.frequency_of_dates
    left join self_rated_attributes sra on sra.iid = p.iid
    order by p.iid"""

partic_and_ratings = pd.read_sql_query(query,conn)
```


```python
features = ['avg_attractiveness_rating']
partner_rated_five_way_classifier = KMeans(5)
partic_and_ratings = partic_and_ratings.dropna()
X = partic_and_ratings[features]
X2 = X # pd.get_dummies(X,columns=['goes_out_desc','freq_date_desc']).dropna()
partic_and_ratings['partner_rated_class'] = partner_rated_five_way_classifier.fit_predict(X)


features = ['self-rated attractiveness at sign-up']
five_way_classifier = KMeans(5)
partic_and_ratings = partic_and_ratings.dropna()
X = partic_and_ratings[features]
partic_and_ratings['self_rated_class'] = five_way_classifier.fit_predict(X)
partic_and_ratings
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>iid</th>
      <th>goes_out_desc</th>
      <th>freq_date_desc</th>
      <th>avg_attractiveness_rating</th>
      <th>avg_fun_rating</th>
      <th>avg_intelligence_rating</th>
      <th>avg_ambition_rating</th>
      <th>number of matches</th>
      <th>self-rated attractiveness at sign-up</th>
      <th>self-rated intelligence at sign-up</th>
      <th>gender</th>
      <th>partner_rated_class</th>
      <th>self_rated_class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>several times a week</td>
      <td>almost never</td>
      <td>6.700000</td>
      <td>7.200000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>4</td>
      <td>6.0</td>
      <td>8.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>several times a week</td>
      <td>almost never</td>
      <td>6.700000</td>
      <td>7.200000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>4</td>
      <td>6.0</td>
      <td>8.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>several times a week</td>
      <td>almost never</td>
      <td>6.700000</td>
      <td>7.200000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>4</td>
      <td>6.0</td>
      <td>8.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>several times a week</td>
      <td>almost never</td>
      <td>6.700000</td>
      <td>7.200000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>4</td>
      <td>6.0</td>
      <td>8.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>several times a week</td>
      <td>almost never</td>
      <td>6.700000</td>
      <td>7.200000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>4</td>
      <td>6.0</td>
      <td>8.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>several times a week</td>
      <td>almost never</td>
      <td>6.700000</td>
      <td>7.200000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>4</td>
      <td>6.0</td>
      <td>8.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>several times a week</td>
      <td>almost never</td>
      <td>6.700000</td>
      <td>7.200000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>4</td>
      <td>6.0</td>
      <td>8.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>several times a week</td>
      <td>almost never</td>
      <td>6.700000</td>
      <td>7.200000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>4</td>
      <td>6.0</td>
      <td>8.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>several times a week</td>
      <td>almost never</td>
      <td>6.700000</td>
      <td>7.200000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>4</td>
      <td>6.0</td>
      <td>8.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>several times a week</td>
      <td>almost never</td>
      <td>6.700000</td>
      <td>7.200000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>4</td>
      <td>6.0</td>
      <td>8.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2</td>
      <td>several times a week</td>
      <td>once a month</td>
      <td>7.700000</td>
      <td>7.500000</td>
      <td>7.900000</td>
      <td>7.500000</td>
      <td>2</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2</td>
      <td>several times a week</td>
      <td>once a month</td>
      <td>7.700000</td>
      <td>7.500000</td>
      <td>7.900000</td>
      <td>7.500000</td>
      <td>2</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2</td>
      <td>several times a week</td>
      <td>once a month</td>
      <td>7.700000</td>
      <td>7.500000</td>
      <td>7.900000</td>
      <td>7.500000</td>
      <td>2</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2</td>
      <td>several times a week</td>
      <td>once a month</td>
      <td>7.700000</td>
      <td>7.500000</td>
      <td>7.900000</td>
      <td>7.500000</td>
      <td>2</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2</td>
      <td>several times a week</td>
      <td>once a month</td>
      <td>7.700000</td>
      <td>7.500000</td>
      <td>7.900000</td>
      <td>7.500000</td>
      <td>2</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2</td>
      <td>several times a week</td>
      <td>once a month</td>
      <td>7.700000</td>
      <td>7.500000</td>
      <td>7.900000</td>
      <td>7.500000</td>
      <td>2</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2</td>
      <td>several times a week</td>
      <td>once a month</td>
      <td>7.700000</td>
      <td>7.500000</td>
      <td>7.900000</td>
      <td>7.500000</td>
      <td>2</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2</td>
      <td>several times a week</td>
      <td>once a month</td>
      <td>7.700000</td>
      <td>7.500000</td>
      <td>7.900000</td>
      <td>7.500000</td>
      <td>2</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2</td>
      <td>several times a week</td>
      <td>once a month</td>
      <td>7.700000</td>
      <td>7.500000</td>
      <td>7.900000</td>
      <td>7.500000</td>
      <td>2</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2</td>
      <td>several times a week</td>
      <td>once a month</td>
      <td>7.700000</td>
      <td>7.500000</td>
      <td>7.900000</td>
      <td>7.500000</td>
      <td>2</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>20</th>
      <td>3</td>
      <td>several times a week</td>
      <td>once a week</td>
      <td>6.500000</td>
      <td>6.200000</td>
      <td>7.300000</td>
      <td>7.111111</td>
      <td>0</td>
      <td>8.0</td>
      <td>9.0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>3</td>
      <td>several times a week</td>
      <td>once a week</td>
      <td>6.500000</td>
      <td>6.200000</td>
      <td>7.300000</td>
      <td>7.111111</td>
      <td>0</td>
      <td>8.0</td>
      <td>9.0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>3</td>
      <td>several times a week</td>
      <td>once a week</td>
      <td>6.500000</td>
      <td>6.200000</td>
      <td>7.300000</td>
      <td>7.111111</td>
      <td>0</td>
      <td>8.0</td>
      <td>9.0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>3</td>
      <td>several times a week</td>
      <td>once a week</td>
      <td>6.500000</td>
      <td>6.200000</td>
      <td>7.300000</td>
      <td>7.111111</td>
      <td>0</td>
      <td>8.0</td>
      <td>9.0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>3</td>
      <td>several times a week</td>
      <td>once a week</td>
      <td>6.500000</td>
      <td>6.200000</td>
      <td>7.300000</td>
      <td>7.111111</td>
      <td>0</td>
      <td>8.0</td>
      <td>9.0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>3</td>
      <td>several times a week</td>
      <td>once a week</td>
      <td>6.500000</td>
      <td>6.200000</td>
      <td>7.300000</td>
      <td>7.111111</td>
      <td>0</td>
      <td>8.0</td>
      <td>9.0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>3</td>
      <td>several times a week</td>
      <td>once a week</td>
      <td>6.500000</td>
      <td>6.200000</td>
      <td>7.300000</td>
      <td>7.111111</td>
      <td>0</td>
      <td>8.0</td>
      <td>9.0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>3</td>
      <td>several times a week</td>
      <td>once a week</td>
      <td>6.500000</td>
      <td>6.200000</td>
      <td>7.300000</td>
      <td>7.111111</td>
      <td>0</td>
      <td>8.0</td>
      <td>9.0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>3</td>
      <td>several times a week</td>
      <td>once a week</td>
      <td>6.500000</td>
      <td>6.200000</td>
      <td>7.300000</td>
      <td>7.111111</td>
      <td>0</td>
      <td>8.0</td>
      <td>9.0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>3</td>
      <td>several times a week</td>
      <td>once a week</td>
      <td>6.500000</td>
      <td>6.200000</td>
      <td>7.300000</td>
      <td>7.111111</td>
      <td>0</td>
      <td>8.0</td>
      <td>9.0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6786</th>
      <td>551</td>
      <td>once a week</td>
      <td>several times a year</td>
      <td>6.142857</td>
      <td>5.571429</td>
      <td>6.761905</td>
      <td>6.238095</td>
      <td>2</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6787</th>
      <td>551</td>
      <td>once a week</td>
      <td>several times a year</td>
      <td>6.142857</td>
      <td>5.571429</td>
      <td>6.761905</td>
      <td>6.238095</td>
      <td>2</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6788</th>
      <td>551</td>
      <td>once a week</td>
      <td>several times a year</td>
      <td>6.142857</td>
      <td>5.571429</td>
      <td>6.761905</td>
      <td>6.238095</td>
      <td>2</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6789</th>
      <td>551</td>
      <td>once a week</td>
      <td>several times a year</td>
      <td>6.142857</td>
      <td>5.571429</td>
      <td>6.761905</td>
      <td>6.238095</td>
      <td>2</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6790</th>
      <td>551</td>
      <td>once a week</td>
      <td>several times a year</td>
      <td>6.142857</td>
      <td>5.571429</td>
      <td>6.761905</td>
      <td>6.238095</td>
      <td>2</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6791</th>
      <td>551</td>
      <td>once a week</td>
      <td>several times a year</td>
      <td>6.142857</td>
      <td>5.571429</td>
      <td>6.761905</td>
      <td>6.238095</td>
      <td>2</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6792</th>
      <td>551</td>
      <td>once a week</td>
      <td>several times a year</td>
      <td>6.142857</td>
      <td>5.571429</td>
      <td>6.761905</td>
      <td>6.238095</td>
      <td>2</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6793</th>
      <td>551</td>
      <td>once a week</td>
      <td>several times a year</td>
      <td>6.142857</td>
      <td>5.571429</td>
      <td>6.761905</td>
      <td>6.238095</td>
      <td>2</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6794</th>
      <td>552</td>
      <td>several times a week</td>
      <td>twice a week</td>
      <td>7.300000</td>
      <td>5.750000</td>
      <td>6.157895</td>
      <td>6.150000</td>
      <td>6</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6795</th>
      <td>552</td>
      <td>several times a week</td>
      <td>twice a week</td>
      <td>7.300000</td>
      <td>5.750000</td>
      <td>6.157895</td>
      <td>6.150000</td>
      <td>6</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6796</th>
      <td>552</td>
      <td>several times a week</td>
      <td>twice a week</td>
      <td>7.300000</td>
      <td>5.750000</td>
      <td>6.157895</td>
      <td>6.150000</td>
      <td>6</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6797</th>
      <td>552</td>
      <td>several times a week</td>
      <td>twice a week</td>
      <td>7.300000</td>
      <td>5.750000</td>
      <td>6.157895</td>
      <td>6.150000</td>
      <td>6</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6798</th>
      <td>552</td>
      <td>several times a week</td>
      <td>twice a week</td>
      <td>7.300000</td>
      <td>5.750000</td>
      <td>6.157895</td>
      <td>6.150000</td>
      <td>6</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6799</th>
      <td>552</td>
      <td>several times a week</td>
      <td>twice a week</td>
      <td>7.300000</td>
      <td>5.750000</td>
      <td>6.157895</td>
      <td>6.150000</td>
      <td>6</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6800</th>
      <td>552</td>
      <td>several times a week</td>
      <td>twice a week</td>
      <td>7.300000</td>
      <td>5.750000</td>
      <td>6.157895</td>
      <td>6.150000</td>
      <td>6</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6801</th>
      <td>552</td>
      <td>several times a week</td>
      <td>twice a week</td>
      <td>7.300000</td>
      <td>5.750000</td>
      <td>6.157895</td>
      <td>6.150000</td>
      <td>6</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6802</th>
      <td>552</td>
      <td>several times a week</td>
      <td>twice a week</td>
      <td>7.300000</td>
      <td>5.750000</td>
      <td>6.157895</td>
      <td>6.150000</td>
      <td>6</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6803</th>
      <td>552</td>
      <td>several times a week</td>
      <td>twice a week</td>
      <td>7.300000</td>
      <td>5.750000</td>
      <td>6.157895</td>
      <td>6.150000</td>
      <td>6</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6804</th>
      <td>552</td>
      <td>several times a week</td>
      <td>twice a week</td>
      <td>7.300000</td>
      <td>5.750000</td>
      <td>6.157895</td>
      <td>6.150000</td>
      <td>6</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6805</th>
      <td>552</td>
      <td>several times a week</td>
      <td>twice a week</td>
      <td>7.300000</td>
      <td>5.750000</td>
      <td>6.157895</td>
      <td>6.150000</td>
      <td>6</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6806</th>
      <td>552</td>
      <td>several times a week</td>
      <td>twice a week</td>
      <td>7.300000</td>
      <td>5.750000</td>
      <td>6.157895</td>
      <td>6.150000</td>
      <td>6</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6807</th>
      <td>552</td>
      <td>several times a week</td>
      <td>twice a week</td>
      <td>7.300000</td>
      <td>5.750000</td>
      <td>6.157895</td>
      <td>6.150000</td>
      <td>6</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6808</th>
      <td>552</td>
      <td>several times a week</td>
      <td>twice a week</td>
      <td>7.300000</td>
      <td>5.750000</td>
      <td>6.157895</td>
      <td>6.150000</td>
      <td>6</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6809</th>
      <td>552</td>
      <td>several times a week</td>
      <td>twice a week</td>
      <td>7.300000</td>
      <td>5.750000</td>
      <td>6.157895</td>
      <td>6.150000</td>
      <td>6</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6810</th>
      <td>552</td>
      <td>several times a week</td>
      <td>twice a week</td>
      <td>7.300000</td>
      <td>5.750000</td>
      <td>6.157895</td>
      <td>6.150000</td>
      <td>6</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6811</th>
      <td>552</td>
      <td>several times a week</td>
      <td>twice a week</td>
      <td>7.300000</td>
      <td>5.750000</td>
      <td>6.157895</td>
      <td>6.150000</td>
      <td>6</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6812</th>
      <td>552</td>
      <td>several times a week</td>
      <td>twice a week</td>
      <td>7.300000</td>
      <td>5.750000</td>
      <td>6.157895</td>
      <td>6.150000</td>
      <td>6</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6813</th>
      <td>552</td>
      <td>several times a week</td>
      <td>twice a week</td>
      <td>7.300000</td>
      <td>5.750000</td>
      <td>6.157895</td>
      <td>6.150000</td>
      <td>6</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6814</th>
      <td>552</td>
      <td>several times a week</td>
      <td>twice a week</td>
      <td>7.300000</td>
      <td>5.750000</td>
      <td>6.157895</td>
      <td>6.150000</td>
      <td>6</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6815</th>
      <td>552</td>
      <td>several times a week</td>
      <td>twice a week</td>
      <td>7.300000</td>
      <td>5.750000</td>
      <td>6.157895</td>
      <td>6.150000</td>
      <td>6</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>6698 rows × 13 columns</p>
</div>




```python
self_rated_five_class_pivot = pd.pivot_table(partic_and_ratings,index='gender',columns=['self_rated_class'],values=['number of matches','self-rated attractiveness at sign-up','avg_attractiveness_rating','self-rated intelligence at sign-up', 'avg_intelligence_rating','avg_fun_rating'],aggfunc='mean')
self_rated_five_class_pivot = self_rated_five_class_pivot.unstack()
self_rated_five_class_pivot = self_rated_five_class_pivot.unstack()
self_rated_five_class_pivot = self_rated_five_class_pivot.unstack()
```


```python
partner_rated_five_class_pivot = pd.pivot_table(partic_and_ratings,index='gender',columns=['partner_rated_class'],values=['number of matches','self-rated attractiveness at sign-up','avg_attractiveness_rating','self-rated intelligence at sign-up', 'avg_intelligence_rating','avg_fun_rating'],aggfunc='mean')
partner_rated_five_class_pivot = partner_rated_five_class_pivot.unstack()
partner_rated_five_class_pivot = partner_rated_five_class_pivot.unstack()
partner_rated_five_class_pivot = partner_rated_five_class_pivot.unstack()

```


```python
speed_data = pd.read_csv('./Speed Dating Data2.csv', encoding = 'iso-8859-1')
speed_data.shape
```




    (8378, 195)




```python

```


```python

```
