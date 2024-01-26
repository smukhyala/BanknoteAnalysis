import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#gathering file
data = pd.read_csv('em-9EhjTEemU7w7-EFnPcg_7aa34fc018d311e980c2cb6467517117_happyscore_income.csv')

#grouping the data
happy = data['happyScore']
income = data['avg_income']
inequality = data['income_inequality']

#set up pyplot structure
plt.xlabel('income')
plt.ylabel('happy score')
plt.scatter(income, happy)

#basic data sorting
data.sort_values('avg_income', inplace=True)
richest = data[ data['avg_income'] > 15000 ]
richest.iloc[-1]
rich_mean = np.mean(richest['avg_income'])
all_mean = np.mean(data['avg_income'])

#creating a second scatter for the happiness of the highest income countries
plt.scatter(richest['avg_income'], richest['happyScore'])

plt.text(richest.iloc[0]['avg_income'], 
         richest.iloc[0]['happyScore'], 
         richest.iloc[0]['country'])
plt.text(richest.iloc[-1]['avg_income'], 
         richest.iloc[-1]['happyScore'], 
         richest.iloc[-1]['country'])

for k,row in richest.iterrows():
    print(row['country'])

plt.scatter(richest['avg_income'], richest['happyScore'])

for k,row in richest.iterrows():
    plt.text(row['avg_income'], row['happyScore'], row['country'])

#another method of plotting, using the built-in Pandas plot function:
scatter2 = richest.plot(x='avg_income', y='happyScore', kind='scatter')
scatter2.set_xlabel('average income')
scatter2.set_ylabel('happy score')

for index in richest.index:
    scatter2.text(richest.loc[index, 'avg_income'], richest.loc[index, 'happyScore'], richest.loc[index, 'country'])

plt.xlabel('income')
plt.ylabel('happy score')
plt.scatter(income, happy, s = inequality*10, alpha=0.25)

#using scikitplot
from sklearn.cluster import KMeans
income_happy = np.column_stack((income, happy))

km_res = KMeans(n_clusters = 3).fit(income_happy)

clusters = km_res.cluster_centers_

plt.scatter(income, happy)
plt.scatter(clusters[:,0], clusters[:,1], s = 100)
plt.xlabel('average income')
plt.ylabel('happiness score')

plt.text(data.iloc[0]['avg_income'], 
         data.iloc[0]['happyScore'], 
         data.iloc[0]['country'])

plt.text(data.iloc[-1]['avg_income'], 
         data.iloc[-1]['happyScore'], 
         data.iloc[-1]['country'])

