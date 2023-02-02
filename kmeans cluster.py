import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans

df=pd.read_csv("Clients.csv",encoding='latin-1')

print(df.shape)

for c in df.columns:
    stats = df[c].describe()
    print(stats)


print(df.dtypes)

print(df.head(5))

df.rename(columns = {'revenu annuel (kDH)': 'Revenu', 'Score de dépenses (1-100)': 'Score'}, inplace = True)

print(df.columns)

X = df[["Genre","Age","Revenu","Score"]]
#X.head()
#sns.pairplot(X, hue ='Genre',aspect=1.5)
#plt.show()

#Q10
#Le graphique montre que le genre n'a pas de rapport direct avec la segmentation de la clientèle. C'est pourquoi nous pouvons ignorer cette variable et passer à d'autres caractéristiques (Revenu et Score)

data = list(zip(X['Revenu'], X['Score']))
print(data)
l = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    l.append(kmeans.inertia_)
fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=list(range(1, 11)), y=clusters, ax=ax)
ax.set_title('Elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')

