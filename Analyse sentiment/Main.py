import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

nlp = spacy.load("fr_core_news_sm")
nltk.download('stopwords')

df = pd.read_csv('C:/Users/21260/Desktop/Master/Python/AI/AI Training/Analyse sentiment/avis.csv')

#print(df.sample(6))

df['Note'] = df['Note'].str.replace(',','.')
df['Avis'] = df['Note'].apply(lambda x: 'positif' if float(x)>=3 else 'négatif')

#print(df.sample(6))

def delete_punctuation(text):
    text = re.sub(r'[^\w\s]', '', text)  
    text = re.sub(r'\d+', '', text)     
    text = text.strip()                 
    return text

df['Description'] = df['Description'].apply(delete_punctuation)

stop_words = set(stopwords.words('french'))

def tokenizer(text):
    tokens = word_tokenize(text.lower())   
    tokens = [t for t in tokens if t not in stop_words]  
    token=' '.join(tokens)
    return token

df['Description'] = df['Description'].apply(tokenizer)


vectorizer = TfidfVectorizer()

tfidf = vectorizer.fit_transform(df['Description'])

tfidf_df = pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names_out())

df = pd.concat([df, tfidf_df], axis=1)

df.drop('Note', axis=1, inplace=True)
df.drop('Description', axis=1, inplace=True)
df.drop('key', axis=1, inplace=True)

#print(df.head())

X = df.loc[:, df.columns != "Avis"]
y = df.Avis
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
#print(X.shape)
"""
#knn
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc = metrics.accuracy_score(y_test, y_pred)
print("kNN model accuracy:",acc,"estt:0.30")


#svm
svm=SVC()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
acc = metrics.accuracy_score(y_test, y_pred)
print("svm model accuracy:",acc,"estt:0.30")
#tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred=clf.predict([X_test])
score=metrics.accuracy_score(y_test, y_pred)
print("son score est :",score)
"""
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
y_pred=clf.predict(X_test)
#print(clf.predict_proba(X_test))
print(clf.score(X_test, y_test))

print(classification_report(y_test, y_pred))

new_comment = "Haaaaaa..... Que dire de ce film? Je crains d'en froisser quelques uns si je dis ce que j'en pense... Mon premier Nolan, que je crois bien dÃ©cevant.Moi qui m'attendais a une Å“uvre hors du commun (c'est dire...), je me retrouve avec un pseudo Matrix Ã  la sauce blockbuster. Bien sur l'idÃ©e de dÃ©part est tout a fait originale, et on pouvais s'attendre Ã  un film d'une envergure impensable! Mais non, tout l'ensemble reste plat, et ne dÃ©colle a aucun moment! Le jeu de Di caprio d'habitude si prenant, est monolithique d'un bout Ã  l'autre, sans jamais s'Ã©garer, sans jamais rien bousculer ( ce qui est, soit dit en passant profondÃ©ment ennuyant).Certes certains effets spÃ©ciaux valent la peine d'Ãªtres vus, mais mis Ã  par cela... pourquoi diable faire un film si mal rÃ©alisÃ©? Telle est la question.Insignifiant d'un bout a l'autre, ce film doit donner Ã  certains l'impression de rÃ©flÃ©chir, ou de comprendre quelque chose. Pourtant mes amis j'ai l'immense regret de vous annoncer qu'il n'y a aucun sens profond et vÃ©ritable dans ce que vous voyez lÃ .Haaaa... douleur de douleur... Un cour instant j'ai bien cru m'endormir.De telles longueurs sont intolÃ©rables. Surtout lorsqu'elles n'ont aucune signification propre et qu'elles ne servent en rien Ã  l'intÃ©rÃªt du film. Mais ne nous Ã©garons pas sur de tels sentiers.. Par ailleurs en sortant de la salle de cinÃ©ma, j'ai trouvÃ© quelque chose d'assez drÃ´le voyez vous.. Chaque personne prÃ©sente dans la salle c'est posÃ© l'unique et mÃªme question : ""Mais Ã  la fin la toupie elle tombe ou pas??"" Je me croyais sur le point d'Ã©clater de rire, voilÃ  a quoi se rÃ©sume le cinÃ©ma de nos jours.. Quelle tristesse.Bien, je crois en avoir fini"
new_comment_tfidf = vectorizer.transform([new_comment])

new_comment_pred = clf.predict(new_comment_tfidf)

print(new_comment_pred)