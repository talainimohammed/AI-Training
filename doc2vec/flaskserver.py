from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
import gensim
import random

app = Flask(__name__)


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words and w not in string.punctuation]
    words = [lemmatizer.lemmatize(w) for w in words]
    return words


docs = []
articles=[]
for category in os.listdir('D:/Master ALX/Master/Python/AI/AI Training/doc2vec/bbc_english'):
    for file_name in os.listdir(f'D:/Master ALX/Master/Python/AI/AI Training/doc2vec/bbc_english/{category}'):
        with open(f'D:/Master ALX/Master/Python/AI/AI Training/doc2vec/bbc_english/{category}/{file_name}', 'r') as f:
            text = f.read()
            articles.append([category,text])
            words = preprocess(text)
            docs.append(TaggedDocument(words, [category]))
#print(articles)
model = Doc2Vec(vector_size=100, min_count=2, epochs=40)
model.build_vocab(docs)
model.train(docs, total_examples=model.corpus_count, epochs=model.epochs)

with open(f'D:/Master ALX/Master/Python/AI/AI Training/doc2vec/bbc_english/business/001.txt', 'r') as f:
            text1 = f.read()
            doc1 = preprocess(text1)
with open(f'D:/Master ALX/Master/Python/AI/AI Training/doc2vec/bbc_english/business/002.txt', 'r') as f:
            text2 = f.read()
            doc2 = preprocess(text2)


doc1_vector = model.infer_vector(doc1)
doc2_vector = model.infer_vector(doc2)
#doc1 = model[doc1]
#doc2 = model[doc2]
#print(doc1_vector)
# Calculer la similarité cosinus entre les deux vecteurs
#similarity = gensim.matutils.cossim(doc1_vector, doc2_vector)

#print(f"Similarité entre les documents : {similarity}")

@app.route('/')
def index():
    return render_template('index.html',bbc_docs=articles)

@app.route('/similar_docs', methods=['POST'])
def similar_docs():
    doc_id = request.form['doc_id']
    doc1_vector = model.infer_vector(preprocess(doc_id[1]))
    similar_docs = model.dv.most_similar(doc1_vector)
    return render_template('similar_docs.html', similar_docs=similar_docs, bbc_docs=docs)

if __name__ == '__main__':
    app.run()
