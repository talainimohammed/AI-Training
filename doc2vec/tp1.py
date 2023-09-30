import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
import gensim
import random

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
for category in os.listdir('D:/Master ALX/Master/Python/AI/AI Training/doc2vec/bbc_english'):
    for file_name in os.listdir(f'D:/Master ALX/Master/Python/AI/AI Training/doc2vec/bbc_english/{category}'):
        with open(f'D:/Master ALX/Master/Python/AI/AI Training/doc2vec/bbc_english/{category}/{file_name}', 'r') as f:
            text = f.read()
            words = preprocess(text)
            docs.append(TaggedDocument(words, [category]))
#print(docs)
# train doc2vec model
#model = Doc2Vec(docs, vector_size=50, window=5, min_count=2, epochs=20)
model = Doc2Vec(vector_size=100, min_count=2, epochs=40)

# Construire le vocabulaire
model.build_vocab(docs)

# Entraîner le modèle
model.train(docs, total_examples=model.corpus_count, epochs=model.epochs)

#print(model)
with open(f'D:/Master ALX/Master/Python/AI/AI Training/doc2vec/bbc_english/business/001.txt', 'r') as f:
            text1 = f.read()
            doc1 = preprocess(text1)
with open(f'D:/Master ALX/Master/Python/AI/AI Training/doc2vec/bbc_english/business/002.txt', 'r') as f:
            text2 = f.read()
            doc2 = preprocess(text2)


doc1_vector = model.infer_vector(doc1)
doc2_vector = model.infer_vector(doc2)

# Calculer la similarité cosinus entre les deux vecteurs
#similarity = gensim.matutils.cosine_similarity(doc1_vector, doc2_vector)

#print(f"Similarité entre les documents : {similarity}")

# Pick a random document from the test corpus and infer a vector from the model
doc_id = random.randint(0, len(docs) - 1)
inferred_vector = model.infer_vector(docs[doc_id])
sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))

# Compare and print the most/median/least similar documents from the train corpus
print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(docs[doc_id])))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(docs[sims[index][0]].words)))