import nltk
import gensim
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize

# Open file and tokenize words
file_docs = []

with open ('TextDoc1.txt') as f:
    tokens = sent_tokenize(f.read())
    for line in tokens:
        file_docs.append(line)

# Tokenize words and create library
gen_docs = [[w.lower() for w in word_tokenize(text)] 
            for text in file_docs]
dictionary = gensim.corpora.Dictionary(gen_docs)

# Create a bag of words
corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]

#TFIDF
tf_idf = gensim.models.TfidfModel(corpus)

# Creating similarity measure object
sims = gensim.similarities.Similarity('workdir/',tf_idf[corpus],
                                        num_features=len(dictionary))

# Create Query Document
file2_docs = []

with open ('TextDoc2.txt') as f:
    tokens = sent_tokenize(f.read())
    for line in tokens:
        file2_docs.append(line)

for line in file2_docs:
    query_doc = [w.lower() for w in word_tokenize(line)]
    #update an existing dictionary and create bag of words
    query_doc_bow = dictionary.doc2bow(query_doc)

# perform a similarity query against the corpus
query_doc_tf_idf = tf_idf[query_doc_bow]

sum_of_sims =(np.sum(sims[query_doc_tf_idf], dtype=np.float32))

avg_sims = [] # array of averages

# for line in query documents
for line in file2_docs:
        # tokenize words
        query_doc = [w.lower() for w in word_tokenize(line)]
        # create bag of words
        query_doc_bow = dictionary.doc2bow(query_doc)
        # find similarity for each document
        query_doc_tf_idf = tf_idf[query_doc_bow]
        # print (document_number, document_similarity)
        print('Comparing Result:', sims[query_doc_tf_idf]) 
        # calculate sum of similarities for each query doc
        sum_of_sims =(np.sum(sims[query_doc_tf_idf], dtype=np.float32))
        # calculate average of similarity for each query doc
        avg = sum_of_sims / len(file_docs)
        # print average of similarity for each query doc
        print(f'avg: {sum_of_sims / len(file_docs)}')
        # add average values into array
        avg_sims.append(avg)  
        # calculate total average
        total_avg = np.sum(avg_sims, dtype=np.float)
        # round the value and multiply by 100 to format it as percentage
        percentage_of_similarity = round(float(total_avg) * 100)
        
        if percentage_of_similarity >= 100:
            percentage_of_similarity = 100
        print('Percentage Similarity:', percentage_of_similarity)    
        