import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import time

# -------------get glove words from word corpus-------------------------
words_corpus = joblib.load('pickle_files/words_corpus.pkl')
glove_words = set(words_corpus.keys())

# --------------load tfidf------------------------------
tfidf = joblib.load('pickle_files/tfidf.pkl')
# get tfidf words
tfidf_words = set(tfidf.get_feature_names())
# dictionary of tfidf
dictionary = dict(zip(tfidf.get_feature_names(), list(tfidf.idf_)))

#-----------------load tfidf_w2v_vectors of corpus----------------
tfidf_w2v_vector = joblib.load('pickle_files/tfidf_glove_vectors.pkl')

#-------------------read dataframe -----------------------------
dff = pd.read_csv('aaic_task_data.csv')


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"\n", "", phrase)
    return phrase

def remove_stopwords(text):
    '''this function will remove stopwords from text using nltk stopwords'''
    final_text = ''
    for word in text.split():
        if word not in stopwords.words('english'):
            final_text += word + ' '
    return final_text

def preprocess_title(text):
    # convert to lower case
    text = text.lower()
    # decontract
    text = decontracted(text)
    # remove all punctuations except a-z and c# and c++
    text = re.sub('[^a-z]+',' ',text)
    # remove stop words
    text = remove_stopwords(text)
    return text


def find_similarity(title,top_n):
    ''' This function will find top similar result for given query'''
    start = time.time()
    # initialize  vector for user query
    main_vec = np.zeros(300)
    # initialize tfidf weight
    weight_sum = 0
    # preprocess question
    text = preprocess_title(title)
    #splitting the sentence
    text_list = list(text.split())
    for word in text_list:
        #finding if word is present in tfidf and in w2v words
        if word in tfidf_words and word in glove_words :
            #finding vector of word from glove model
            vect = words_corpus[word]
            #compute tfidf
            tf_idf = dictionary[word]*(text_list.count(word)/len(text_list)) 
            # adding vector * tfidf to main_vec
            main_vec+= (vect*tf_idf)
            # summing tfidf values
            weight_sum += tf_idf
    if weight_sum !=0:
        # devide by weight_sum
        main_vec /= weight_sum
    # find cosine similarity
    similarities =  cosine_similarity((main_vec).reshape(1, -1), Y=tfidf_w2v_vector, dense_output=True)
    # sort similarities 
    #print(similarities[0])
    
    sort = np.argsort(similarities[0])
    # get top similarity indices  in descending order
    similarity_index = np.array(list(reversed(sort)))
    # finad top n similarities
    top_similarity_index = similarity_index[:top_n]
    # print top similarity values
    # print('Top cosine similarities are ======>',similarities[0][top_similarity_index])
    # get original title of similar questions
    similar_questions = dff.iloc[top_similarity_index]
    #print(similar_questions)
    
    total_time = (time.time() - start)
    print('Total time ===========> ',total_time)
    return list(top_similarity_index+1)