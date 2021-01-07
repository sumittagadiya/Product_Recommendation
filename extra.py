'''import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import time'''

from ml_model import find_similarity

text = 'womens unique 100 cotton  special olympics world games 2015 white size l'
top_10 = find_similarity(text,10)
print(top_10['product_type_name'])