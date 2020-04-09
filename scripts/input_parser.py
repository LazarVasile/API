from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
stemmer = SnowballStemmer("romanian")
# print(stemmer.stem("frumosul"))
stop_words = stopwords.words("romanian")

punctuation = re.compile(r'[-.?!:;\'\"\`\\\/\|()]')

#message = input()
# message = stemmer.stem(message)
#tokens_list = word_tokenize(message)
#print(tokens_list)

def delete_punctuation(tokens_list):
    my_list = list()

    for item in tokens_list:
        word = punctuation.sub("", item)
        if len(word) > 0:
            my_list.append(word)

    return my_list

#tokens_list = delete_punctuation(tokens_list)
#print(tokens_list)

def transform_lower_case(tokens_list):
    my_list = list()
    
    for item in tokens_list:
        my_list.append(str.lower(item))
    
    return my_list

#tokens_list = transform_lower_case(tokens_list)
#print(tokens_list)

def delete_stopwords(tokens_list):
    my_list = list()

    for item in tokens_list:
        if item not in stop_words:
            my_list.append(item)
        
    return my_list

#tokens_list = delete_stopwords(tokens_list)
#print(tokens_list)

def stemming_words(tokens_list):
    my_list = list()

    for item in tokens_list:
        my_list.append(stemmer.stem(item))
    
    return my_list

#tokens_list = stemming_words(tokens_list)
#print(tokens_list)

def delete_same_words(tokens_list):
    my_list = list()

    for item in tokens_list:
        if item not in my_list:
            my_list.append(item)
    
    return my_list

#tokens_list = delete_same_words(tokens_list)
#print(tokens_list)

my_words = list(["fragment", "text", "paragraf", "capitol", "secventa"])


def delete_my_words(tokens_list):
    my_list = list()

    for item in tokens_list:
        if item not in my_words:
            my_list.append(item)

    return my_list

#tokens_list = delete_my_words(tokens_list)
#print(tokens_list)

#from nltk.stem import WordNetLemmatizer
#import nltk
#word_lem = WordNetLemmatizer()

#x = word_lem.lemmatize("I am a boy")
#print(x)

def input_parser(message):
    tokens_list = word_tokenize(message)
    tokens_list = delete_punctuation(tokens_list)
    tokens_list = transform_lower_case(tokens_list)
    tokens_list = delete_stopwords(tokens_list)
    tokens_list = stemming_words(tokens_list)
    tokens_list = delete_same_words(tokens_list)
    tokens_list = delete_my_words(tokens_list)
    return tokens_list
