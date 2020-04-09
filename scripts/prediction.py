from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_yaml
import numpy as np
import os
import json

from scripts.input_parser import input_parser

dataDir = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__),'../data')))

def bow(sentence_words, words):
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1

    return np.array(bag)

def fragment_prediction(sentence, filename): #filename se paseaza cu extensia .json

    sentence_words = input_parser(sentence)

    with open(os.path.join(dataDir,filename.replace(".json", '_') + "words.txt"),"r", encoding="iso8859_2") as f:
        words = json.loads(f.read())
    words = sorted(words)
    # load YAML and create model
    yaml_file = open(os.path.join(dataDir,filename.replace(".json", '_') + "model.yaml"), 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()

    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights(os.path.join(dataDir,filename.replace(".json", '_') + "model.h5"))
    #print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    x = bow(sentence_words, words)
    x = np.reshape(x, (-1, len(words)))

    # make fragment prediction with the model 
    prediction = loaded_model.predict(x)

    #prediction[0] -> predictia cu scorul cel mai mare

    with open(os.path.join(dataDir,filename), "r", encoding="iso8859_2") as f:
        book_info = json.load(f)
    
    return (book_info[str(np.argmax(prediction) + 1)]) #prediction[0]+1 pentru ca output incepe de la 0 si id-urile fragmentelor incep de la 1
    
#print(fragment_prediction("Budapesta","2000_a_3325.json"))