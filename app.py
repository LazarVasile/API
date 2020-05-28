# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session  
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True) 
# dynamically grow the memory used on the GPU  
# to log device placement (on which device the operation ran)  
                                    # (nothing gets printed in Jupyter, only if you run it standalone)



import flask
from flask import request, jsonify, Response
import json
import os
#import scripturi.processing
# from scripturi.network import BidirectionalAttentionFlow

app = flask.Flask(__name__)
app.config["DEBUG"] = True

# booksDir = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__),'data', 'books')))

# bidaf_model = BidirectionalAttentionFlow(emdim=300)

# model_name = "bidaf.h5"

# bidaf_model.load_bidaf(os.path.join(os.path.dirname(__file__), 'data', model_name))

# print("Model loaded!")




with open("references2.json", "r", encoding="utf-8") as f:
    books = json.load(f)


@app.route('/api/v1/resources/books/all', methods = ['GET'])
def api_all():
    return Response(json.dumps(books),content_type="application/json; charset=utf-8")




@app.route('/api/v1/resources/books/chapters', methods = ['GET'])
def get_chapters():
    if "id" in request.args:
        book_id = request.args['id']

    for book in books:
        if book_id  == book['id']:
            f = open("./data/books_json/{}".format(book_id), "r")
            data = json.load(f)
            f.close()
            my_dict = dict()
            my_dict["Chapters"] = list()
            for i in range(len(data["Chapters"])):
                my_dict["Chapters"].append(i)
            
            return jsonify(my_dict)   

@app.route('/api/v1/resources/books', methods = ['POST'])
def api_question():
    if request.method == 'POST':
        question = request.json['question']
        id_book = request.json['id']
        chapter = request.json['chapter']
        print(chapter)
        #title_book = request.json['title']
        #author_book = request.json['author']
        if len(question) > 0:
            # intrebarea va fi trecuta prin reteaua neuronala si se va scoate fragmentul care se potriveste
            # vom avea nevoie si de id-ul
            #with open(os.path.join(booksDir,id_book.replace("json","txt")), "r", encoding="utf-8") as f:
            #    book_content = f.read()
            #deschis fisierul cu cartea in data/books
            #passage = book_content

            # passage = "Tesla, Inc. este un constructor de automobile electrice de înaltă performanță, din Silicon Valley. Tesla a primit o atenție deosebită când au lansat modelul de producție Tesla Roadster, prima mașină sport 100 electrică. A doua mașina produsă de Tesla este Model S, 100 electric sedan de lux."

            # answer = bidaf_model.predict_ans(passage, question)
            
            return jsonify("Fragmentul a fost găsit!")
        else:
            return jsonify("Fragmentul nu a fost găsit!")
    # return jsonify(results)



""" bidaf_model = BidirectionalAttentionFlow(emdim=300)

model_name = "bidaf.h5"

bidaf_model.load_bidaf(os.path.join(os.path.dirname(__file__), 'data', model_name))

print("Model loaded!") """

app.run(host="192.168.0.100")


#if __name__ == "__main__":
#    app.debug = True
#    app.run()