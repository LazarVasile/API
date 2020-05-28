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

from scripturi.network import BidirectionalAttentionFlow

app = flask.Flask(__name__)
app.config["DEBUG"] = True


bidaf_model = BidirectionalAttentionFlow(emdim=300)

model_name = "bidaf_2.h5"

bidaf_model.load_bidaf(os.path.join(os.path.dirname(__file__), 'data', model_name))

print("Model loaded!")




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

        if len(question) > 0:

            if chapter == -1:
                f = open("./data/books/{}".format(id_book.replace(".json",".txt")), "r")
                passage = f.read()
                f.close()
            else:
                f = open("./data/books_json/{}".format(id_book), "r")
                data = json.load(f)
                f.close()
                passage = data['Chapters'][chapter]

            answer = bidaf_model.predict_ans(passage, question)
            
            return jsonify(answer)
        else:
            return jsonify("Fragmentul nu a fost găsit!")



app.run(host="192.168.1.4")


#if __name__ == "__main__":
#    app.debug = True
#    app.run()