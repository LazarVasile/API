import flask
from flask import request, jsonify
import json
# from scriptss.prediction import prediction
# from scriptss.input_parser import input_parser
import time

app = flask.Flask(__name__)
app.config["DEBUG"] = True

with open("references2.json", "r", encoding="utf-8") as f:
    books = json.load(f)


@app.route('/api/v1/resources/books/all', methods = ['GET'])
def api_all():
    return jsonify(books)



@app.route('/api/v1/resources/books/<string:title>', methods = ['GET'])
def get(title):
    for book in books:
        if book["title"] == title:
            return jsonify(book)
            

@app.route('/api/v1/resources/books', methods = ['POST'])
def api_question():
    if request.method == 'POST':
        question = request.json['question']
        id_book = request.json['id']
        #title_book = request.json['title']
        #author_book = request.json['author']
        # print(question)
        time.sleep(5)
        if len(question) > 0:
            print(question)
            # intrebarea va fi trecuta prin reteaua neuronala si se va scoate fragmentul care se potriveste
            # vom avea nevoie si de id-ul
            # fragment = fragment_prediction(question,id_book)
            # return jsonify(fragment)
            return jsonify("Fragmentul a fost găsit. Totul este în regulă. Așa sper să fie.")
        else:
            return jsonify("Fragmentul nu a fost găsit!")
    # return jsonify(results)

app.run(host = '192.168.0.101')

#if __name__ == "__main__":
#    app.debug = True
#    app.run()