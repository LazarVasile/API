import flask
from flask import request, jsonify
import json
from scripts.prediction import fragment_prediction

app = flask.Flask(__name__)
app.config["DEBUG"] = True

with open("references.json", "r", encoding="iso8859_2") as f:
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
        if len(question) > 0:
            # intrebarea va fi trecuta prin reteaua neuronala si se va scoate fragmentul care se potriveste
            # vom avea nevoie si de id-ul
            fragment = fragment_prediction(question,id_book)
            return jsonify("Fragment found: question -> " + question + fragment)
        else:
            return jsonify("Fragment not found!")
    # return jsonify(results)

app.run()

#if __name__ == "__main__":
#    app.debug = True
#    app.run()