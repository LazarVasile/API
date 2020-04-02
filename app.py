import flask
from flask import request, jsonify

app = flask.Flask(__name__)
app.config["DEBUG"] = True

books = [
    {   'id': 0,
        'title': "Eloquent JavaScript, Second Edition",
        'author': "Marijn Haverbeke",
    },
    {
        'id' : 1,
        'title' : 'Harry Potter',
        'author' : 'J.K. Rowling'
        
    },
    {
        'id' : 2,
        'title' : 'After',
        'author' : 'Anna Todd'
        
    },
    {
        'id' : 3,
        'title' : 'Ciocoii vechi si noi',
        'author' : 'Nicolae filimon'
        
    },
    {
        'id' : 4,
        'title' : 'Mara',
        'author' : 'Ioan Slavici'
        
    }
]

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
        title_book = request.json['title']
        author_book = request.json['author']
        if question == 'Who is the enemy of Harry Potter?':
            # intrebarea va fi trecuta prin reteaua neuronala si se va scoate fragmentul care se potriveste
            # vom avea nevoie si de id-ul, titlul si autorul cartii
            return jsonify("Fragment found: question -> " + question + 
            " id ->" + str(id_book) + " title-> " + title_book + 
            " author-> " + author_book)
        else:
            return jsonify("Fragment nout found!");
    # return jsonify(results)

app.run()

