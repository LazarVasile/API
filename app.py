from flask import Flask, redirect, url_for, request,render_template_string, Response
import time
app = Flask(__name__)


@app.route('/api/question', methods = ['POST'])
def question():
    if request.method == 'POST':
        search_text = request.form['key']
        if len(search_text) != 0:
            fragment = 'asd'
            return Response("{'success':'true', 'fragment':" + str(fragment) + "}", status=200, mimetype='application/json')
        else:
            return Response("{'success':'false', 'message': 'Bad request! Please check the request body'}", status=400, mimetype='application/json')


@app.route('/api/question', methods = ['GET'])
def success():
    return Response("{'success' : 'true', 'message' : 'Please use the POST method to send the question!!'}", status=200, mimetype='application/json' )


if __name__ == '__main__':
   app.run(debug = True)