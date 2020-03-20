from flask import Flask, redirect, url_for, request,render_template_string
import time
app = Flask(__name__)


@app.route('/success/yoursearch')
def success_1():
    return 'Langdon felt a fresh wave of remorse engulfing him. From the sounds of the message,Dr. Marconi had been permitting Sienna to work at the hospital. Now Langdon’s presencehad cost Marconi his life, and Sienna’s instinct to save a stranger had dire implications forher future.'    

@app.route('/success')
def success():
    time.sleep(3)
    return redirect(url_for('success_1'))

@app.route('/failure')
def failure():
    return 'Please try again'

@app.route('/search',methods = ['POST'])
def search():
    if request.method == 'POST':
        search_text = request.form['key']
        if len(search_text) != 0:
            print("Ai cautat ", search_text)
            return redirect(url_for('success'))
        else:
            print("Gol")
            return redirect(url_for('failure'))


if __name__ == '__main__':
   app.run(debug = True)