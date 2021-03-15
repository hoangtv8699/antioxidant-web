from flask import Flask, request, render_template, abort

app = Flask(__name__)
app.config['SERVER_NAME'] = '127.0.0.1:5000'

from FlaskApp import my_utils
from FlaskApp.process_theads import ProcessThread


@app.route("/")
def index():
    return render_template("home.html")


@app.route("/confirm", methods=['GET', 'POST'])
def confirm():
    if request.method == 'POST':
        fasta_seq = request.form['seq']
        email = request.form['email']
        fasta_seq = fasta_seq.strip()

        Thread = ProcessThread(fasta_seq, email)
        Thread.start()

        return render_template("home.html", result=True)


@app.route("/result/<filename>")
def result(filename):
    response_status, result_dict = my_utils.read_response_problem(filename)
    if not response_status:
        abort(404)
    return render_template("result.html", email=result_dict['email'],
                           seq=result_dict['seq'], result=result_dict['result'])


if __name__ == "__main__":
    app.run()
