from flask import Flask, request, render_template, current_app
import utils
import time
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("home.html")


@app.route("/predict", methods=['POST'])
def result():
    global name
    sub_type = request.form['radio']

    if sub_type == 'Sequence':
        seq = request.form['seq']
        name = utils.save_seq(seq)
    elif sub_type == 'Fasta':
        if request.files.get('fasta'):
            fasta = request.files['fasta']
            name = utils.save_fasta(fasta)
            if not name:
                return False

    utils.transfer(name)

    # str_result = "Your email: " + result_dict['email']
    # str_result += "<br>Your FASTA sequence: " + result_dict['seq']
    # str_result += "<br>Your result: " + result_dict['result']
    return "Hello world!"


if __name__ == "__main__":
    app.run()
