from flask import Flask, flash, request, redirect, url_for, render_template
import time
import my_utils
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("problem.html")


@app.route("/result", methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        fasta_seq = request.form['seq']
        email = request.form['email']

        fasta_seq = fasta_seq.strip()

        my_utils.save_request_problem(fasta_seq, email)

        str_result = "Email: " + email
        str_result += "<br>FASTA sequence: " + fasta_seq

        str_result = "We have received your request: <br><br> " + str_result
        str_result += "<br><br>We will send you the result one your job is finished."
        str_result += "<br> Press Back for other request!"

        return str_result


if __name__ == "__main__":
    app.run()
