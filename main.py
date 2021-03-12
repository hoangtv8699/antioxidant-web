from flask import Flask, request, render_template, current_app

app = Flask(__name__)
app.config['SERVER_NAME'] = '127.0.0.1:5000'

import my_utils
from process_theads import ProcessThread


@app.route("/")
def index():
    return render_template("problem.html")


@app.route("/confirm", methods=['GET', 'POST'])
def confirm():
    if request.method == 'POST':
        fasta_seq = request.form['seq']
        email = request.form['email']
        fasta_seq = fasta_seq.strip()

        Thread = ProcessThread(fasta_seq, email)
        Thread.start()

        str_result = "Email: " + email
        str_result += "<br>FASTA sequence: " + fasta_seq

        str_result = "We have received your request: <br><br> " + str_result
        str_result += "<br><br>We will send you the result one your job is finished."
        str_result += "<br> Press Back for other request!"

        return str_result


@app.route("/result/<filename>")
def result(filename):
    response_status, result_dict = my_utils.read_response_problem(filename)
    if not response_status:
        return "your result not yet calculated!"

    str_result = "Your email: " + result_dict['email']
    str_result += "<br>Your FASTA sequence: " + result_dict['seq']
    str_result += "<br>Your result: " + result_dict['result']
    return str_result


if __name__ == "__main__":
    app.run()
