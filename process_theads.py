import my_utils
from flask import url_for
from threading import Thread
from main import app


class ProcessThread(Thread):
    def __init__(self, seq, email):
        super(ProcessThread, self).__init__()
        self.seq = seq
        self.email = email

    def run(self):
        name = my_utils.process(self.seq, self.email)
        with app.app_context():
            my_utils.send_result_ready(self.email, url_for('result', filename=name))
