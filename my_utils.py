import datetime
import os
import re
import math
import tensorflow as tf
import pandas as pd
import numpy as np
import subprocess
from Bio import SeqIO
from joblib import dump, load
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import sequence as seq
from transformers import BertModel, BertTokenizer

from flask import Flask, url_for
from flask_mail import Mail, Message
from main import app

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'sktt1nhox99@gmail.com'
app.config['MAIL_PASSWORD'] = 'tranvanhoang99'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)


DIR_REQUESTS = "requests/"
DIR_RESPONSES = "responses/"

DIR_PSSM = 'data/pssm/'
DIR_BERT = 'data/bert/'
DIR_SEQ = 'data/sequences/'

DIR_MODEL_CNN = 'models/3518/'
DIR_MODEL_RF = 'models/RFtop1.joblib'
DIR_MODEL_DCT = 'models/DCT.joblib'


def create_file_name():
    currentDT = datetime.datetime.now()
    print("Current Year is: %d" % currentDT.year)
    print("Current Month is: %d" % currentDT.month)
    print("Current Day is: %d" % currentDT.day)
    print("Current Hour is: %d" % currentDT.hour)
    print("Current Minute is: %d" % currentDT.minute)
    print("Current Second is: %d" % currentDT.second)
    print("Current Microsecond is: %d" % currentDT.microsecond)

    result = str(currentDT.year) + "_" + str(currentDT.month) + "_" + str(currentDT.day) + "__" \
             + str(currentDT.hour) + "_" + str(currentDT.minute) + "_" + str(currentDT.second)
    return result


def save_request_problem(fasta_seq, email, name):
    file_name = name + ".txt"
    file_path = DIR_REQUESTS + file_name
    print(file_path)

    f_request = open(file_path, "w")
    f_request.write(">email\n")
    f_request.write(email + "\n")
    f_request.write(">seq\n")
    f_request.write(fasta_seq)
    f_request.close()


def save_result_problem(fasta_seq, email, name, result):
    file_path = DIR_RESPONSES + name + '.fasta'
    print(file_path)

    f_request = open(file_path, "w")
    f_request.write(">email\n")
    f_request.write(email + "\n")
    f_request.write(">seq\n")
    f_request.write(fasta_seq + "\n")
    f_request.write(">result\n")
    if result == 1:
        f_request.write("Antioxidant")
    elif result == 0:
        f_request.write("Non-Antioxidant")
    else:
        f_request.write(result)
    f_request.close()


def send_result_ready(email, link):
    msg = Message('Result ready', sender='sktt1nhox99@gmail.com', recipients=[email])
    msg.body = "Hello,\n" +\
        "your result checking antioxidant protein have been done!. you can click this link to see the result\n" +\
        link
    mail.send(msg)
    print('email sent!')


def read_response_problem(file_name):
    response_status = True

    file_path = DIR_RESPONSES + file_name + '.fasta'
    if not os.path.exists(file_path):
        return False, None

    result = {
        'email': '',
        'seq': '',
        'result': ''
    }

    with open(file_path) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            result[record.id] = str(record.seq)

    return response_status, result


def normalize(data):
    for j in range(len(data)):
        row = data[j]
        sample_mean = sum(row) / (len(row) + K.epsilon())
        standard_deviation = math.sqrt(sum([pow(x - sample_mean, 2) for x in row]) / (len(row) + K.epsilon()))
        data[j] = [(x - sample_mean) / (standard_deviation + K.epsilon()) for x in row]
    return np.asarray(data)


def get_top(DCT_path, top):
    clf = load(DCT_path)
    importance = clf.feature_importances_
    return np.argsort(importance)[::-1][:top]


def read_pssm(path):
    data = pd.read_csv(path, sep=',', header=None)
    data = np.asarray(data)
    data = normalize(data.T)
    data = seq.pad_sequences(data, maxlen=400, padding='post', truncating='post')
    return data


def read_bert(path):
    df = pd.read_csv(path, sep=',', header=None)
    df = np.asarray(df)
    df = df[0, 1:401]
    if df.shape[0] < 400:
        tmp = np.zeros(400 - df.shape[0])
        df = np.append(df, tmp)
    return df


def voting(models, data):
    pre = []
    pre_bin = []
    for model in models:
        pre_tmp = model.predict(data)
        pre_bin.append(np.argmax(pre_tmp, axis=-1))

    for i in range(len(pre_bin[0])):
        tmp = [0, 0]
        for x in pre_bin:
            tmp[x[i]] += 1
        tmp[0] = tmp[0] / 10
        tmp[1] = tmp[1] / 10
        pre.append(tmp)
    return pre


def ensemble(data_pssm_path, data_bert_path):
    # load models
    model_cnn_paths = os.listdir(DIR_MODEL_CNN)
    models_cnn = []
    for model_path in model_cnn_paths:
        models_cnn.append(tf.keras.models.load_model(DIR_MODEL_CNN + model_path, compile=False))
    model_rf = load(DIR_MODEL_RF)

    data_pssm = read_pssm(data_pssm_path)
    data_bert = read_bert(data_bert_path)

    data_pssm = np.expand_dims(data_pssm, axis=0)
    data_pssm = np.expand_dims(data_pssm, axis=-1)
    data_bert = np.expand_dims(data_bert, axis=0)

    print(data_pssm.shape)
    print(data_bert.shape)

    pssm_proba = voting(models_cnn, data_pssm)
    bert_proba = model_rf.predict_proba(data_bert)

    ave = []
    for i in range(len(bert_proba)):
        tmp = [(bert_proba[i][0] + pssm_proba[i][0]) / 2, (bert_proba[i][1] + pssm_proba[i][1]) / 2]
        ave.append(tmp)

    return np.argmax(ave)


def format_seq(sequence):
    return ' '.join(sequence)


tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
model = BertModel.from_pretrained("Rostlab/prot_bert")


def seq2bert(sequence, model_dct_path, filename):
    save_path = DIR_BERT + filename + '.csv'
    top1 = get_top(model_dct_path, 1)
    sequence = format_seq(sequence)
    encoded_input = tokenizer(sequence[0:1200], return_tensors='pt')
    output = model(**encoded_input)
    output = output.last_hidden_state.detach().numpy()[0].T
    output = output[top1]
    pd.DataFrame(output).to_csv(save_path, header=None, index=False)
    return save_path


def seg2pssm(sequence, filename):
    seq_name = DIR_SEQ + filename + '.txt'
    save_path = DIR_PSSM + filename + '.csv'

    f = open(seq_name, 'w')
    f.write(sequence)
    f.close()

    command = ['./ncbi-blast-2.9.0+/bin/psiblast', '-db', 'ncbi-blast-2.9.0+/bin/db/swissprot',
               '-evalue', '0.01', '-query', seq_name, '-out_ascii_pssm', save_path,
               '-num_iterations', '3', '-num_threads', '6']

    # command[6] = seq_name
    # command[8] = save_path
    process = subprocess.run(command)

    with open(save_path) as fd:
            content = fd.readlines()[3:-6]
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    # with open(_path + 'csv/' + pssm_file.replace("fasta", "csv"), 'w') as wfd:
    with open(save_path, 'w') as wfd:
        for index, line in enumerate(content):
            line = line[6:-92].strip()  # line[6:-92].strip() for only PSSM
            line = re.sub(r' +', ',', line)
            # csvLine = sequence[index] + ',' + line    # include label as first column
            csvLine = line
            # validity check
            cnt = csvLine.count(',')
            if cnt != 19:  # 20 for just PSSM, 42 for all, 19 for just PSSM and not include label
                isValid = False
            # print(csvLine)
            # write to csv files
            wfd.write(csvLine + '\n')
    return save_path


def preprocess(sequence):
    filename = create_file_name()
    pssm_path = seg2pssm(sequence, filename)
    bert_path = seq2bert(sequence, DIR_MODEL_DCT, filename)
    return pssm_path, bert_path


def process(sequence, email):
    name = create_file_name()
    save_request_problem(sequence, email, name)
    try:
        pssm_path, bert_path = preprocess(sequence)
        result = ensemble(pssm_path, bert_path)
        save_result_problem(sequence, email, name, result)
    except Exception:
        result = 'can\'t parse pssm from input sequence'
        save_result_problem(sequence, email, name, result)
    return name


if __name__ == "__main__":
    # fasta_seq = "DKIIHLTDDSFDTDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAKLNIDQNPGTAPKYGIRGIPTLLLFKNGEVAATKVGALSKGQLKEFLDANL"
    # threshold = 0.5
    # user_email = "abc@123"
    # save_request_problem(fasta_seq, user_email)

    # sequence = 'MRSPSLAVAATTVLGLFSSSALAYYGNTTTVALTTTEFVTTCPYPTTFTVSTCTNDVCQPTVVTVTEETTITIPGTVVCPVVSTPSGSASASASAGASSEEEGSVVTTQVTVTDFTTYCPYPTTLTITKCENNECHPTTIPVETATTVTVTGEVICPTTTSTSPKESSSEAASSEVITTQVTVTDYTTYCPLPTTIVVSTCDEEKCHPTTIEVSTPTTVVVPGTVVCPTTSVATPSQSEVATKPTTINSVVTTGVTTTDYTTYCPSPTTIVVSTCDEEKCHPTTIEVSTPTTVVVPGTVVHPSTSATIITTTAEQPPASPEVSTIESVVTTPATLTGYTTYCPEPTTIVLTTCSDDQCKPHTVSATGGETVSIPATIVVPSSHTTQVEITVSSASVPASEKPTTPVTVAAVSSSPAVSTETPSLVTPAISIAGAAAVNVVPTTAFGLFAIILASIF'
    # process(sequence, 'abc@gmail.com')

    with app.app_context():
        send_result_ready('nhoxkhang351@gmail.com', url_for('result', filename='test'))

    # response, result = read_response_problem('example')
    # print(result)
