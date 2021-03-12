import datetime
import os
import math
import tensorflow as tf
import pandas as pd
import numpy as np
from joblib import dump, load
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import sequence
from transformers import BertModel, BertTokenizer

DIR_REQUESTS = "requests/"
DIR_RESPONSES = "responses/"

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


def save_request_problem(fasta_seq, email):
    file_name = create_file_name() + ".txt"
    file_path = DIR_REQUESTS + file_name
    print(file_path)

    f_request = open(file_path, "w")
    f_request.write(">User_Email\n")
    f_request.write(email + "\n")
    f_request.write(">Fasta sequences:\n")
    f_request.write(fasta_seq)
    f_request.close()


def read_response_problem(file_name):
    # file_name = "2019_8_19__12_5_47_problem2.txt"
    response_status = True

    file_path = DIR_RESPONSES + file_name
    if not os.path.exists(file_path):
        return False, ""

    result = "RESULT"
    f = open(file_path, "r")
    result = f.readlines()
    f.close()

    result = "<br>" + '<br>'.join(result)
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
    data = sequence.pad_sequences(data, maxlen=400, padding='post', truncating='post')
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


def ensemble(model_cnn_path, model_rf_path, data_pssm_path, data_bert_path):
    # load models
    model_cnn_paths = os.listdir(model_cnn_path)
    models_cnn = []
    for model_path in model_cnn_paths:
        models_cnn.append(tf.keras.models.load_model(model_cnn_path + model_path, compile=False))
    model_rf = load(model_rf_path)

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


def seg2bert(sequence, model_dct_path, save_path):
    top1 = get_top(model_dct_path, 1)
    sequence = format_seq(sequence)
    encoded_input = tokenizer(sequence[0:1200], return_tensors='pt')
    output = model(**encoded_input)
    output = output.last_hidden_state.detach().numpy()[0].T
    output = output[top1]
    pd.DataFrame(output).to_csv(save_path, header=None, index=False)


def seg2pssm(sequence, save_path):
    a = 10


if __name__ == "__main__":
    # fasta_seq = "DKIIHLTDDSFDTDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAKLNIDQNPGTAPKYGIRGIPTLLLFKNGEVAATKVGALSKGQLKEFLDANL"
    # threshold = 0.5
    # user_email = "abc@123"
    # save_request_problem(fasta_seq, user_email)

    model_cnn_path = 'models/3518/'
    model_rf_path = 'models/RFtop1.joblib'
    data_pssm_path = 'data/pssm/121_0_test.csv'
    data_bert_path = 'data/bert/test2.csv'
    result = ensemble(model_cnn_path, model_rf_path, data_pssm_path, data_bert_path)
    print(result)

    # model_dct_path = 'models/DCT.joblib'
    # sequence = 'MRSPSLAVAATTVLGLFSSSALAYYGNTTTVALTTTEFVTTCPYPTTFTVSTCTNDVCQPTVVTVTEETTITIPGTVVCPVVSTPSGSASASASAGASSEEEGSVVTTQVTVTDFTTYCPYPTTLTITKCENNECHPTTIPVETATTVTVTGEVICPTTTSTSPKESSSEAASSEVITTQVTVTDYTTYCPLPTTIVVSTCDEEKCHPTTIEVSTPTTVVVPGTVVCPTTSVATPSQSEVATKPTTINSVVTTGVTTTDYTTYCPSPTTIVVSTCDEEKCHPTTIEVSTPTTVVVPGTVVHPSTSATIITTTAEQPPASPEVSTIESVVTTPATLTGYTTYCPEPTTIVLTTCSDDQCKPHTVSATGGETVSIPATIVVPSSHTTQVEITVSSASVPASEKPTTPVTVAAVSSSPAVSTETPSLVTPAISIAGAAAVNVVPTTAFGLFAIILASIF'
    # seg2bert(sequence, model_dct_path, 'data/bert/test2.csv')

