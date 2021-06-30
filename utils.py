import pysftp
import time
import datetime
import os
import math
from Bio import SeqIO

DIR_REQUESTS = 'requests/'
DIR_RESPONSES = 'responses/'

DIR_REMOTE_RESPONSES = 'Antioxidant_API/responses/'
DIR_REMOTE_REQUESTS = 'Antioxidant_API/data/fasta/'


def create_file_name():
    currentDT = datetime.datetime.now()
    # print("Current Year is: %d" % currentDT.year)
    # print("Current Month is: %d" % currentDT.month)
    # print("Current Day is: %d" % currentDT.day)
    # print("Current Hour is: %d" % currentDT.hour)
    # print("Current Minute is: %d" % currentDT.minute)
    # print("Current Second is: %d" % currentDT.second)
    # print("Current Microsecond is: %d" % currentDT.microsecond)

    result = str(currentDT.year) + "_" + str(currentDT.month) + "_" + str(currentDT.day) + "__" \
             + str(currentDT.hour) + "_" + str(currentDT.minute) + "_" + str(currentDT.second)
    return result


def transfer(filename):
    cnopts = pysftp.CnOpts()
    cnopts.hostkeys = None
    try:
        # Tạo kết nối qua sFTP
        with pysftp.Connection('118.70.187.15', username='hoangtv', password='abc@123', cnopts=cnopts,
                               port=23968) as sftp:
            # file = sftp.get('test.fasta')
            sftp.put(DIR_REQUESTS + filename + '.fasta', DIR_REMOTE_REQUESTS + filename + '.fasta')
            # sftp.remove('test.fasta')
            while not sftp.exists(DIR_REMOTE_RESPONSES + filename + '.json'):
                time.sleep(1)
            sftp.get(DIR_REMOTE_RESPONSES + filename + '.json', DIR_RESPONSES + filename + '.json')
            sftp.remove(DIR_REMOTE_RESPONSES + filename + '.json')
        sftp.close()
        return True
    except Exception as e:
        return False


def save_seq(seq):
    file_name = create_file_name()
    file_path = DIR_REQUESTS + file_name + '.fasta'
    f_request = open(file_path, "w")
    f_request.write(seq)
    f_request.close()
    return file_name


def save_fasta(fasta_file):
    file_name = create_file_name()
    fasta_file.save(DIR_REQUESTS + file_name + '.fasta')
    if not check_fasta(file_name):
        os.remove(DIR_REQUESTS + file_name + '.fasta')
        return False
    return file_name


def check_fasta(filename):
    try:
        file_path = DIR_REQUESTS + filename + '.fasta'
        with open(file_path, "r") as handle:
            fasta = SeqIO.parse(handle, "fasta")
            return any(fasta)
    except:
        return False


def floor_10fold(fold):
    for key, value in fold.items():
        value[0] = math.floor(value[0] * 1000) / 1000
        value[1] = math.floor(value[1] * 1000) / 1000
    return fold
