from Seq2Seq.seq2seq_model import *
from parametrs import DialogQAParams as dqap

load_data(dqap,1000)
define_graph(dqap, 'v4.1', True)
train('logs_v4.1', dqap)
inputt = input('Enter question\n')
while inputt != 'q':
    translate(inputt, dqap)
    inputt = input('Enter question\n')
compute_blue(data1_validation, BLEU_score, dqap)

# import data_preprocessor
# data_preprocessor.Diaalog_QA()

# data_preprocessor.generate_data_from_clusters('datasets/DialogQA/result300',dqap.train_questions,dqap.train_answers)


# data_preprocessor.create_Diaalog_QA_data()
