from Seq2Seq.seq2seq_model import *
load_data(250)
define_graph('new_v3', True)
train('logs_v3')
# inputt = input('Enter question\n')
# while inputt != 'q':
#     translate(inputt)
#     inputt = input('Enter question\n')
# compute_blue(data1_validation,BLEU_score)
