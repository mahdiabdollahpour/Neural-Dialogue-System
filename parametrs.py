hidden_size = 512
source_sequence_length = 32
decoder_length = 45
max_gradient_norm = 5.0
batch_size = 64
layer_num = 2
embed_size = 100
lr = 0.001
epoch_num = 120

dataset_path = 'datasets/DailyDialog'
all_questions = dataset_path + '/Q_all.txt'
all_answers = dataset_path + '/A_all.txt'
train_questions = dataset_path + '/train/Q_train.txt'
train_questions_csv = dataset_path + '/train/Q_train.csv'
train_answers = dataset_path + '/train/A_train.txt'
train_answers_csv = dataset_path + '/train/A_train.csv'
validation_questions = dataset_path + '/validation/Q_validation.txt'
validation_questions_csv = dataset_path + '/validation/Q_validation.csv'
validation_answers_csv = dataset_path + '/validation/A_validation.csv'
validation_answers = dataset_path + '/validation/A_validation.txt'
test_questions = dataset_path + '/test/Q_test.txt'
test_questions_csv = dataset_path + '/test/Q_test.csv'
test_answers_csv = dataset_path + '/test/A_test.csv'
test_answers = dataset_path + '/test/A_test.txt'

vocab_path = dataset_path + '/vocab.csv'

model_file_name = '\model.ckpt'
config_file = '\config.json'
# log_file_name = '\logs.txt'

unknown_token = '<unk>'
empty_token = '<EMP>'
start_token = '<s>'
end_token = '</s>'
