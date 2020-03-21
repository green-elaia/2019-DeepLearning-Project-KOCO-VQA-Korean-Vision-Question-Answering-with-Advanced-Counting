# paths
qa_path = '/mnt/crawl/engdata/' #ko-data/'  # directory containing the question and annotation jsons
bottom_up_trainval_path = 'data/trainval36'#/mnt/crawl/ryan/trainval'#'data/trainval36'  # directory containing the .tsv file(s) with bottom up features
bottom_up_test_path = '/mnt/crawl/data/test2015'#'data/trainval36' #/mnt/crawl/data/test2015'  # directory containing the .tsv file(s) with bottom up features
preprocessed_trainval_path = 'genome-trainval36.h5' #'/mnt/crawl/data/genome-trainval36.h5'  # path where preprocessed features from the trainval split are saved to and loaded from
preprocessed_test_path = '/mnt/crawl/data/genome-test36.h5' #'genome-trainval36.h5'  # path where preprocessed features from the test split are saved to and loaded from
vocabulary_path = 'vocab_eng.json'  # path where the used vocabularies for question and answers are saved to

task = 'OpenEnded'
dataset = 'mscoco'

test_split = 'test2015'  # either 'test-dev2015' or 'test2015'



# preprocess config
output_size = 100  # max number of object proposals per image
output_features = 2048  # number of features in each object proposal
# output_size = 100
# output_features = 1024 # 1536

# training config
epochs = 100
batch_size_256 = 256#64#128#256 #512
#batch_size_512 = 512
initial_lr = 0.0015 #0.0001#1.5e-3 #0.0015
#initial_lr_01 = 0.0001
lr_halflife = 50000  #in iterations
data_workers = 12#12 #8
max_answers = 3000


#Mlb
att_fusion_input_size = 512
att_fusion_input_size = 512

classi_fusion_input_size = 1024
classi_fusion_input_size = 1024


#embedding config
question_embed = 300
embeding_data_link = 'embed_model/embedding_train_dataset.txt"'