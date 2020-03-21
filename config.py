# paths
qa_path = '/mnt/crawl/ryan/count_demo/vqa-v2/data_aug_bert'
# qa_path = '/mnt/crawl/demo_data'
# qa_path = '/mnt/crawl/kordata'  # directory containing the question and annotation jsons
# qa_path = '/mnt/crawl/ryan/backup/counting/vqa-v2/data' # english
# qa_path = '/mnt/backup/ran/vqa'
# korean
# qa_path = '/mnt/crawl/ryan/backup/counting/vqa-v2/data_kor'  # directory containing the question and annotation jsons

root = '/mnt/crawl/mvdata/kordata/' # korean
# root = '/mnt/backup/ran/vqa/'
# bottom_up_trainval_path = '/mnt/ryan_folder/counting/vqa-v2/data/' + 'trainval36'  # directory containing the .tsv file(s) with bottom up features
bottom_up_trainval_path = '/mnt/crawl/ryan/count_demo/vqa-v2/imageh5/'
# bottom_up_trainval_path = './data/sample_image'
bottom_up_test_path = root + 'test2015'  # directory containing the .tsv file(s) with bottom up features
# preprocessed_trainval_path = root + 'genome-trainval36.h5'  # path where preprocessed features from the trainval split are saved to and loaded from
preprocessed_trainval_path = '/mnt/crawl/ryan/count_demo/vqa-v2/imageh5/trainSample.h5'  # path where preprocessed features from the trainval split are saved to and loaded from

preprocessed_test_path = '/mnt/crawl/data/' + 'genome-test36.h5'  # path where preprocessed features from the test split are saved to and loaded from
vocabulary_path = '/mnt/crawl/ryan/count_demo/vqa-v2/' + 'vocab_demo_final.json'  # path where the used vocabularies for question and answers are saved to

task = 'OpenEnded'
dataset = 'mscoco'

test_split = 'test2015'  # either 'test-dev2015' or 'test2015'

# preprocess config
output_size = 100  # max number of object proposals per image
output_features = 2048  # number of features in each object proposal
# output_size = 100
# output_features = 1024 # 1536

# training config
epochs = 1000
batch_size = 1024
initial_lr = 1.5e-3  # 0.0015
lr_halflife = 50000  # in iterations
data_workers = 16
max_answers = 160

# MCB
att_fusion_input_size = 100
att_fusion_output_size = 100

classi_fusion_input_size = 1
classi_fusion_output_size = 1024
