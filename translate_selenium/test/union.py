import json
from tqdm import tqdm

path_list = []
"""
path_list.append('/mnt/crawl/VQA_v2/v2_Annotations_Val_mscoco/ko_v2_mscoco_val2014_annotations.json')
path_list.append('/mnt/ryan_folder/complete_crawl/val_anno/val_0_10.json')
path_list.append('/mnt/ryan_folder/complete_crawl/val_anno/val_10_20.json')
path_list.append('/mnt/ryan_folder/complete_crawl/val_anno/val_20_30.json')
path_list.append('/mnt/ryan_folder/complete_crawl/val_anno/val_30_40.json')
path_list.append('/mnt/ryan_folder/complete_crawl/val_anno/val_40_50.json')
path_list.append('/mnt/ryan_folder/complete_crawl/val_anno/val_50_60.json')
path_list.append('/mnt/ryan_folder/complete_crawl/val_anno/val_60_70.json')
path_list.append('/mnt/ryan_folder/complete_crawl/val_anno/val_70_80.json')
path_list.append('/mnt/ryan_folder/complete_crawl/val_anno/val_80_90.json')
path_list.append('/mnt/ryan_folder/complete_crawl/val_anno/val_90_100.json')
path_list.append('/mnt/ryan_folder/complete_crawl/val_anno/val_100_110.json')
path_list.append('/mnt/ryan_folder/complete_crawl/val_anno/val_110_120.json')
path_list.append('/mnt/ryan_folder/complete_crawl/val_anno/val_120_130.json')
path_list.append('/mnt/ryan_folder/complete_crawl/val_anno/val_130_140.json')
path_list.append('/mnt/ryan_folder/complete_crawl/val_anno/val_140_155.json')
path_list.append('/mnt/ryan_folder/complete_crawl/val_anno/val_155_170.json')
path_list.append('/mnt/ryan_folder/complete_crawl/val_anno/val_170_185.json')
path_list.append('/mnt/ryan_folder/complete_crawl/val_anno/val_185_200.json')
path_list.append('/mnt/ryan_folder/complete_crawl/val_anno/val_200_end.json')
"""
"""
path_list.append('/mnt/crawl/VQA_v2/v2_Annotations_Train_mscoco/ko_v2_mscoco_train2014_annotations.json')
path_list.append('/mnt/crawl/ryan/filter_anno/train/final_filter_train_1.json')
path_list.append('/mnt/crawl/ryan/filter_anno/train/final_filter_train_2.json')
path_list.append('/mnt/crawl/ryan/filter_anno/train/final_filter_train_3.json')
path_list.append('/mnt/crawl/ryan/filter_anno/train/final_filter_train_4.json')
path_list.append('/mnt/crawl/ryan/filter_anno/train/final_filter_train_5.json')
path_list.append('/mnt/crawl/ryan/filter_anno/train/final_filter_train_5_2.json')
path_list.append('/mnt/crawl/ryan/filter_anno/train/final_filter_train_6.json')
path_list.append('/mnt/crawl/ryan/filter_anno/train/final_filter_train_6_2.json')
path_list.append('/mnt/crawl/ryan/filter_anno/train/final_filter_train_7.json')
path_list.append('/mnt/crawl/ryan/filter_anno/train/final_filter_train_7_2.json')
path_list.append('/mnt/crawl/ryan/filter_anno/train/final_filter_train_8.json')
path_list.append('/mnt/crawl/ryan/filter_anno/train/final_filter_train_8_2.json')
path_list.append('/mnt/crawl/ryan/filter_anno/train/final_filter_train_9.json')
path_list.append('/mnt/crawl/ryan/filter_anno/train/final_filter_train_9_2.json')
path_list.append('/mnt/crawl/ryan/filter_anno/train/final_filter_train_10.json')
path_list.append('/mnt/crawl/ryan/filter_anno/train/final_filter_train_10_2.json')
path_list.append('/mnt/crawl/ryan/filter_anno/train/final_filter_train_11.json')
path_list.append('/mnt/crawl/ryan/filter_anno/train/final_filter_train_11_2.json')
path_list.append('/mnt/crawl/ryan/filter_anno/train/final_filter_train_12.json')
path_list.append('/mnt/crawl/ryan/filter_anno/train/final_filter_train_12_2.json')
path_list.append('/mnt/crawl/ryan/filter_anno/train/final_filter_train_13.json')
path_list.append('/mnt/crawl/ryan/filter_anno/train/final_filter_train_13_2.json')
path_list.append('/mnt/crawl/ryan/filter_anno/train/final_filter_train_14.json')
path_list.append('/mnt/crawl/ryan/filter_anno/train/final_filter_train_14_2.json')
path_list.append('/mnt/crawl/ryan/filter_anno/train/final_filter_train_15.json')
path_list.append('/mnt/crawl/ryan/filter_anno/train/final_filter_train_15_2.json')
"""
"""
path_list.append('/mnt/crawl/VQA_v2/v2_Annotations_Val_mscoco/ko_v2_mscoco_val2014_annotations.json')
path_list.append('/mnt/crawl/ryan/filter_anno/val/final_filter_val_1.json')
path_list.append('/mnt/crawl/ryan/filter_anno/val/final_filter_val_2.json')
path_list.append('/mnt/crawl/ryan/filter_anno/val/final_filter_val_3.json')
path_list.append('/mnt/crawl/ryan/filter_anno/val/final_filter_val_3_2.json')
path_list.append('/mnt/crawl/ryan/filter_anno/val/final_filter_val_4.json')
path_list.append('/mnt/crawl/ryan/filter_anno/val/final_filter_val_5.json')
path_list.append('/mnt/crawl/ryan/filter_anno/val/final_filter_val_6.json')
path_list.append('/mnt/crawl/ryan/filter_anno/val/final_filter_val_7.json')
path_list.append('/mnt/crawl/ryan/filter_anno/val/final_filter_val_8.json')
path_list.append('/mnt/crawl/ryan/filter_anno/val/final_filter_val_9.json')
"""
path_list.append('/mnt/crawl/VQA_v2/v2_Questions_Test_mscoco/ko_v2_OpenEnded_mscoco_test2015_questions.json')
path_list.append('/mnt/crawl/VQA_v2/v2_Questions_Test_mscoco/ko3940_v2_OpenEnded_mscoco_test2015_questions.json')
path_list.append('/mnt/crawl/VQA_v2/v2_Questions_Test_mscoco/ko4041_v2_OpenEnded_mscoco_test2015_questions.json')
path_list.append('/mnt/crawl/VQA_v2/v2_Questions_Test_mscoco/ko2_v2_OpenEnded_mscoco_test2015_questions.json')
path_list.append('/mnt/crawl/VQA_v2/v2_Questions_Test_mscoco/ko4243_v2_OpenEnded_mscoco_test2015_questions.json')
path_list.append('/mnt/crawl/VQA_v2/v2_Questions_Test_mscoco/ko4344_v2_OpenEnded_mscoco_test2015_questions.json')
path_list.append('/mnt/crawl/VQA_v2/v2_Questions_Test_mscoco/ko4344_2_v2_OpenEnded_mscoco_test2015_questions.json')
path_list.append('/mnt/crawl/VQA_v2/v2_Questions_Test_mscoco/ko44end_v2_OpenEnded_mscoco_test2015_questions.json')

data_list = []
for path in path_list:
    with open(path) as f:
        data = json.load(f)
        data_list.append(data)



text_list1 = data_list[0]['questions']
text_list2 = data_list[1]['questions']
for i in tqdm(range(390000, 400000)):
    text_list1[i] = text_list2[i]

text_list1 = data_list[0]['questions']
text_list2 = data_list[2]['questions']
for i in tqdm(range(400000, 410000)):
    text_list1[i] = text_list2[i]

text_list1 = data_list[0]['questions']
text_list2 = data_list[3]['questions']
for i in tqdm(range(410000, 420000)):
    text_list1[i] = text_list2[i]

text_list1 = data_list[0]['questions']
text_list2 = data_list[4]['questions']
for i in tqdm(range(420000, 430000)):
    text_list1[i] = text_list2[i]

text_list1 = data_list[0]['questions']
text_list2 = data_list[5]['questions']
for i in tqdm(range(430000, 435000)):
    text_list1[i] = text_list2[i]

text_list1 = data_list[0]['questions']
text_list2 = data_list[6]['questions']
for i in tqdm(range(435000, 440000)):
    text_list1[i] = text_list2[i]

text_list1 = data_list[0]['questions']
text_list2 = data_list[7]['questions']
end = len(text_list1)
for i in tqdm(range(440000, end)):
    text_list1[i] = text_list2[i]
"""
text_list1 = data_list[0]['annotations']
text_list2 = data_list[8]['annotations']
for i in tqdm(range(180000, 190000)):
    text_list1[i] = text_list2[i]

text_list1 = data_list[0]['annotations']
text_list2 = data_list[9]['annotations']
for i in tqdm(range(190000, 200000)):
    text_list1[i] = text_list2[i]

text_list1 = data_list[0]['annotations']
text_list2 = data_list[10]['annotations']
for i in tqdm(range(200000, 214354)):
    text_list1[i] = text_list2[i]

text_list1 = data_list[0]['annotations']
text_list2 = data_list[11]['annotations']
for i in tqdm(range(210000, 225000)):
    text_list1[i] = text_list2[i]

text_list1 = data_list[0]['annotations']
text_list2 = data_list[12]['annotations']
for i in tqdm(range(225000, 240000)):
    text_list1[i] = text_list2[i]

text_list1 = data_list[0]['annotations']
text_list2 = data_list[13]['annotations']
for i in tqdm(range(240000, 255000)):
    text_list1[i] = text_list2[i]

text_list1 = data_list[0]['annotations']
text_list2 = data_list[14]['annotations']
for i in tqdm(range(255000, 270000)):
    text_list1[i] = text_list2[i]

text_list1 = data_list[0]['annotations']
text_list2 = data_list[15]['annotations']
for i in tqdm(range(270000, 285000)):
    text_list1[i] = text_list2[i]

text_list1 = data_list[0]['annotations']
text_list2 = data_list[16]['annotations']
for i in tqdm(range(285000, 300000)):
    text_list1[i] = text_list2[i]

text_list1 = data_list[0]['annotations']
text_list2 = data_list[17]['annotations']
for i in tqdm(range(300000, 315000)):
    text_list1[i] = text_list2[i]

text_list1 = data_list[0]['annotations']
text_list2 = data_list[18]['annotations']
for i in tqdm(range(315000, 330000)):
    text_list1[i] = text_list2[i]

text_list1 = data_list[0]['annotations']
text_list2 = data_list[19]['annotations']
for i in tqdm(range(330000, 345000)):
    text_list1[i] = text_list2[i]

text_list1 = data_list[0]['annotations']
text_list2 = data_list[20]['annotations']
for i in tqdm(range(345000, 360000)):
    text_list1[i] = text_list2[i]

text_list1 = data_list[0]['annotations']
text_list2 = data_list[21]['annotations']
for i in tqdm(range(360000, 375000)):
    text_list1[i] = text_list2[i]

text_list1 = data_list[0]['annotations']
text_list2 = data_list[22]['annotations']
for i in tqdm(range(375000, 390000)):
    text_list1[i] = text_list2[i]

text_list1 = data_list[0]['annotations']
text_list2 = data_list[23]['annotations']
for i in tqdm(range(390000, 400000)):
    text_list1[i] = text_list2[i]

text_list1 = data_list[0]['annotations']
text_list2 = data_list[24]['annotations']
for i in tqdm(range(400000, 410000)):
    text_list1[i] = text_list2[i]

text_list1 = data_list[0]['annotations']
text_list2 = data_list[25]['annotations']
for i in tqdm(range(410000, 428000)):
    text_list1[i] = text_list2[i]

text_list1 = data_list[0]['annotations']
text_list2 = data_list[26]['annotations']
for i in tqdm(range(428000, 443757)):
    text_list1[i] = text_list2[i]
"""
#path = '/mnt/crawl/VQA_v2/v2_Annotations_Train_mscoco/ko_v2_mscoco_train2014_annotations.json'
#path = '/mnt/crawl/VQA_v2/v2_Annotations_Val_mscoco/ko_v2_mscoco_val2014_annotations.json'
path = '/mnt/crawl/VQA_v2/v2_Questions_Test_mscoco/ko_v2_OpenEnded_mscoco_test2015_questions.json'
with open(path, 'w', encoding='utf-8') as make_file:
    json.dump(data_list[0], make_file, indent=4)

print('end')