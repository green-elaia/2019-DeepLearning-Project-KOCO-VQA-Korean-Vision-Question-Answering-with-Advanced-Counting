#import pandas as pd
#import numpy as np
import json
from konlpy.tag import Mecab

m = Mecab()

with open('/mnt/crawl/ryan/count_demo/vqa-v2/data_reduce_v2/v2_OpenEnded_mscoco_train2014_questions.json', 'r') as f:
# with open('/mnt/crawl/ryan/count_demo/vqa-v2/data/v2_OpenEnded_mscoco_val2014_questions.json', 'r') as f:
    json_data = json.load(f)

temp = ''
new_temp =''
for i in range(len(json_data['questions'])):
     temp = json_data['questions'][i]['question']
     new_temp = " ".join(m.morphs(temp))
     json_data['questions'][i]['question'] = new_temp

with open('/mnt/crawl/ryan/count_demo/vqa-v2/data_reduce_v2/v2_OpenEnded_mscoco_train2014_questions.json', 'w', encoding='utf-8') as f:
# with open('/mnt/crawl/ryan/count_demo/vqa-v2/data/v2_OpenEnded_mscoco_val2014_questions.json', 'w', encoding='utf-8') as f:
    json.dump(json_data,f)