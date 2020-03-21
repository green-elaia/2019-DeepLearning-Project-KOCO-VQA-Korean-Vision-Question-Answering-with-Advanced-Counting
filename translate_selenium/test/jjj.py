import json

#path = 'C:\\Users\\green\\Desktop\\ko_v2_mscoco_val2014_annotations.json'
path = '/mnt/crawl/kordata/ko_v2_mscoco_val2014_annotations.json'

with open(path) as json_file:
    json_data = json.load(json_file)

for i in range(0, 5):
    print(json_data['annotations'][i])
#print(json_data)