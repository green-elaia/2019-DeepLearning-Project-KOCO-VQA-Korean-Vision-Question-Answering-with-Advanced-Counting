import json


#path = 'C:\\Users\\green\\Desktop\\ko_v2_mscoco_train2014_annotations.json'
# 247 9 /
#path = 'C:\\Users\\green\\Desktop\\ko_v2_mscoco_train2014_annotations.json'
#path = 'C:\\Users\\green\\Desktop\\ko3_v2_mscoco_train2014_annotations.json'
#path = 'C:\\Users\\green\\Desktop\\ko4_v2_mscoco_train2014_annotations.json'
path = 'C:\\Users\\green\\Desktop\\ko_v2_mscoco_val2014_annotations.json'

with open(path) as json_file:
    json_data = json.load(json_file)


# print(json_data['annotations'][8000]['answers'])
# print(json_data['annotations'][13000]['answers'])
# print(json_data['annotations'][120000]['answers'])
# print(json_data['annotations'][135000]['answers'])
# print(json_data['annotations'][145000]['answers'])
# print(json_data['annotations'][150000]['answers'])
# print(json_data['annotations'][180000]['answers'])
# print(json_data['annotations'][201000]['answers'])
# print(json_data['annotations'][134000]['answers'])
# print(json_data['annotations'][270000]['answers'])
# print(json_data['annotations'][290000]['answers'])
# print(json_data['annotations'][310000]['answers'])
# print(json_data['annotations'][330000]['answers'])
# print(json_data['annotations'][340000]['answers'])

# for i in range(0, 10):
#     print('index : ', i ,"  " ,json_data['annotations'][i]['answers'])

#print(len(json_data['annotations']))
print(json_data)