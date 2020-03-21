import json

#path = 'C:\\Users\\green\\Desktop\\ko_v2_OpenEnded_mscoco_train2014_questions.json'
#path = 'C:\\Users\\green\\Desktop\\ko_v2_OpenEnded_mscoco_val2014_questions.json'
#path = 'C:\\Users\\green\\Desktop\\ko_v2_OpenEnded_mscoco_test-dev2015_questions.json'
path = 'C:\\Users\\green\\Desktop\\ko_v2_OpenEnded_mscoco_test2015_questions.json'
#path = 'C:\\Users\\green\\Desktop\\v2_mscoco_train2014_complementary_pairs.json'

with open(path) as json_file:
    json_data = json.load(json_file)


#print(json_data['questions'][9000]['question'])
#print(json_data['questions'][10000]['question'])
# print(json_data['questions'][81881]['question'])
#print(json_data['questions'][88000]['question'])
print(json_data['questions'][10]['question'])
print(json_data['questions'][347629]['question'])
#print(json_data['questions'][166990]['question'])
# print(json_data['questions'][70000]['question'])
# print(json_data['questions'][80000]['question'])
# print(json_data['questions'][90000]['question'])
# print(json_data['questions'][100000]['question'])
#print(json_data['questions'][81921]['question'])

print(len(json_data['questions']))