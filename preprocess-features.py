import sys
import argparse
import base64
import os
import csv
import itertools

csv.field_size_limit(sys.maxsize)

import h5py
import torch.utils.data
import numpy as np
from tqdm import tqdm

import config
import data
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

    features_shape = (
        # 82783 + 40504 if not args.test else 81434,  # number of images in trainval or in test
        2,
        config.output_features,
        config.output_size,
    )
    boxes_shape = (
        features_shape[0],
        4,
        config.output_size,
    )

    if not args.test:
        path = config.preprocessed_trainval_path
    else:
        path = config.preprocessed_test_path
    with h5py.File(path, 'w') as fd:
        # features = fd.create_dataset('features', shape=features_shape, dtype='float32')
        # boxes = fd.create_dataset('boxes', shape=boxes_shape, dtype='float32')
        # coco_ids = fd.create_dataset('ids', shape=(features_shape[0],), dtype='int32')
        # widths = fd.create_dataset('widths', shape=(features_shape[0],), dtype='int32')
        # heights = fd.create_dataset('heights', shape=(features_shape[0],), dtype='int32')
        features = fd.create_dataset('features', shape=features_shape, dtype='float32')
        boxes = fd.create_dataset('boxes', shape=boxes_shape, dtype='float32')
        coco_ids = fd.create_dataset('ids', shape=(features_shape[0],), dtype='int32')
        widths = fd.create_dataset('widths', shape=(features_shape[0],), dtype='int32')
        heights = fd.create_dataset('heights', shape=(features_shape[0],), dtype='int32')

        readers = []
        if not args.test:
            path = config.bottom_up_trainval_path
        else:
            path = config.bottom_up_test_path
        for filename in os.listdir(path):
            if not '.tsv' in filename:
                continue
            full_filename = os.path.join(path, filename)
            fd = open(full_filename, 'r')
            reader = csv.DictReader(fd, delimiter='\t', fieldnames=FIELDNAMES)
            readers.append(reader)

        # id_num = [194194, 453003]
        # j = 0
        # for i, item in enumerate(tqdm(reader, total=features_shape[0])):
        #     if int(item['image_id']) in id_num:
        #         coco_ids[j] = int(item['image_id'])
        #         widths[j] = int(item['image_w'])
        #         heights[j] = int(item['image_h'])
        #
        #         print("id : ", coco_ids[j], " w : ", widths[j], " h : ", heights[j])
        #
        #         buf = base64.decodestring(item['features'].encode('utf8'))
        #         array = np.frombuffer(buf, dtype='float32')
        #         array = array.reshape((-1, config.output_features)).transpose()
        #         features[j, :, :array.shape[1]] = array
        #
        #         buf = base64.decodestring(item['boxes'].encode('utf8'))
        #         array = np.frombuffer(buf, dtype='float32')
        #         array = array.reshape((-1, 4)).transpose()
        #         boxes[j, :, :array.shape[1]] = array
        #         j += 1
        #     if j == len(id_num): break
        id_num = [524881, 194194]
        j = 0
        while j != len(id_num):
            print("j : ", j)
            if j == len(id_num): break
            for i, item in enumerate(tqdm(reader, total=82783 + 40504)):
                if int(item['image_id']) == id_num[j]:
                # if int(item['image_id']) == id_num:
                    print(int(item['image_id']))
                    coco_ids[j] = int(item['image_id'])
                    widths[j] = int(item['image_w'])
                    heights[j] = int(item['image_h'])

                    print("id : ", coco_ids[j], " w : ", widths[j], " h : ", heights[j])

                    buf = base64.decodestring(item['features'].encode('utf8'))
                    array = np.frombuffer(buf, dtype='float32')
                    array = array.reshape((-1, config.output_features)).transpose()
                    features[j, :, :array.shape[1]] = array

                    buf = base64.decodestring(item['boxes'].encode('utf8'))
                    array = np.frombuffer(buf, dtype='float32')
                    array = array.reshape((-1, 4)).transpose()
                    boxes[j, :, :array.shape[1]] = array
                    j += 1
                    break


    with h5py.File('sample36.h5', 'r') as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data = list(f[a_group_key])
        # print(data)
        print(f['features'][0].shape)

if __name__ == '__main__':
    main()
