import pandas as pd
import csv

# read = pd.read_csv("C:\Users\Ryan_SYLee\Desktop\Program\vqa\karpathy_val_resnet101_faster_rcnn_genome", sep='\t')

dataset = pd.read_csv("C:\\Users\\Ryan_SYLee\\Desktop\\Program\\vqa\\karpathy_val_resnet101_faster_rcnn_genome.tsv", delimiter='\t')

# rdr = csv.reader(dataset, delimiter='\t')
r = list(dataset)
print(r[0:10])