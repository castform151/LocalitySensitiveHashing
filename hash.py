import pandas as pd
import os

df = pd.read_csv('dataset/20newsgroup_preprocessed.csv', sep=';')

fo = open('dataset/originals.txt', 'w')
fp = open('dataset/pre_processed.txt', 'w')
# print(fp.read())
# query_path = "Query_Doc/" + os.listdir("Query_Doc/")[-1]
# print(query_path)

k = 0
for i in df.itertuples():
    if k > 96:
        break
    fo.writelines(f'{i[2]}\nEND OF DOCUMENT\n')
    fp.writelines(f'{i[3]}\n####\n')
    k += 1
