import json
import pandas as pd

bbox_json = 'sub.json'
sub_csv = 'final_sub.csv'
sub_json = 'final_99.json'


with open(bbox_json, 'r') as fp:
    info = json.load(fp)
sub_df = pd.read_csv(sub_csv)

sub = {}
labels = ['1', '3', '6', '11', '12']
for label in labels:
    sub[label] = {}

for id,label in zip(sub_df['id'],sub_df['label']):
    source, frame = id.split('/')
    item = {frame:info[source][frame]}
    if source not in sub[str(label)].keys():
        sub[str(label)][source] = {}
    sub[str(label)][source].update(item)

# print(sub)

with open(sub_json, 'w') as fp:
    json.dump(sub,fp)