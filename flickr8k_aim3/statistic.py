import json
from tqdm import tqdm
import os

dataset = json.load(open('./flickr8k_aim3/dataset_flickr8k.json', 'r', encoding='utf-8'))
images = dataset['images']
statistic = {'train':{'image':0, 'sent':0}, 'val':{'image':0, 'sent':0}, 'test':{'image':0, 'sent':0}}
for i in images:
    statistic[i['split']]['image']+=1
    statistic[i['split']]['sent']+=len(i['sentences'])
    image_path = './flickr8k_aim3/images/'+i['filename']
    assert os.path.exists(image_path)
    if i['filename']=='1000268201_693b08cb0e.jpg':
        for j in i['sentences']:
            print(j['raw'])
print(statistic)
    
