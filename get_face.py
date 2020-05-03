# -*- coding: utf-8 -*-
"""
Created on Sat May  2 18:33:15 2020

@author: shiyl
"""

import os
import json
import urllib.request

code_path = 'D:/git/stacked-autoencoder-pytorch/'
data_path = 'D:/git/imgs/'

if not os.path.exists(data_path):
    os.mkdir(data_path)


n_photos = 100000  # 3 #
# cmd
# curl --header "Authorization: API-Key 51iKJY_txZKmCYwN7ftChQ" "https://api.generated.photos/api/v1/faces?per_page=100000” >> D:\git\stacked-autoencoder-pytorch\face_urls.json

with open(os.path.join(code_path + 'face_urls.json'),'r') as f:
    meta = json.load(f)
    
urls = [face['urls'][1].get('64') for face in meta['faces']]

for i,url in enumerate(urls):
    urllib.request.urlretrieve(url, data_path + f'{i:05}.jpg')  



    
    
    
    