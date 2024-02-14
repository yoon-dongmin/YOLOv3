import glob
import pprint as pp
import time
from re import T
from urllib.parse import parse_qs, urlparse
# import clip
import numpy as np
import requests
import torch
from PIL import Image, ImageFont
import io
import requests
from io import BytesIO
import warnings
import uuid
import tempfile
import os
import cv2
import torch
import glob
import sys
import clip
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import ipyplot
#from IPython.display import Image, display #yolo weight 부를 때 필요


url = 'https://productplacementblog.com/wp-content/uploads/2018/02/Jack-Daniels-and-Al-Pacino-in-Scent-of-a-Woman-12-800x500.jpg' #@param {type:"string"}
response = requests.get(url)
img1 = Image.open(BytesIO(response.content)).convert("RGB")
# imshow(np.asarray(img1))
# plt.show()

##Model load##
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

##Inference##
results = model(img1)  # includes NMS
print(results)
#한번만 실행
# image_path = "./content/"
# os.mkdir(image_path) 
###
dirpath = "./content/image"
results.save(dirpath)
#dirpath = tempfile.mkdtemp() #임시 디렉토리에 저장
results.crop(save_dir=dirpath) 
# glob을 사용할 때 '**'은 하위 디렉토리 모두 탐색
# '*'는 임의 길이의 모든 문자열을 의미
path= dirpath +'/crops/**/*.jpg'
print(path)

l = []
#keyList = list(range(len(txtfiles)))
for filename in glob.glob(path):
  foo = Image.open(filename).convert('RGB')
  print(foo)
  #resized_image = foo.resize((250,250))
  l.append(foo) 


###Using CLIP###
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

images = torch.stack([preprocess(im) for im in l]).to(device) # torch.stack으로 새로운 차원으로 주어진 텐서들을 붙입니다
with torch.no_grad():
  image_features = model.encode_image(images)
  image_features /= image_features.norm(dim=-1, keepdim=True)

image_features.cpu().numpy()
image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()

images = [preprocess(im) for im in l]
image_input = torch.tensor(np.stack(images)).cuda()
image_input -= image_mean[:, None, None]
image_input /= image_std[:, None, None]
with torch.no_grad():
    image_features = model.encode_image(image_input).float()
image_features /= image_features.norm(dim=-1, keepdim=True)

def similarity_top(similarity_list,N):
  results = zip(range(len(similarity_list)), similarity_list)
  results = sorted(results, key=lambda x: x[1],reverse= True)
  top_images = []
  scores=[]
  for index,score in results[:N]:
    scores.append(score)
    top_images.append(l[index])
  return scores,top_images

###Query###
search_query = "Give me the cup"

###Crop###
#@title Crop
with torch.no_grad():
    # Encode and normalize the description using CLIP
    text_encoded = model.encode_text(clip.tokenize(search_query).to(device))
    text_encoded /= text_encoded.norm(dim=-1, keepdim=True)

# Retrieve the description vector and the photo vectors

similarity = text_encoded.cpu().numpy() @ image_features.cpu().numpy().T
similarity = similarity[0]
scores,imgs= similarity_top(similarity,N=1)
print(scores,imgs,123312)
