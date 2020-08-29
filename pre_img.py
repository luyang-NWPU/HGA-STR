import torch
from torch.autograd import Variable
import tools.utils as utils
import tools.dataset as dataset
from PIL import Image
from collections import OrderedDict
import numpy as np
import cv2
from models.model import MODEL
from models.transformer.Constants import UNK,PAD,BOS,EOS,PAD_WORD,UNK_WORD,BOS_WORD,EOS_WORD
import os,sys,pdb

model_path = sys.argv[1]
img_path = sys.argv[2]
img_name = img_path.split('.')[0].split('/')[-1]

alphabet = '0 1 2 3 4 5 6 7 8 9 a b c d e f g h i j k l m n o p q r s t u v w x y z A B C D E F G H I J K L M N O P Q R S T U V W X Y Z ! " \' # $ % & ( ) * + , - . / : ; < = > ? @ [ \\ ] _ ` ~'
n_bm = 5
imgW = 160
imgH = 48
nclass = len(alphabet.split(' '))
MODEL = MODEL(n_bm, nclass)


if torch.cuda.is_available():
    MODEL = MODEL.cuda()

print('loading pretrained model from %s' % model_path)
state_dict = torch.load(model_path)
MODEL_state_dict_rename = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "") # remove `module.`
    MODEL_state_dict_rename[name] = v
MODEL.load_state_dict(MODEL_state_dict_rename)

for p in MODEL.parameters():
    p.requires_grad = False
MODEL.eval()

converter = utils.strLabelConverterForAttention(alphabet, ' ')
transformer = dataset.resizeNormalize((imgW, imgH))
image = Image.open(img_path).convert('RGB')
image = transformer(image)

if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)
text = torch.LongTensor(1 * 5)
length = torch.IntTensor(1)
text = Variable(text)
length = Variable(length)

max_iter = 35
t, l = converter.encode('0'*max_iter)
utils.loadData(text, t)
utils.loadData(length, l)

preds = MODEL(image, length, text, text, test=True, cpu_texts='')[0]
pred = converter.decode(preds.data, length.data + 5)
pred = pred.split(' ')[0]
print('################# Answer: '+pred)
