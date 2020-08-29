from __future__ import print_function
import argparse
import random
import math
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import os,pdb
import tools.utils as utils
import tools.dataset as dataset
import time
from collections import OrderedDict
from models.model import MODEL
from tools.utils import adjust_lr_exp, str2bool
from PIL import Image
from tools.utils import addEOS


parser = argparse.ArgumentParser()
parser.add_argument('--train_1', required=True, help='path to dataset')
parser.add_argument('--train_2', required=True, help='path to dataset')
parser.add_argument('--test_1', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imgH', type=int, default=48, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=160, help='the width of the input image to network')
parser.add_argument('--niter', type=int, default=5, help='number of epochs to train for')
parser.add_argument('--dec_layer', type=int, default=1, help='Decoder Block layer number.')
parser.add_argument('--val_start_epoch', type=float, default=0.0, help='val is Time-consuming, only start val at this epoch')
parser.add_argument('--lr', type=float, default=1.0, help='learning rate for Critic, default=1.0')
parser.add_argument('--LR', type=str2bool, default=False, help='Char form left to right, and from right to left.')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--MODEL', default='', help="path to model (to continue training)")
parser.add_argument('--n_bm', type=int, default=5, help='number of n_bm')
parser.add_argument('--alphabet', type=str, default='0 1 2 3 4 5 6 7 8 9 a b c d e f g h i j k l m n o p q r s t u v w x y z A B C D E F G H I J K L M N O P Q R S T U V W X Y Z ! " \' # $ % & ( ) * + , - . / : ; < = > ? @ [ \\ ] _ ` ~')
parser.add_argument('--alphabet1', type=str, default='0 1 2 3 4 5 6 7 8 9 a b c d e f g h i j k l m n o p q r s t u v w x y z A B C D E F G H I J K L M N O P Q R S T U V W X Y Z')
parser.add_argument('--alphabet2', type=str, default='a b c d e f g h i j k l m n o p q r s t u v w x y z A B C D E F G H I J K L M N O P Q R S T U V W X Y Z')
parser.add_argument('--sep', type=str, default=' ')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--valInterval', type=int, default=10000, help='Interval to be displayed')
opt = parser.parse_args()
print(opt)

batchSize1 = int(opt.batchSize*0.5)
batchSize2 = opt.batchSize - batchSize1

if opt.experiment is None:
    opt.experiment = 'output'
if not os.path.exists(opt.experiment):
    os.system('mkdir {0}'.format(opt.experiment))

opt.manualSeed = 0 #random.randint(0, 10000) 
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Train data
train_dataset_1 = dataset.lmdbDataset( root=opt.train_1, 
    transform=dataset.resizeNormalize((opt.imgW, opt.imgH)))
assert train_dataset_1

train_dataset_2 = dataset.lmdbDataset( root=opt.train_2, 
    transform=dataset.resizeNormalize((opt.imgW, opt.imgH)))
assert train_dataset_2

# Train data
print('batchSize: %d and %d' % (batchSize1, batchSize2))
train_loader1 = torch.utils.data.DataLoader(
    train_dataset_1, batch_size=batchSize1,
    shuffle=False, sampler=dataset.randomSequentialSampler(train_dataset_1, batchSize1),
    num_workers=int(opt.workers))
train_loader2 = torch.utils.data.DataLoader(
    train_dataset_2, batch_size=batchSize2,
    shuffle=False, sampler=dataset.randomSequentialSampler(train_dataset_2, batchSize2),
    num_workers=int(opt.workers))    
    

test_dataset1 = dataset.lmdbDataset( test=True,root=opt.test_1, 
    transform=dataset.resizeNormalize((opt.imgW, opt.imgH)))

nclass = len(opt.alphabet.split(opt.sep))
converter = utils.strLabelConverterForAttention(opt.alphabet, opt.sep)
criterion = torch.nn.CrossEntropyLoss()

MODEL = MODEL(opt.n_bm, nclass, dec_layer=opt.dec_layer, LR=opt.LR )

# print("MODEL have {} paramerters in total".format(sum(x.numel() for x in MODEL.parameters())))

if opt.MODEL != '':
    print('loading pretrained model from %s' % opt.MODEL)
    state_dict = torch.load(opt.MODEL)
    MODEL_state_dict_rename = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") # remove `module.`
        MODEL_state_dict_rename[name] = v
    MODEL.load_state_dict(MODEL_state_dict_rename, strict=True)

image = torch.FloatTensor(opt.batchSize, 1, opt.imgH, opt.imgW)
text1_ori = torch.LongTensor(opt.batchSize * 5)
text2_ori = torch.LongTensor(opt.batchSize * 5)
length_ori = torch.IntTensor(opt.batchSize)

if opt.cuda:
    MODEL.cuda()
    MODEL = torch.nn.DataParallel(MODEL, device_ids=range(opt.ngpu))
    text1_ori = text1_ori.cuda()
    text2_ori = text2_ori.cuda()
    criterion = criterion.cuda()
    length_ori = length_ori.cuda()


image = Variable(image)
length_ori = Variable(length_ori)
text1_ori = Variable(text1_ori)
text2_ori = Variable(text2_ori)

# loss averager
loss_avg = utils.averager()
loss_pred_avg1 = utils.averager()
loss_pred_avg2 = utils.averager()
optimizer = optim.Adadelta(filter(lambda p: p.requires_grad,MODEL.parameters()), lr=opt.lr)
    
toPIL = transforms.ToPILImage()
toTensor = transforms.ToTensor()
def val_beam(dataset ,max_iter=9999):
    rotate90 = dataset.ifRotate90
    
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=opt.batchSize, num_workers=1) # opt.batchSize
    val_iter = iter(data_loader)
    max_iter = min(max_iter, len(data_loader))
    n_correct = 0
    n_total = 0

    for i in range(max_iter):
        data = val_iter.next()
        ori_cpu_images = data[0]
        flag_rotate90 = data[2]
        cpu_texts1 = data[1]
        cpu_texts2 = data[3]
    
        t1, l1 = converter.encode(cpu_texts1, scanned=True)
        t2, l2 = converter.encode(cpu_texts2, scanned=True)
        utils.loadData(text1_ori, t1)
        utils.loadData(text2_ori, t2)
        utils.loadData(length_ori, l1)
        All_preds_add5EOS1 = []
        All_scores1 = []
        All_preds_add5EOS2 = []
        All_scores2 = []
        
        
        cpu_images = ori_cpu_images
           
        utils.loadData(image, cpu_images)
        if opt.LR:
           local_preds1, local_scores1, local_preds2, local_scores2 = MODEL(image, length_ori, text1_ori, text2_ori, test=True, cpu_texts=cpu_texts1)
           All_preds_add5EOS1.append(local_preds1)
           All_preds_add5EOS2.append(local_preds2)
           All_scores1.append(local_scores1)
           All_scores2.append(local_scores2)
        else:
           local_preds1, local_scores1 = MODEL(image, length_ori, text1_ori, None, test=True, cpu_texts=cpu_texts1)
           All_preds_add5EOS1.append(local_preds1)
           All_scores1.append(local_scores1)
        
        length_label = (length_ori-1).data.cpu().numpy()
                        
            
        # %%% Left/Right Rotate %%%   
        if rotate90==True:
           PIL_imgs = [toPIL(ori_cpu_images[i].div(2).sub(-0.5)) for i in range(ori_cpu_images.shape[0])]
           PIL_imgs_left90 = [PIL_imgs[i].transpose(Image.ROTATE_90).resize((opt.imgW,opt.imgH),Image.BILINEAR) if flag_rotate90[i] else PIL_imgs[i] for i in range(ori_cpu_images.shape[0])] 
           PIL_imgs_right90 = [PIL_imgs[i].transpose(Image.ROTATE_270).resize((opt.imgW,opt.imgH),Image.BILINEAR) if flag_rotate90[i] else PIL_imgs[i] for i in range(ori_cpu_images.shape[0])] 
           imgs_Tensor_left90 = [toTensor(PIL_imgs_left90[i]) for i in range(ori_cpu_images.shape[0])]
           imgs_Tensor_right90 = [toTensor(PIL_imgs_right90[i]) for i in range(ori_cpu_images.shape[0])]
           
           # Left
           cpu_images = torch.stack(imgs_Tensor_left90)
           cpu_images.sub_(0.5).div_(0.5)
           utils.loadData(image, cpu_images)
           if opt.LR:
              local_preds1, local_scores1, local_preds2, local_scores2, _ = MODEL(image, length_ori, text1_ori, text2_ori, test=True, cpu_texts=cpu_texts1)
              All_preds_add5EOS1.append(local_preds1)
              All_preds_add5EOS2.append(local_preds2)
              All_scores1.append(local_scores1)
              All_scores2.append(local_scores2)
           else:
              local_preds1, local_scores1, _ = MODEL(image, length_ori, text1_ori, None, test=True, cpu_texts=cpu_texts1)
              All_preds_add5EOS1.append(local_preds1)
              All_scores1.append(local_scores1)
               
           # Right
           cpu_images = torch.stack(imgs_Tensor_right90)
           cpu_images.sub_(0.5).div_(0.5)
           utils.loadData(image, cpu_images)
           if opt.LR:
              local_preds1, local_scores1, local_preds2, local_scores2, _ = MODEL(image, length_ori, text1_ori, text2_ori, test=True, cpu_texts=cpu_texts1)
              All_preds_add5EOS1.append(local_preds1)
              All_preds_add5EOS2.append(local_preds2)
              All_scores1.append(local_scores1)
              All_scores2.append(local_scores2)
           else:
              local_preds1, local_scores1, _ = MODEL(image, length_ori, text1_ori, None, test=True, cpu_texts=cpu_texts1)
              All_preds_add5EOS1.append(local_preds1)
              All_scores1.append(local_scores1)
        
        # Start to decode
        preds_add5EOS1 = []
        preds_score1 = []
        for j in range(cpu_images.size(0)):
            text_begin = 0 if j == 0 else (length_ori.data[:j].sum()+j*5)
            max_score = -99999
            max_index = 0
            for index in range(len(All_scores1)):
                local_score = All_scores1[index][j]
                if local_score > max_score:
                   max_score = local_score
                   max_index = index
            preds_add5EOS1.extend(All_preds_add5EOS1[max_index][text_begin:text_begin+int(length_ori[j].data)+5])
            preds_score1.append(max_score)
        preds_add5EOS1 = torch.stack(preds_add5EOS1)
        sim_preds_add5eos1 = converter.decode(preds_add5EOS1.data, length_ori.data + 5)
        
        if opt.LR:
           preds_add5EOS2 = []
           preds_score2 = []
           for j in range(cpu_images.size(0)):
               text_begin = 0 if j == 0 else (length_ori.data[:j].sum()+j*5)
               max_score = -99999
               max_index = 0
               for index in range(len(All_scores2)):
                   local_score = All_scores2[index][j]
                   if local_score > max_score:
                      max_score = local_score
                      max_index = index
               preds_add5EOS2.extend(All_preds_add5EOS2[max_index][text_begin:text_begin+int(length_ori[j].data)+5])
               preds_score2.append(max_score)
           preds_add5EOS2 = torch.stack(preds_add5EOS2)
           sim_preds_add5eos2 = converter.decode(preds_add5EOS2.data, length_ori.data + 5)
        
        if opt.LR:
           batch_index = 0        
           for pred1, target1, pred2, target2 in zip(sim_preds_add5eos1, cpu_texts1, sim_preds_add5eos2, cpu_texts2):
               if preds_score1[batch_index] > preds_score2[batch_index]:
                  pred = pred1
                  target = target1
               else:
                  pred = pred2
                  target = target2
                  
               pred = pred.split(opt.sep)[0]+opt.sep
               test_alphabet = dataset.test_alphabet.split(opt.sep)
               pred = ''.join(pred[i].lower() if pred[i].lower() in test_alphabet else '' for i in range(len(pred)))
               target = ''.join(target[i].lower() if target[i].lower() in test_alphabet else '' for i in range(len(target)))
               
               if pred.lower() == target.lower():
                   n_correct += 1
               n_total += 1
               batch_index += 1  
        else:
           for pred, target in zip(sim_preds_add5eos1, cpu_texts1):
               pred = pred.split(opt.sep)[0]+opt.sep
               test_alphabet = dataset.test_alphabet.split(opt.sep)
               pred = ''.join(pred[i].lower() if pred[i].lower() in test_alphabet else '' for i in range(len(pred)))
               target = ''.join(target[i].lower() if target[i].lower() in test_alphabet else '' for i in range(len(target)))
               
               if pred.lower() == target.lower():
                   n_correct += 1
               n_total += 1
            
    accuracy = n_correct / float(n_total)
    
    dataset_name = dataset.root.split('/')[-1]
    print(dataset_name+' ACCURACY -----> %.1f%%, ' % (accuracy*100.0))
    return accuracy
  

train_PredNum_correct = 0
train_ADDEOS_correct = []
def trainBatch():
    data1 = train_iter1.next()
    data2 = train_iter2.next()
    cpu_images = torch.cat((data1[0],data2[0]),0)
    cpu_texts1 = data1[1] + data2[1]
    cpu_texts2 = data1[3] + data2[3]
        
    utils.loadData(image, cpu_images)
    t1, l1 = converter.encode(cpu_texts1, scanned=True)
    utils.loadData(text1_ori, t1)
    utils.loadData(length_ori, l1)
    t2, l2 = converter.encode(cpu_texts2, scanned=True)
    utils.loadData(text2_ori, t2)
            
    N = len(cpu_texts1)
    if opt.LR is True:
       preds1, preds2 = MODEL(image, length_ori, text1_ori, text2_ori, cpu_texts=cpu_texts1)
       
       text1_new = text1_ori
       text2_new = text2_ori
       
       cost_pred1 = criterion(preds1, text1_new) /2.0
       cost_pred2 = criterion(preds2, text2_new) /2.0
       loss_pred_avg1.add(cost_pred1)
       loss_pred_avg2.add(cost_pred2)
       
       cost = cost_pred1 + cost_pred2
    else:
       preds1 = MODEL(image, length_ori, ngpu_text1, None, cpu_texts=cpu_texts1)
       
       text1_new = text1_ori
       
       cost_pred1 = criterion(preds1, text1_new)
       loss_pred_avg1.add(cost_pred1)

       cost = cost_pred1 
          
    loss_avg.add(cost)
    MODEL.zero_grad()
    cost.backward()
    optimizer.step()
    
    return cost


t0 = time.time()
training_iters = min(len(train_loader1), len(train_loader2))-opt.workers
print('ep iters: ',len(train_loader1), len(train_loader2))
total_iter = training_iters*opt.niter
for epoch in range(opt.niter):
    
    train_iter1 = iter(train_loader1)
    train_iter2 = iter(train_loader2)
    i = 0
    while i < training_iters:
        
        adjust_lr_exp(optimizer, opt.lr, i+training_iters*epoch, training_iters*opt.niter)
        
        if i % opt.valInterval == 0 and epoch+float(i)/training_iters>=opt.val_start_epoch:
            for p in MODEL.parameters():
                p.requires_grad = False
            MODEL.eval()
            print('=============== Start val (beam size:'+str(opt.n_bm)+') ===============')
            
            acc = val_beam(test_dataset1)
            if acc>=0.945:
               torch.save(MODEL.state_dict(), 'output/acc_%.3f.pth' %(acc))
               
        for p in MODEL.parameters():
            p.requires_grad = True
        MODEL.train()
        
        cost = trainBatch()
        
        if i % opt.displayInterval == 0 and i!=0:
            t1 = time.time()            
            ADDEOS_acc = np.mean(train_ADDEOS_correct)
            if opt.LR is True:
                 print ('Epoch: %d/%d; iter: %d/%d; Pred1Loss: %.2f; Pred2Loss: %.2f; TotalLoss: %.2f; time: %.2f s;' % \
                       (epoch, opt.niter, i, training_iters, loss_pred_avg1.val(), loss_pred_avg2.val(),  loss_avg.val(), t1-t0)),
            else:    
                 print ('Epoch: %d/%d; iter: %d/%d; PredLoss: %.2f;  TotalLoss: %.2f; time: %.2f s;' % \
                       (epoch, opt.niter, i, training_iters, loss_pred_avg1.val(), loss_avg.val(), t1-t0)),
            loss_pred_avg1.reset()
            loss_pred_avg2.reset()
            train_PredNum_correct = 0
            train_ADDEOS_correct = []
            loss_avg.reset()
            t0 = time.time()
            torch.save(MODEL.state_dict(), '{0}/latest.pth'.format(opt.experiment))
        i += 1
