import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import sampler
import lmdb
import six
import sys
from PIL import Image
import numpy as np

class lmdbDataset(Dataset):

    def __init__(self, train_alphabet=None, test_alphabet=None, test=False, root=None, transform=None, ifRotate90=False):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'))
            self.nSamples = nSamples

        self.transform = transform
        if train_alphabet is None:
           self.train_alphabet = '0 1 2 3 4 5 6 7 8 9 a b c d e f g h i j k l m n o p q r s t u v w x y z A B C D E F G H I J K L M N O P Q R S T U V W X Y Z ! " \' # $ % & ( ) * + , - . / : ; < = > ? @ [ \\ ] _ ` ~'
        else:
           self.train_alphabet = train_alphabet
        
        if test_alphabet is None:
           self.test_alphabet = '0 1 2 3 4 5 6 7 8 9 a b c d e f g h i j k l m n o p q r s t u v w x y z A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'
        else:
           self.test_alphabet = test_alphabet
        self.test = test
        self.root = root
        self.ifRotate90 = ifRotate90

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('RGB') 
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key))

            label = ''.join(label[i] if label[i] in self.train_alphabet else '' 
                for i in range(len(label)))
            
            if len(label) <= 0:
                return self[index + 1]

            label_rev = label[-1::-1]
            label_rev += ' '
            label += ' '

            w = img.size[0]
            h = img.size[1]
            very_high_flag = h > ( w * 2 )
            if self.transform is not None:
                img = self.transform(img,test=self.test)
                
        if len(label)>35:
           #print('####### '+img_key+' have a long label: '+label)
           label = label[:34] +' '
           label_rev = label_rev[:34] +' '
           #print('####### So we tailored it to '+label)
           
        return (img, label, very_high_flag, label_rev )


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.crop = transforms.RandomCrop((size[1],size[0]))        
        self.pre_pro = transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.05)
        self.toTensor = transforms.ToTensor()

    def __call__(self, img, test=False):
        
        img = img.resize(self.size, self.interpolation)
        if np.random.random() < 0.4 and test==False:
           img = self.pre_pro(img)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.arange(0, self.batch_size)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.arange(0, tail)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)