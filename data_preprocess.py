#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
from torch.utils.data import Dataset,DataLoader
from pycocotools.coco import COCO
from PIL import Image
import os
import nltk


# In[3]:


class CocoDataset(Dataset):
    def __init__(self, root, json, vocab, transform = None):
        
        self.root = root
        self.coco = COCO(json)
        self.ids  = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform
        
    def __getitem__(self,index):
        
        ann_id = self.ids[index]
        caption = self.coco.anns[ann_id]['caption']
        image_id = self.coco.anns[ann_id]['image_id']
        image_name = self.coco.loadImgs(image_id)[0]['file_name']
        
        image = Image.open(os.path.join(self.root,image_name)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
             
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(self.vocab('<start>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        target = torch.Tensor(caption)
        
        return image,target
    
    def __len__(self):
        return len(self.ids)


# In[4]:


def my_collate(data):
    
    data.sort(key= lambda x: len(x[1]), reverse = True)
    images , captions = zip(*data)
    
    images = torch.stack(images,0)
    
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions),max(lengths)).long()
    for i,cap in enumerate(captions):
        end = lengths[i]
        targets[i,:end] = cap[:end]
    return images,targets,lengths


# In[6]:


def get_loader(root, json ,vocab , transform, batch_size ,shuffle ,num_workers):
    coco = CocoDataset(root = root,
                      json = json,
                      vocab = vocab,
                      transform = transform)
    
    data_loader = DataLoader(dataset= coco ,
                            batch_size = batch_size,
                            shuffle = shuffle,
                            num_workers = num_workers,
                            collate_fn = my_collate)
    
    return data_loader

