#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import torch
import torch.nn as nn
import os
from torchvision import transforms
import pickle
from data_preprocess import get_loader
from model import EncoderCNN,DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from build_vocab import Vocabulary
import numpy as np


# In[ ]:


device = ('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[ ]:


def main(args):
    #create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    #image preprocessing and normalzation stuff(define transforms)
    transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),
                             (0.229,0.224,0.225)),
    ])
    
    
    #load vocabulary wrapper
    with open(args.vocab_path , 'rb') as f:
        vocab = pickle.load(f)


    #build data-loader
    data_loader = get_loader(args.image_dir , args.caption , vocab ,transform ,args.batch_size ,shuffle = True ,
                             num_workers = args.num_workers)
    
    #build the model
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(len(vocab) , args.embed_size , args.hidden_size , args.num_layers).to(device)
    
    #define loss and optimizer function
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.batch_norm.parameters())
    optimizer = torch.optim.Adam(params , lr = args.learning_rate)
    
    
    #train the model
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        for i,(images,captions,lengths) in enumerate(data_loader):
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions , lengths ,batch_first=True)[0]
            
            #forward and backprop
            features = encoder(images)
            outputs = decoder(features,captions,lengths)
            
            loss = criterion(outputs , targets)
            decoder.zero_grad()
            encoder.zero_grad()
            
            loss.backward()
            optimizer.step()
            
            #print log info
            if i%args.log_step == 0:
                print('Epoch {}/{}  , Step {}/{}  , Loss {:.4f} , Perplexity{:5.4f}'.format(epoch,args.num_epochs,
                                                                                           i,total_step , loss.item(),
                                                                                           np.exp(loss.item())))
            
            #save the model
            if (i+1)%args.save_step == 0:
                torch.save(decoder.state_dict(),os.path.join(args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1,i+1)))
                torch.save(encoder.state_dict(),os.path.join(args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1,i+1)))
            


# In[ ]:


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path' , type=str , default='models/',help='path for saving models')
    parser.add_argument('--crop_size' , type=int , default= 224, help= 'crop size')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl' ,help = 'this is path for vocabulary wrapper')
    parser.add_argument('--image_dir',type =str, default='data/processed_images/' ,help = 'path for processed images')
    parser.add_argument('--caption' ,type=str, default='data/annotations/captions_train2014.json' ,help = 'path for train annotations')
    parser.add_argument('--batch_size' ,type=int ,default = 128)
    
    parser.add_argument('--embed_size', type =int ,default =256 ,help = 'dimension of word embedding vectors')
    parser.add_argument('--hidden_size',type = int ,default = 512 ,help = 'dimension of lstm hidden states')
    parser.add_argument('--num_layers', type = int , default = 1, help = 'number of layers of lstm')
    
    parser.add_argument('--learning_rate', type =float , default =0.001)
    parser.add_argument('--num_epochs' ,type = int , default = 5)
    parser.add_argument('--num_workers', type =int ,default =4)
    parser.add_argument('--log_step' ,type =int ,default = 10 ,help = 'step size for printing log info')
    parser.add_argument('--save_step',type =int ,default = 1000,help = 'step size for saving the model')
    args = parser.parse_args(args =[])
    print(args)
    main(args)
    


# In[ ]:





# In[ ]:




