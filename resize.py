#!/usr/bin/env python
# coding: utf-8

# In[7]:


from PIL import Image
import os
import argparse


# In[3]:


def resize_image(image,size):
    return image.resize(size,Image.BICUBIC)


# In[11]:


def resize_images(input_dir , output_dir , size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    images = os.listdir(input_dir)
    num_images = len(images)
    for i, image in enumerate(images):
        with open(os.path.join(input_dir,image),'r+b') as f:
            with Image.open(f) as img:
                img = resize_image(img,size)
                img.save(os.path.join(output_dir,image),img.format)
        if (i+1)%1000==0:
            print('{}/{} images resized and saved into {} '.format(i+1,num_images,output_dir))


# In[12]:


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    image_size = [args.image_size,args.image_size]
    resize_images(input_dir,output_dir,image_size)


# In[13]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',type=str,
                       default='data/train2014/',
                       help='this is directory of images')
    parser.add_argument('--output_dir',type=str,
                       default='data/processed_images/',
                       help='this is directory for output of processed images')
    parser.add_argument('--image_size',type=int,
                       default = 256, help = 'size for image after processing')
    args = parser.parse_args(args=[])
    main(args)


# In[ ]:




