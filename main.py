#!/usr/bin/env python
# coding: utf-8

# In[61]:


import os
from pathlib import Path
import pandas as pd
from openai import OpenAI
import base64


# In[57]:


client = OpenAI(api_key='INSERT_API_KEY_HERE')


# In[11]:


def convert_path_to_df(dataset):
    image_dir = Path(dataset)

     # Get filepaths and labels
    filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) + list(image_dir.glob(r'**/*.png')) + list(image_dir.glob(r'**/*.PNG'))
    
    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    # Concatenate filepaths and labels
    image_df = pd.concat([filepaths, labels], axis=1)
    return image_df


# In[12]:


dataset = os.getcwd() + '/datasets/farm_insects'
image_df = convert_path_to_df(dataset)


# In[58]:


labels = ', '.join(set(image_df['Label']))


# In[ ]:


quantidade_total = 1000 
acerto = 0
for index, row in enumerate(image_df.sample(n=quantidade_total).iterrows()):
    filepath = row[1]['Filepath']
    label = row[1]['Label']
    with open(filepath, "rb") as image_file:
        extension = os.path.splitext(filepath)
        extension = extension[len(extension) - 1].replace('.', '')
        encoded_string = base64.b64encode(image_file.read()).decode()
        image_url = "data:image/"+ extension +";base64,"+encoded_string
        response = client.chat.completions.create(
          model="gpt-4-vision-preview",
          messages=[
            {
              "role": "user",
              "content": [
                {"type": "text", "text": "Dentre essas opções " + labels + ". Me retorne como você classifica essa imagem, retorne apenas o nome igual foi passado na lista"},
                {
                  "type": "image_url",
                  "image_url": {
                    "url": image_url,
                  },
                },
              ],
            }
          ],
          max_tokens=1000,
        )
        resposta_ia = response.choices[0].message.content
        if (resposta_ia == label):
            acerto = acerto + 1
        print(str(index + 1) + ": " + str((acerto/(index + 1)) * 100))


# In[ ]:




