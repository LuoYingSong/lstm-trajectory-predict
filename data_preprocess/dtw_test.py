#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import sys
sys.path.append(
    '..'
)
from DTW import DTW
from net_work.sender2 import Sender
from util import haversine


# In[ ]:


sender = Sender()


# In[ ]:


data = sender.send(1)


# In[ ]:


import ujson
with open('./sender.json') as f:
    data2 = ujson.load(f)


# In[ ]:


value_list, ptr_list = [], []
for road,ptr in data2[2:]:
    value,ptr = DTW(data2[1][0], road)
    if  value != 1:
        if value < 0.00006:
            print(value,ptr)
        value_list.append(value)
        ptr_list.append(ptr)


# In[ ]:

print(ptr_list[value_list.index(min(value_list))],data2[0][1])
print(haversine(ptr_list[value_list.index(min(value_list))],data2[0][1]))
print(min(value_list))

# In[ ]:




