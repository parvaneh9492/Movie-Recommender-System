import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim 
import torch.utils.data
from torch.autograd import Variable


# Boltzman Machines (A movie recommender system)

movies = pd.read_csv('movies.dat', sep='::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('users.dat', sep='::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ratings.dat', sep='::', header = None, engine = 'python', encoding = 'latin-1')

training_set = pd.read_csv('u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')

test_set = pd.read_csv('u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

users_num = int(max(max(training_set[:,0]), max(test_set[:,0])))
movies_num = int(max(max(training_set[:,1]), max(test_set[:,1])))


def convert(data):
  new_data = []
  for users_id in range(1, users_num + 1):
    movies_id = data[:, 1] [data[:, 0] == users_id]
    ratings_id = data[:, 2] [data[:, 0] == users_id]
    ratings = np.zeros(movies_num)
    ratings[movies_id - 1] = ratings_id
    new_data.append(list(ratings))
  return new_data 

training_set = convert(training_set)
test_set = convert(test_set)

training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3 ] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3 ] = 1


class RBM():
  def __init__(self, nv, nh):
    self.W = torch.randn(nh, nv)
    self.a = torch.randn(1, nh)
    self.b = torch.randn(1, nv)
  def sample_h(self, x):
    wx = torch.mm(x, self.W.t())
    activation = wx + self.a.expand_as(wx)
    p_hv = torch.sigmoid(activation)
    return p_hv, torch.bernoulli(p_hv)
  def sample_v(self, y):
    wy = torch.mm(y, self.W)
    activation = wy + self.b.expand_as(wy)
    p_vh = torch.sigmoid(activation)
    return p_vh, torch.bernoulli(p_vh)
  def train(self, v0, vk, ph0, phk):
    self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
    self.b += torch.sum((v0 - vk), 0)
    self.a += torch.sum((ph0 - phk), 0)
    
nv = len(training_set[0])   
nh = 100
batch_size = 100
rbm = RBM(nv, nh)   



epoch_num = 10
for epoch in range(1, epoch_num + 1):
  train_loss = 0
  s = 0.
  for user_id in range(0, users_num - batch_size, batch_size):
    vk = training_set[user_id : user_id + batch_size]
    v0 = training_set[user_id : user_id + batch_size]
    ph0,_ = rbm.sample_h(v0)
    for k in range(10):
      _,hk = rbm.sample_h(vk)
      _,vk = rbm.sample_v(hk)
      vk[v0 < 0] = v0[v0 < 0]
    phk,_ = rbm.sample_h(vk)
    rbm.train(v0, vk, ph0, phk)
    train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
    s += 1.
  print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))  
  
  
test_loss = 0
s = 0.
for user_id in range(users_num):
    v = training_set[user_id:user_id + 1]
    vt = test_set[user_id:user_id + 1]
    if len(vt[vt >= 0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
        s += 1.
print('test loss: '+str(test_loss/s))
































