from config import TRAIN_DATA_PATH,TEST_DATA_PATH,epoches

from dataProcessing import dataSet
from pytorch_model import Neuralnetwork
import numpy as np
import torch


train_data = dataSet('train', TRAIN_DATA_PATH)
test_data = dataSet("test", TEST_DATA_PATH)
print(test_data.data.shape)


 
model = Neuralnetwork(32, 400, 200, 1)
 

optimizer = torch.optim.Adam(list(model.parameters()), lr=0.0001, betas=(0.9, 0.999),weight_decay=0.004) 
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.997) 
loss = torch.nn.MSELoss()


print(model)

if torch.cuda.is_available(): # 有GPU，则用GPU计算
    model.cuda() 
    loss.cuda()

for epoch in range(epoches): 
    losses = [] 
    # ERROR_Train = [] 
    model.train() 
    for (data,label) in train_data.data: 
        model.zero_grad()# 首先提取清零 
        
        data = torch.Tensor(data.values).cuda()
        label = torch.Tensor(label.values).cuda()

        if torch.cuda.is_available():# CUDA可用情况下，将Tensor 在GPU上运行   
            datav = torch.autograd.Variable(data).cuda()
            labelv = torch.autograd.Variable(label).cuda()
            output = model(datav) 
            err = loss(output, labelv) 
            err.backward() 
            optimizer.step() 
            losses.append(err.item()) 

    print('[%d/%d] Loss: %.4f ' % (epoch, epoches, np.average(losses)))

datat = torch.Tensor(test_data.data.values).cuda()
result = model(datat) 
test_data.label = result
test_data.write_back()
print(result.shape)
