import pandas as pd
import numpy as np
from config import batch_size

def mapping(line:pd.core.series.Series)->pd.core.series.Series:
    alpha = 'abcdefghijklmnopqrstuvwxyz'
    alpha_dict = dict(zip(alpha,range(1,27)))
    count = 0
    ret = []
    for i in line:
        ret.append(alpha_dict[i])
        count += 1
    return ret


class dataSet:

    def __init__(self, dataType:str,path:str):
        self.dataType = dataType
        self.data = pd.read_csv(path)
        count = 0

        # object refers to all str 
        for index in self.data.columns[self.data.dtypes=='object']:
            #print(type(self.data[index]))
            self.data[index] = mapping(self.data[index])
            # print(self.data[index])

        if dataType == 'train':
            self.label = self.data['Score']
            del self.data['Score']
            del self.data['Id']

            m = self.data.shape[0]
            num = int(m/batch_size)
            print(num)
            batch_data = []
            for i in range(num):
                temp_d = self.data[i*batch_size:(i+1)*batch_size]
                temp_l = self.label[i*batch_size:(i+1)*batch_size]
                batch_data.append((temp_d,temp_l))
            print(len(batch_data))
            self.data = batch_data
        else: #test
            self.label = None
            del self.data['Id']
    
    def write_back(self):
        if self.dataType != 'test':
            return
        Id = range(1, len(self.label) + 1)
        label = self.label.cpu()
        df = pd.DataFrame({
            'Score':list(label.detach().numpy().reshape(-1))
        })
        df.to_csv("submission.csv",index_label='Id')




           


