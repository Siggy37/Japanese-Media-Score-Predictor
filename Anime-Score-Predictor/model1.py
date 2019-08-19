# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 15:06:24 2019

@author: brand
"""
import wordencoding as we
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

seed = 37
torch.manual_seed(seed)
np.random.seed(seed)

class title_encoder(nn.Module):
    def __init__(self):
        super(title_encoder, self).__init__()
        
        self.linear1 = nn.Linear(18415, 9208)
        self.linear2 = nn.Linear(9208, 8)
        

        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x
    
class genre_encoder(nn.Module):
    def __init__(self):
        super(genre_encoder, self).__init__()
        
        self.linear1 = nn.Linear(222, 6)

        
    def forward(self, x):
        return F.relu(self.linear1(x))
    
    
class summary_encoder(nn.Module):
    def __init__(self, batch_size):
        super(summary_encoder, self).__init__()
        self.batch_size = batch_size
        self.gru        = nn.GRU(100, 64)
        self.linear1    = nn.Linear(64, 8)
        
        
    def forward(self, x):
        x, _ = self.gru(x)
    #   print(x.shape)
        x     = F.relu(self.linear1(x))
     #   print(x.shape)
        #x     = x.view(self.batch_size, -1, 8)
        
        return x

class predictor(nn.Module):
    def __init__(self, batch_size, output_size):
        super(predictor, self).__init__()
        self.batch_size      = batch_size
        self.output_size     = output_size
        self.summary_encoder = summary_encoder(batch_size)
        self.genre_encoder   = genre_encoder()
        self.title_encoder   = title_encoder()
        self.linear_input    = 27
        self.linear1         = nn.Linear(self.linear_input, output_size)
        #self.softmax         = nn.Softmax(dim=1)
    
    def forward(self, summaries, genres, titles, platforms):
        s = self.summary_encoder(summaries)
        g = self.genre_encoder(genres)
        t = self.title_encoder(titles)
        s = s.view(self.batch_size, -1)
        full = torch.cat((s,g,t,platforms), dim=1)
        x = self.linear1(full)
        
        return x
        
        
        
        

    
def train_model(new_model=True):
    max_epoch = 10    
    #237
    batch_divisor = 237    
    BATCH_SIZE = 17

    output_size = 11


    if new_model == True:
        model = predictor(BATCH_SIZE, output_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)    
    else:
        modelinfo = torch.load('/ModelState.pt')
        model = predictor(17, 11)
        model.load_state_dict(modelinfo['model_state_dict'])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        optimizer.load_state_dict(modelinfo['optimizer_state_dict'])
    
    model.train()

    objective_function = nn.CrossEntropyLoss()    
    print("Loading Data...")
    summary, title_vecs, platform_vecs, genre_vecs, tags = we.get_all_data()
    print(summary.shape)
    print(title_vecs.shape)
    print("Data Loaded")    
    for epoch in range(max_epoch):
        print("Starting Epoch {}".format(epoch))
        currstart = 0
        for minibatch in range(batch_divisor):
#            minilosses = list()
            print("Starting minibatch {}".format(minibatch))
            optimizer.zero_grad()
            summary_batch  = torch.from_numpy(summary[currstart: currstart + BATCH_SIZE]).float()
            summary_batch  = summary_batch.view(BATCH_SIZE, -1, 100)
       #     print(summary_batch.shape)
            title_batch    = torch.from_numpy(title_vecs[currstart: currstart + BATCH_SIZE]).float()
            platform_batch = torch.from_numpy(platform_vecs[currstart: currstart + BATCH_SIZE]).float()
            genre_batch    = torch.from_numpy(genre_vecs[currstart: currstart + BATCH_SIZE]).float()
            tag_batch      = torch.Tensor(tags[currstart: currstart + BATCH_SIZE]).long()
            currstart += BATCH_SIZE            
            outputs = model(summary_batch, genre_batch, title_batch, platform_batch)        
            loss    = objective_function(outputs, tag_batch)
            print(loss)
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            
    model_info = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()           
            }
    
    torch.save(model_info, 'ModelState3.pth')
    
    
def train_test():
    max_epoch = 1    
    #237
    batch_divisor = 2    
    BATCH_SIZE = 17

    output_size = 11
    model = predictor(17,11)  
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    objective_function = nn.CrossEntropyLoss()    
    print("Loading Data...")
    summary, title_vecs, platform_vecs, genre_vecs, tags = we.get_all_data()
    print(summary.shape)
    print(title_vecs.shape)
    print("Data Loaded")    
    for epoch in range(max_epoch):
        print("Starting Epoch {}".format(epoch))
        currstart = 0
        for minibatch in range(batch_divisor):
#            minilosses = list()
            print("Starting minibatch {}".format(minibatch))
            optimizer.zero_grad()
            summary_batch  = torch.from_numpy(summary[currstart: currstart + BATCH_SIZE]).float()
            summary_batch  = summary_batch.view(BATCH_SIZE, -1, 100)
            title_batch    = torch.from_numpy(title_vecs[currstart: currstart + BATCH_SIZE]).float()
            platform_batch = torch.from_numpy(platform_vecs[currstart: currstart + BATCH_SIZE]).float()
            genre_batch    = torch.from_numpy(genre_vecs[currstart: currstart + BATCH_SIZE]).float()
            tag_batch      = torch.Tensor(tags[currstart: currstart + BATCH_SIZE]).long()
            currstart += BATCH_SIZE            
            outputs = model(summary_batch, genre_batch, title_batch, platform_batch)        
            loss    = objective_function(outputs, tag_batch)
            print(loss)
            loss.backward()
            optimizer.step() 
            model_info = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()           
                    }
            torch.save(model_info, "model_info.pth")
            model = predictor(17,11)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            info = torch.load("model_info.pth")
            model.load_state_dict(info['model_state_dict'])
            optimizer.load_state_dict(info['optimizer_state_dict'])
            model.train()

            
        
    
if __name__ == "__main__":
    """
    senc = summary_encoder(17).cuda()
    summaries = np.load('./bert/summary_array.npy')
    s = torch.from_numpy(summaries[:17]).float().cuda()
    y = senc(s)
    print(y.shape)
    """
    print("Starting Training")
    train_model()
    print("Finished")
    
    