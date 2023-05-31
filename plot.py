import pandas as pd
import matplotlib.pyplot as plt
import os, time, json, subprocess, sys, threading, re
import numpy as np
import datetime as dt
import matplotlib.dates as mdates


df = pd.DataFrame()

path="/root/training_results_v2.1/"
#path="/root/inference_results_v3.0/closed/"


#matplotlib.use('tkagg')
fname ="ResNet Dell 16xXE8545x4A100-SXM-40GB"
fname= "ResNet Azure ND96amsr_A100_v4_n8_mxnet"
fname="Resnet Dell 4xXE8545x4A100-SXM-80GB"
fname=" Resnet HPE-ProLiant-XL675d-Gen10-Plus_A100-SXM-80GB_8N_mxnet"
name = "res"
num_node = 1
image_size= 224*224*3*num_node
colnames = ['time','node', 'epoch','speed']
df = pd.read_csv('res0',names=colnames, header=None, delimiter=" " )
df = df.dropna(axis=1, how='all')
df['time'] = pd.to_datetime(df['time'],format='%Y-%m-%d:%H:%M:%S,%f')
df['time'] = pd.to_datetime(df['time']).dt.floor('S')
#df['uni']=df['time'].view('int64')
df['ts'] = df.time.values.astype(np.int64) // 10 ** 9
df.set_index('time', inplace=True)
df['speed']=df['speed']*image_size/1024/1024/1024



colnames = ['time', 'speed']
df = pd.read_csv('res1',names=colnames, header=None, delimiter=" ")
df = df.replace(',','', regex=True)
df['speed'] = df['speed'].astype(float)
df['time'] = pd.to_datetime(df['time'],unit='ms')
#df['thput']=df['thput']*image_size/1024/1024/1024
df.set_index('time', inplace=True)
df['speed']=df['speed']*image_size/1024/1024/1024



'''
#pd.merge(df, df2, left_index=True, right_index=True)
df3 = pd.concat([df, df2], axis=1)
print(df3)
'''
print(df)
print(df.dtypes)

ax = df.plot(y='speed',legend=None, figsize=(8,4))
plt.ylabel("Read IO Throughput (GB/sec)")
plt.xlabel("Time")
plt.title(fname)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
#ticklabels = df.index.strftime('%M-%S')
#res.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
#res.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

plt.savefig(fname+".png")


'''
speed, time = {}, {}
for i in range(1,2):
    fname = name + str(i)
    f = open(fname,'r')
    speed[i] = []
    for row in f:
        row = row.replace("\n","")
        row = row.split(' ')
        speed[i].append(row[3])
        #print(row)

   # print(speed)
   '''
