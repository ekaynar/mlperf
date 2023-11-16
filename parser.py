import pandas as pd
import matplotlib.pyplot as plt
import csv, os, time, json, subprocess, sys, threading, re
import numpy as np
import datetime as dt
import matplotlib.dates as mdates
import yaml



with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)


benchmark_name = config['benchmark']
gpu = config['gpu']
memory = config['memory']
path = config['input_dir']

def get_sample_size(benchmark_name):
    if benchmark_name == "gpt3":
        sample_size = 8*1024
    elif benchmark_name == "resnet":
        sample_size = 118784
    elif benchmark_name == "bert":
        sample_size = 2048
    elif benchmark_name == "unet3d":
        sample_size = 96468992
    elif benchmark_name == "dlrm_dcnv2":
        sample_size = 912 
    elif benchmark_name == "maskrcnn":
        sample_size = 163840 

    return sample_size

sample_size = get_sample_size(benchmark_name)
output = benchmark_name+"_system"


start="results/"
end="/"+benchmark_name



# Get System Info and Build Dataframe
def get_system_info():
    df = pd.DataFrame()
    res =  subprocess.run(['find '+path+' -type d -name "' +benchmark_name+'"'], shell=True, capture_output=True, text=True)
    dirs, runs = [],[]
    dirs = res.stdout.split("\n")
    
    dfs = []
    for item in dirs[:]:
        if "/results/" not in item:
            dirs.remove(item)
        else:
            runs.append(item[item.find(start)+len(start):item.rfind(end)])
    dirs = [*set(dirs)]

    for run_name in runs:
        res =  subprocess.run(['find '+path+' -type f -name "' +run_name+'.json"'], shell=True, capture_output=True, text=True)
        file =  res.stdout.split("\n")[0]
        if "/systems/" in file:
            with open(file) as fd:
                data = pd.json_normalize(json.loads(fd.read()))
                data['run_name'] = "/"+run_name+"/"
                dfs.append(data)
    df = pd.concat(dfs, ignore_index = True)
    names =['run_name','submitter', 'system_name', 'number_of_nodes','accelerators_per_node', 'accelerator_model_name', 'accelerator_memory_capacity', 'accelerator_interconnect'] 
    df = df[names]
    df = df[df['accelerator_model_name'].str.contains(gpu)]
    df = df[df['accelerator_memory_capacity'].str.contains(str(memory))]
    df['accelerators_per_node'] = df['accelerators_per_node'].astype(int)
    df['number_of_nodes'] = df['number_of_nodes'].astype(int)
    df.reset_index(drop=True, inplace = True)
    return df, dirs

df , dirs = get_system_info()
print(df)
#exit(0)

if config['workload'] == "training":
    col_names = ["avg_thput","batch_size","num_epochs","global_batch","epoch_num","run_start", "run_stop", "t_seq_per_sec", "train_samples", "eval_samples", 'samples_per_sec', "train_batch_size", "eval_batch_size"]

elif config['workload'] == "inference":
    col_names = ["inference_time","inference_img_p_sec","avg_thput","batch_size","num_epochs","global_batch","epoch_num","run_start", "run_stop", "t_seq_per_sec", "train_samples", "eval_samples", 'samples_per_sec', "train_batch_size", "eval_batch_size"]

for col in col_names:
    df[col] = 0


# Parse result filesto extract informations

def find_sub(lword, lstr):
    out = set()
    for st in lstr:
        rex = st+"(.*?)"+","
        if st == '"epoch_num": ':
            rex = st+"(.*?)"+"}"
        
        result = re.search(rex, lword)
        if result:
            out.add(st+result.group(1))
    return out

def file_parser(line, run):
    t = "\"time_ms\":"
    if "run_start\"" in line:
        rex = t+"(.*?)"+","
        time = re.search(rex, line)
        time = re.sub("[^0-9]","",time.group(1))
        df.at[run,"run_start"] = int(time)
    elif "run_stop\"" in line:
        rex = t+"(.*?)"+","
        time = re.search(rex, line)
        time = re.sub("[^0-9]","",time.group(1))
        df.at[run,"run_stop"] = int(time)
    
    elif "train_samples" in line and "time_ms" in line:
        t = "train_samples\","
        rex = t+"(.*?)"+","
        time = re.search(rex, line)
        time = re.sub("[^0-9]","",time.group(1))
        df.at[run,"train_samples"] = int(time)

    elif "eval_samples" in line and "time_ms" in line:
        t = "eval_samples\","
        rex = t+"(.*?)"+","
        time = re.search(rex, line)
        time = re.sub("[^0-9]","",time.group(1))
        df.at[run,"eval_samples"] = int(time)

     
    if "training_sequences_per_second" in line:
        t = "training_sequences_per_second\':"
        rex = t+"(.*?)"+","
        time = re.search(rex, line)
        time = re.sub("[^0-9.]","",time.group(1))
        df.at[run,"t_seq_per_sec"] = round(float(time),2)
   
    if "--train_batch_size=" in line:
        t = " --train_batch_size="
        rex = t+"(.*?)"+"\s"
        time = re.search(rex, line)
        df.at[run,"train_batch_size"] = int(time.group(1))
    
    if " train_batch_size=" in line:
        t = " train_batch_size="
        rex = t+"(.*?)"+","
        time = re.search(rex, line)
        if int(df.loc[run]["train_batch_size"]) < int(time.group(1)) :
            df.at[run,"train_batch_size"] = int(time.group(1))
    
    if "--eval_batch_size=" in line:
        t = " --eval_batch_size="
        rex = t+"(.*?)"+"\s"
        time = re.search(rex, line)
        df.at[run,"eval_batch_size"] = int(time.group(1))
   
    if " eval_batch_size=" in line:
        #print(line)
        t = " eval_batch_size="
        rex = t+"(.*?)"+","
        time = re.search(rex, line)
        #print(time.group(1), "--", run, df.loc[run]["eval_batch_size"], "tt")
       # print(df)
        if df.loc[run]["eval_batch_size"] < int(time.group(1)) :
            df.at[run,"eval_batch_size"] = int(time.group(1))

    if "Total inference time" in line:
        rex = "time: "+"(.*?)"+"/"
        time = re.search(rex, line)
        res = time.group(1)
        tmp = res.split()
        df.at[run,"inference_time"] = tmp[0]
        df.at[run,"inference_img_p_sec"] = tmp[1].replace("(","")

    
def get_thput(line, thput, timestamp):
    if benchmark_name in ["gpt3", "resnet", "dlrm_dcnv2", "maskrcnn"]:
        t = "throughput\": " 
        rex = t+"(.*?)"+"}" 
    
    elif benchmark_name in ["stable_diffusion", "unet3d", "bert"]:
        t = "throughput\":"
        rex = t+"(.*?)"+"," 
    

    if "throughput\": " in line:  #markcnn #unet3d
        time = re.search(rex, line)
        time = re.sub("[^0-9.]","",time.group(1))
        thput.append(float(time))

        t = "time_ms"
        rex = t+"(.*?)"+","
        time = re.search(rex, line)
        time = re.sub("[^0-9]","",time.group(1))
        timestamp.append(int(time))
    
def get_param(fname, param, run_name):
    res = set()
    thput=[]
    timestamp=[]
    print(fname)
    with open(fname, encoding='utf-8', errors='ignore') as file:
        for line in file:
            r = find_sub(line,param)
            res = res.union(r)
            file_parser(line, run_name)
            get_thput(line, thput, timestamp)
        if len(thput)>0:
            avg_thput = sum(thput)/len(thput)
            df.at[run_name,'avg_thput'] = float(avg_thput)
    return res, thput, timestamp


params=['BATCHSIZE=','batch_size=', '"d_batch_size",','"global_batch_size",', '"local_batch_size"','"epoch_num": ', 'num_epochs=', ]

def add_to_df(df, run, val):
    for v in val:
        if "batch_size=" in v or "d_batch" in v or "BATCHSIZE=" in v or "local_batch" in v:
            v = re.sub("[^0-9.]","",v)
            df.at[run,"batchgsize"] = v
        
        elif "num_epochs" in v: 
            v = re.sub("[^0-9.]","",v)
            df.at[run,"num_epochs"] = float(v)
        
        elif "global_batch" in v: 
            v = re.sub("[^0-9.]","",v)
            df.at[run,"global_batch"] = v
       
        elif "epoch_num" in v:
            v = re.sub("[^0-9.]","",v)
            if float(df.loc[run]["epoch_num"]) < float(v) :
                df.at[run,"epoch_num"] = v

thput_path = "/root/thput_results/"+benchmark_name
try:
    os.mkdir(thput_path)
except OSError as error:
    print(error)

system_list = list(df['run_name'])
df = df.set_index(['run_name'])
for sys in system_list:
    for fname in dirs:
        if sys in fname:
        #if sys in fname and "Azure" not in fname:
            path = fname + "/result_1.txt" 
            if not os.path.isfile(path):
                path = fname + "/result_01.txt"
            thput = []
            if os.path.isfile(path):
                res, thput, timestamp = get_param(path, params, sys)
                add_to_df(df, sys, res)
                #name = sys[:-1]
                p = thput_path  + sys[:-1] + ".csv"
                with open(p, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerows(zip(timestamp, thput))

def calculate_per_gpu_numbers(df):
    mb = 1024*1024
    gb = mb*1024
    df['train_data_bw_mb'] = df['train_sample_size'] * df['avg_thput'] / mb
    df['train_data_bw_gb'] = df['train_sample_size'] * df['avg_thput'] / gb
    df['total_gpus'] = df['number_of_nodes'] * df ['accelerators_per_node']
    df['train_data_per_gpu_mbps'] = df['train_data_bw_mb'].astype(float) / df['total_gpus'].astype(float)
    df['samples/gpu'] =  df['avg_thput'] / df['total_gpus'].astype(float)

df['train_sample_size'] = float(sample_size)
df['run_time'] = df['run_stop'] - df['run_start']
df['run_time'] = df['run_time']/1000
df['samples_per_sec'] = df['train_samples'] / df['run_time']
df['t_seq_per_sec'].round(decimals = 2)
#print(df[['inference_time','inference_img_p_sec']])
#print(df[['run_start','run_stop','run_time','raw_train_time','t_seq_per_sec','s_p_sec', 'train_samples']])

calculate_per_gpu_numbers(df)
df = df.loc[:, df.any()]

df.to_csv("/root/thput_results/"+output+".csv")


