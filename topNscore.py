# encoding: utf-8
import os
import sys
import pandas as pd
import numpy as np
import time
import calendar
import math

N = 10
sample_num = 100
delta_time_weight = 5 * 24 * 3600

test_data = pd.read_csv('./test_data.txt',header=None,sep='\t').sample(n=sample_num, random_state=0, axis=0)
train_data = pd.read_csv('./train_data.txt',header=None,sep='\t')

poi_vec_data = pd.read_csv('./net_POI_vec.txt',header=None,sep=' ')[[x for x in range(101)]]
time_vec_data = pd.read_csv('./net_time_vec.txt',header=None,sep=' ')[[x for x in range(101)]]
region_vec_data = pd.read_csv('./net_reg_vec.txt',header=None,sep=' ')[[x for x in range(101)]]

user_vec = pd.DataFrame(columns=['userID','VenueId','Time_Slot','Time','Region']+[x for x in range(100)])
poi_vec_dict = {}
time_vec_dict = {}
region_vec_dict = {}

month_dict = dict((v,k) for k,v in enumerate(calendar.month_abbr))

def get_time_slot(time_str):
    return time_str[11:13]

def get_timestamp(time_read):
    week = time_read[0:3]
    month = time_read[4:7]
    day = time_read[8:10]
    hour = time_read[11:13]
    minute = time_read[14:16]
    second = time_read[17:19]
    year = time_read[-4:]
    try:
        format_time = year+'-'+str(month_dict[month])+'-'+day+' '+hour+':'+minute+':'+second
        ts = time.strptime(format_time, "%Y-%m-%d %H:%M:%S")
        time_ = time.mktime(ts)
        print_str = str(int(time_)) + '\t' + str(hour) + '\n'
        return time_
    except:
        print(time_read)

def get_region(loc_str):
    location = loc_str[1:-1]
    locations = location.split(',')
    state, country = locations[3], locations[4]
    if len(locations) is 5:
        city = locations[2]
    else: # len(locations) is 6:
        city = locations[-4] + locations[-3]
    region = ''.join(city.split(' ')) + '_' + ''.join(state.split(' ')) + '_' + ''.join(country.split(' '))
    return region

def write_in_file(f_out_path,str_out):
    f_out = open(f_out_path, 'w+')
    f_out.write(str_out)
    f_out.close()

def get_dicts():
    poi_vec_dict = eval(open('./poi_vec_dict.txt','r').read())
    time_vec_dict = eval(open('./time_vec_dict.txt','r').read())
    region_vec_dict = eval(open('./region_vec_dict.txt','r').read())

def get_user_vec():
    user_vec = pd.read_csv('./user_vec.txt',header=0,sep='\t')

def dicts_gen():
    print('dicts_gen')
    # poi_vec_dict
    for index, row in poi_vec_data.iterrows():
        if row[0] not in poi_vec_dict:
            poi_vec_dict[row[0]] = np.array(row[1:])
        else:
            print('poi_vec_dict重复:',row[0])
    write_in_file('./poi_vec_dict.txt',str(poi_vec_dict))
    # time_vec_dict
    for index, row in time_vec_data.iterrows():
        if int(row[0]) not in time_vec_dict:
            time_vec_dict[int(row[0])] = np.array(row[1:])
        else:
            print('time_vec_dict重复:',row[0])
    write_in_file('./time_vec_dict.txt',str(time_vec_dict))
    # region_vec_dict
    for index, row in region_vec_data.iterrows():
        if row[0] not in region_vec_dict:
            region_vec_dict[row[0]] = np.array(row[1:])
        else:
            print('region_vec_dict重复:',row[0])
    write_in_file('./region_vec_dict.txt',str(region_vec_dict))

def user_vec_gen():
    print('user_vec_gen')
    user_vec['userID'] = test_data[1]
    user_vec['VenueId'] = test_data[3]
    user_vec['Time_Slot'] = test_data[2].apply(get_time_slot)
    user_vec['Time'] = test_data[2].apply(get_timestamp)
    user_vec['Region'] = test_data[5].apply(get_region)
    
    train_data['Time'] = train_data[2].apply(get_timestamp)
    # print(poi_vec_data)
    for index, row in user_vec.iterrows():
        vec = np.zeros(100)
        pois_time = train_data[(train_data['Time']<row['Time'])&(train_data[1]==row['userID'])][[3,'Time']]
        for index_, row_ in pois_time.iterrows():
            p_v = np.array(poi_vec_data[poi_vec_data[0]==row_[3]][poi_vec_data.columns[1:]].iloc[0])
            vec = vec + math.exp(-(row['Time']-row_['Time'])/delta_time_weight) * p_v
        user_vec.loc[index,[x for x in range(100)]] = vec
    user_vec.to_csv('./user_vec.txt',sep='\t',header=True,index=0)


def topNscore():
    print('topNscore')
    hit_num = 0
    sample_num = 0
    for index, row in user_vec.iterrows():
        sample_num += 1
        
        userID = row['userID']
        poiID = row['VenueId']
        time_slot = row['Time_Slot']
        region = row['Region']
        
        userID_vec = np.array(row[5:])
        # # poiID_vec = np.array(poi_vec_data[poi_vec_data[0]==poiID][poi_vec_data.columns[1:]].iloc[0])
        # time_slot_vec = np.array(time_vec_data[time_vec_data[0]==int(time_slot)][time_vec_data.columns[1:]].iloc[0])
        # region_vec = np.array(region_vec_data[region_vec_data[0]==region][region_vec_data.columns[1:]].iloc[0])
        time_slot_vec = time_vec_dict[int(time_slot)]
        region_vec = region_vec_dict[region]
        
        topN = [('',-sys.maxsize-1)] * N
        # for index_, row_ in poi_vec_data.iterrows():
        for key, value in poi_vec_dict.items():
            # poi_vec = np.array(row_[1:])
            score = (userID_vec+region_vec+time_slot_vec).dot(value)
            topN.sort(key=lambda x:x[1])
            if topN[0][1] < score:
                topN[0] = (key, score)
        topN_pois = [x[0] for x in topN]
        if poiID in topN_pois:
            hit_num += 1
        print('第',sample_num,'个样本',hit_num,sample_num, hit_num/sample_num)
        
        
if __name__ == '__main__':
    # dicts_gen()
    get_dicts()
    # user_vec_gen()
    get_user_vec()
    topNscore()