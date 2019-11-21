# encoding: utf-8
import os
import pandas as pd
import numpy as np
import time
import calendar
import math
from progress.bar import Bar

train_frac = 0.9
delta_t = 25 * 24 *3600
d = 100
k = 5
checkin_data_path = 'checkin_CA_venues.txt'

checkin_data = pd.read_csv('./'+checkin_data_path,header=0,sep='\t')
train_data = checkin_data.sample(frac=train_frac, random_state=0, axis=0)
test_data = checkin_data[~checkin_data.index.isin(train_data.index)]

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

def get_words(words_str):
    words = words_str[1:-1].split(',')[:-1]
    return [x.replace(' ','') for x in words]
    
def get_strline_out(str_list):
    return '\t'.join([str(x) for x in str_list]) + '\n'

def write_in_file(f_out_path,str_out):
    f_out = open(f_out_path, 'w+')
    f_out.write(str_out)
    f_out.close()

def pois_gen():
    print('pois_gen')
    pois = train_data['VenueId'].drop_duplicates()
    str_out = ''
    for index, poiID in pois.iteritems():
        str_out += get_strline_out([poiID])
    write_in_file('./POIs.txt',str_out)

def poi_poi_gen():
    print('poi_poi_gen')
    userIDs = train_data['userID'].drop_duplicates()
    poi_poi_dict = {}
    bar = Bar('poi_poi_gen', max=len(userIDs))
    for uid in userIDs:
        databyid = train_data[train_data['userID']==uid][['VenueId','Time(GMT)']]
        data_list = []
        for index, row in databyid.iterrows():
            data_list.append((row['VenueId'],get_timestamp(row['Time(GMT)'])))
        for index,(poiID,timestamp) in enumerate(data_list):
            for (poiID_,timestamp_) in data_list[index+1:]:
                if timestamp > timestamp_ and timestamp - timestamp_ < delta_t:
                    if poiID_ not in poi_poi_dict:
                        poi_poi_dict[poiID_] = {poiID : 1}
                    else:
                        if poiID not in poi_poi_dict[poiID_]:
                            poi_poi_dict[poiID_][poiID] = 1
                        else:
                            poi_poi_dict[poiID_][poiID] += 1
        bar.next()
    bar.finish()
    str_out = ''
    for poiID, value in poi_poi_dict.items():
        for poiID_, weight in poi_poi_dict[poiID].items():
            str_out += get_strline_out([poiID,poiID_,weight])                        
    write_in_file('./net_POI.txt',str_out)

def poi_reg_gen():
    print('poi_reg_gen')
    poi_reg_dict = {}
    for index, row in train_data.iterrows():
        poiID = row['VenueId']
        region = get_region(row['VenueLocation'])
        if poiID not in poi_reg_dict:
            poi_reg_dict[poiID] = region
    str_out = ''
    for poiID, region in poi_reg_dict.items():
        str_out += get_strline_out([poiID, region, 1])
    write_in_file('./net_POI_reg.txt',str_out)

def poi_time_gen():
    print('poi_time_gen')
    poi_time_dict = {}
    for index, row in train_data.iterrows():
        poiID = row['VenueId']
        time_slot = get_time_slot(row['Time(GMT)'])
        if poiID not in poi_time_dict:
            poi_time_dict[poiID] = {time_slot:1}
        else:
            if time_slot not in poi_time_dict[poiID]:
                poi_time_dict[poiID][time_slot] = 1
            else:
                poi_time_dict[poiID][time_slot] += 1
    str_out = ''
    for poiID, value in poi_time_dict.items():
        for time_slot, weight in value.items():
            str_out += get_strline_out([poiID,time_slot,weight])
    write_in_file('./net_POI_time.txt',str_out)

def poi_word_gen():
    print('poi_word_gen')
    data = train_data.drop_duplicates(subset='VenueId')
    length = len(data)
    poi_word_tf_dict = {}
    word_numinpoi_dict = {}
    for index, row in data.iterrows():
        poiID = row['VenueId']
        words = get_words(row['VenueCategory'])
        words_length = len(words)
        for word in words:
            if poiID not in poi_word_tf_dict:
                poi_word_tf_dict[poiID] = {word : 1/words_length}
            else:
                if word not in poi_word_tf_dict[poiID]:
                    poi_word_tf_dict[poiID][word] = 1/words_length
                else:
                    print('poi_word_gen错误(poiID,word)',poiID,word)
            if word not in word_numinpoi_dict:
                word_numinpoi_dict[word] = 1
            else:
                word_numinpoi_dict[word] += 1
    str_out = ''
    for poiID, value in poi_word_tf_dict.items():
        for word, tf in poi_word_tf_dict[poiID].items():
            str_out += get_strline_out([poiID,word,tf*math.log(length/(word_numinpoi_dict[word]+1))])
    write_in_file('./net_POI_word.txt',str_out)
            
            
def train_data_gen():
    print('train_data_gen')
    train_data.to_csv('./train_data.txt',sep='\t',header=False)
    # to_csv index=0 不保存行索引
    
def test_data_gen():
    print('test_data_gen')
    test_data.to_csv('./test_data.txt',sep='\t',header=False)
    # to_csv index=0 不保存行索引
    
if __name__ == '__main__':
    print('checkin数据为:',checkin_data_path)
    print('训练集占比:',train_frac)
    print('时间间隔:',str(int(delta_t/24/3600)),'天')
    train_data_gen()
    test_data_gen()
    pois_gen()
    poi_poi_gen()
    poi_time_gen()
    poi_reg_gen()
    poi_word_gen()
    
    # os.system('./GEmodel -size '+str(d)+' -order 2 -negative '+str(k)+' -rho 0.025 -threads 2')