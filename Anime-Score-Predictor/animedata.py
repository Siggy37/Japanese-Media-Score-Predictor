# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 21:38:10 2019

@author: brand
"""
import csv
import pandas as pd
import json
import numpy as np

genrepath   = './data/datagenre-all-share-new.csv'
scorepath   = './data/datascorehist-all-share-new.csv'
titlepath   = './data/datatitle-all-share-new.csv'
summarypath = './data/datasynopsis-all-share-new.csv'
staffpath   = './data/datastaff-all-share-new.csv'

def get_genres():
    genre_dict = dict()
    genres = list(csv.reader(open(genrepath, 'r')))
    for g in genres:
        info = g[0].split('|')
        ID = info[0]
        genres = info[1].split(';')        
        genre_dict[ID] = genres
    return genre_dict

def get_scores():
    score_dict = dict()
    scores = list(csv.reader(open(scorepath, 'r')))
    for s in scores[1:]:
        info = s[0].split('|')
        ID = info[0]
        ss = list(info[1:])
        score_dict[ID] = ss        
    return score_dict

def get_titles():
    title_dict = dict()
    type_list = ["OAV", "TV", "special", "movie", "ONA"]
    titles = list(csv.reader(open(titlepath, 'r', encoding='utf-8')))
    i = 0
    for t in titles[1:]:
        info = t[0].split('|')
        ID = info[0]
        
        
        
        lmed = info[1].rfind('(') + 1
        rmed = info[1].rfind(')')
        platform = info[1][lmed:rmed]
        for t in type_list:
            if t in platform:
                platform = t        
#        print(platform)
        title = info[1][:lmed-2]
#        print(i)
        i += 1
        data = (title, platform, type_list.index(platform))
        title_dict[ID] = data        
    return title_dict

def get_summaries():
    summary_dict = dict()
    with open(summarypath, encoding='utf-8') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            full_data = ''.join(row)
            info = full_data.split('|')
            ID = info[0]
            summary = info[1].replace("\"", "")
            summary_dict[ID] = summary
    return summary_dict

def get_staff():
    staff_list = dict()
    staff = list(csv.reader(open(staffpath, 'r', encoding='utf-8')))        
    for s in staff:
        info = s[0].split('|')
        ID = info[0]
        data = info[1].split(';')
        staff_list[ID] = data
    return staff_list

def get_ids():
    ids = list()
    x = list(csv.reader(open(genrepath, 'r')))
    for o in x:
        info = o[0].split('|')
        ID = info[0]
        if ID != 'Anime_ID':
            ids.append(ID)
    return ids
        
def normalize_scores():
    """
    Returns the tag for each anime to be used in cross entropy loss
    """
    tiers = ['Masterpiece', 'Excellent', 'Very Good', 'Good', 'Decent',
             'So-so', 'Not Very Good', 'Weak', 'Bad', 'Awful', 'Worst Ever'] 

    tag_dict = dict()
    scores = get_scores()
    
    for ID in scores.keys():
        values = scores[ID]
        high = max(values)
        loc = values.index(high)
        
        tag_dict[ID] = (loc, tiers[loc])
        
    return tag_dict

def encode_genres():
    voids = ["Genre", "genres"]
    genre_map = list()
    items = get_genres()
    for ID in items.keys():
        genres = items[ID]
        for genre in genres:
            if genre not in genre_map:
                if genre not in voids and len(genre) > 0:
                    genre_map.append(genre)
    return sorted(set(list(genre_map)))

def clean_genres():
    g = encode_genres()
    initial = len(g)
   
    new_list = list()
    for genre in g:
        new_list.append(genre.lower())
    nw = sorted(set(new_list))
    
    for word in nw:
        for others in nw:
            if others == str(word + 's'):
                nw.pop(nw.index(word))
    print("Removed {} redundant genres".format(str(initial-len(nw))))
    
    return nw

def fix_genre_data():
    valid_genres = clean_genres()
    
    

        
        
if __name__ == "__main__":
    clean_genres()
    
    
    
    
    
    
    
    
    # np.save("genre_array.npy", np.char.array(g))
    
    
    
    
