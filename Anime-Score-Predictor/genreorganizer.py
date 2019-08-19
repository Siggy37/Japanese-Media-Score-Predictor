#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 21:05:23 2019

@author: siggy
"""
import animedata as ad
import numpy as np

genres = ad.clean_genres()
print(genres)
for g in genres:
    template = "<option value={} >{}</option>".format("\""+g+"\"",g)
    print(template)