#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 13:56:21 2018

@author: nehaj1993
"""

import json
import requests
import urllib
import argparse
from random import *

color_dict = {"male":["skinColor", "hairColor", "eyeColor"],
            "female":["skinColor", "hairColor", "lipstickColor", "eyeColor"]}
attributes_dict = {"male":["eyebrows", "eyes", "nose", "mouth", "ears", "facialHair", "glasses", "jaw", "hair"],
              "female":["eyebrows", "eyes", "nose", "mouth", "ears", "glasses", "jaw", "hair", "eyelashes"]}

def generateColors(data, gender):
    colors = "{"
    # for i in range(len(data[gender]["colors"])):
    for i in range(len(color_dict[gender])):
        color = color_dict[gender][i]
        if color == "lipstickColor" and randint(0,1) == 0:
            continue;
        index = randint(0, len(data[gender][color_dict[gender][i]])-1)
        color_val = data[gender][color_dict[gender][i]][index]
        for key, value in color_val.iteritems():
            colors += "\"" + key + "\":" + str(value)
            colors += ","
    colors = colors[:-1]
    colors += "}"
    return colors

def generateAttributes(data, gender):
    attributes = "{"
    for i in range(len(attributes_dict[gender])):
        attribute = attributes_dict[gender][i]
        if attribute == "facialHair" and randint(0,9) < 6:
            continue;
        if attribute == "glasses" and randint(0,9) < 8:
            continue;
        index = randint(0, len(data[gender][attributes_dict[gender][i]])-1)
        attribute_val = data[gender][attributes_dict[gender][i]][index]
        for key, value in attribute_val.iteritems():
            attributes += "\"" + key + "\":\"" + str(value) + "\""
            attributes += ","
    attributes = attributes[:-1]
    attributes += "}"
    return attributes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, help="month for dataset")
    args = parser.parse_args()
    i = 0
    while i < args.num:
        with open('./bitmoji/bitmoji_properties.json') as json_file:
              data = json.load(json_file)
              count = 0
              gender = randint(1,2)
              gender_str = "male" if gender == 1 else "female"
              user_id = '371434407_1_s1' if gender_str == "male" else '122369401_1_s1'
              colors = generateColors(data, gender_str)
              attributes = generateAttributes(data, gender_str)
              proportion = str(randint(0, 8))
              request = 'https://render.bitstrips.com/render/6688424/' + user_id + '-v1.png?colours=' + colors + '&pd2=' + attributes + '&head_rotation=0&proportion='+ str(proportion) +'&sex=' + str(gender) + '&scale=0.382&style=1'
              r = requests.get(request, allow_redirects=True)
              if r.status_code == 200:
                urllib.urlretrieve(request, './data/emojis/{}.png'.format(i))
                print("i:", i, "gender:", gender)
                i = i+1
