import re
import csv
from pprint import pprint
import numpy
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from decimal import Decimal


def extract_dates(text):
    ignore_values = '(?!7/11)'
    prefix_days = '(?:\d{1,2}|[fF]ir|[sS]eco|[tT]hi)(?:st|nd|rd|th)'
    months = '(?:[jJ]an[\.]?(?:uary)?|[Ff]eb(?:uary)?|[Mm]ar[\.]?(?:ch)?|[Aa]pr[\.]?(?:il)?|[Mm]ay|[jJ]un[\.]?(?:e)?|[jJ]ul[\.]?(?:y)?|[Aa]ug[\.]?(?:ust)?|[Ss]ept[\.]?(?:ember)?|[Oo]ct[\.]?(?:ober)?|[Nn]ov[\.]?(?:ember)?|[Dd]ec[\.]?(?:ember)?)'
    suffix_days = '(?:\d{1,2}|[fF]ir|[sS]eco|[tT]hi)(?:st|nd|rd|th)?)'
    year = '(?:(?:\d{2})|(?:\d{4}))?)'
    calendar_formating = '(?:\d{1,2}[\.\/]\d{1,2}(?:[\.\/](?:\d{2})|(?:\d{4}))?)'
    temporal_adverbs = '[nN]ow|[iI]mmediately|[aA]sap|ASAP'
    dates = re.findall(r'%s\b((?:%s[\.\s\/])?%s{1}(?:[\.\s\/]%s?(?:[\.\/\s]%s?|%s|%s)\b' % (ignore_values,prefix_days,months,suffix_days,year,calendar_formating,temporal_adverbs), text)
    return dates

def extract_prices(text):
    prefix_pattern = '(?:(?:(?:(?:[rR]ent|[dD]eposit)(?:(?:\:\s)|(?:\sis\s))?)|(?:[dD]eposit|\$|[dD]ollars?|[eE]uros[dD]eposit|[rR]ent))\s*)'
    currency_value_pattern = '(\d[\d\,\.\-\$]*\d)'
    suffix_pattern = '(?:(?:\$|[dD]ollars?|[rR]ent|[dD]eposit|[eE]uros?|\s?(?:\/|a|per)\s?month))'
    prices = re.findall(r'%s*%s%s+|%s+%s%s*' % (prefix_pattern,currency_value_pattern,suffix_pattern,prefix_pattern,currency_value_pattern,suffix_pattern), text)
    return prices

if __name__== "__main__":

    data_set = pd.read_csv('data/nlp_assignment_dataset.csv', header=0)

    for index, post in data_set.iterrows():
        pprint(extract_prices(post['text']))
        pprint(extract_dates(post['text']))
