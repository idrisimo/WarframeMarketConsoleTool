from pprint import pprint
import pandas as pd
import re
import requests

def format_pytesseract_as_data(data):
    df = pd.DataFrame(data)
    line = []
    sentence = []
    top = 0
    left = 0
    for index, row in df.iterrows():
        if row['word_num'] != 0:
            line.append(row['text'])
        else:
            top = row['top']
            left = row['left']
            sentence.append({'text':' '.join(line), 'left':left , 'top':top })
            line = []
    sentence = [obj for obj in sentence if obj['text'] != '']
    return sentence

def get_brackets(line):
    # items_brackets = re.findall(r'\[[^\]]*\]', line)
    items_brackets = re.findall(r'\[(.*?)\]', line)
    return items_brackets


def get_market_price(item):
    item_cleaned = item.lower().replace(' ', '_')
    if item_cleaned.endswith('prime'):
        item_cleaned = item_cleaned + '_set' 
    headers = {'Platform': 'xbox', 'accept': 'application/json'}
    url = f"https://api.warframe.market/v1/items/{item_cleaned}/orders"
    r = requests.get(url, headers=headers)
    orders = r.json()['payload']['orders']
    platinum_for_orders = [order['platinum'] for order in orders]
    average_plat_price = round(sum(platinum_for_orders) / len(platinum_for_orders))
    return average_plat_price


