from utils import *
import csv

def load_data(csv_filename):
    with open(csv_filename) as csv_file:
        items_data = csv.reader(csv_file)

        headers = next(items_data)

        data = []
        for item_data in items_data:
            data.append(item_data[1:])
            vprint(item_data);
    
    return data