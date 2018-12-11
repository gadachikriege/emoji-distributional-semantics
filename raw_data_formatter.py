import pandas as pd
import numpy as np
import emoji
import string
import csv


'''
The following methods are only used to clean, format, and save the original data and should not be run.

    The cleaned data can be found in emoji_datasets/all_data.csv
'''

# Clean data
def load_test_data(file_path):
    with open(file_path) as fp:
        result = []
        translator = str.maketrans('', '', string.punctuation)
        line = fp.readline()
        while line:
            line = line.strip()
            line = line.split(',',1)
            if len(line) == 2:
                clean_row1 = line[1].translate(translator)
                clean_row2 = clean_row1.replace(chr(8220),'')
                clean_row3 = clean_row2.replace(chr(8221),'')
                line = [line[0], clean_row3]
                value = np.array([line[0], line[1]])
                result.append(value)
            line = fp.readline()
        return pd.DataFrame(np.array(result, dtype='object'))

def extract_emojis(example):
    return (' '.join(c for c in example if c in emoji.UNICODE_EMOJI)).split()

def prune_dataset_emojis(data):
    result = []
    translator = str.maketrans('', '', string.punctuation)
    for i,row in enumerate(data):
        try:
            if extract_emojis(row[1]) != []:
                clean_row1 = row[1].translate(translator)
                clean_row2 = clean_row1.replace(chr(8220),'')
                clean_row3 = clean_row2.replace(chr(8221),'')
                new_row = np.array([row[0], clean_row3])
                result.append(new_row)
        except TypeError:
            pass
    return pd.DataFrame(np.array(result))


# Clean data and write to CSV
train_data_raw = pd.read_csv('emoji_datasets/data_train_RAW.csv', header=None, encoding='utf-8')

test_data_raw = load_test_data('emoji_datasets/data_test_RAW.txt')

train_data_clean = prune_dataset_emojis(train_data_raw.values)
test_data_clean = prune_dataset_emojis(test_data_raw.values)

all_data_clean_np = np.vstack((train_data_clean.values, test_data_clean.values))
np.random.shuffle(all_data_clean_np)
all_data_clean = pd.DataFrame(all_data_clean_np)

all_data_clean.to_csv('emoji_datasets/all_data.csv', header=None, index=False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
