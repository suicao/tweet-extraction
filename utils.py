import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm 
import regex as re
import html
import operator
from fuzzywuzzy import fuzz
import itertools
from scipy.special import softmax

def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    try:
        return float(len(c)) / (len(a) + len(b) - len(c))
    except:
        return 0

def fuzzy_match(x,y,weights=None):
    l1 = len(x.split())
    matches = dict()
    x_ = x.split()
    if type(y) is str:
        y = [y]
    for curr_length in range(l1 + 1):
#        if curr_length <= 0:
#            continue
        for i in range(l1 + 1 - curr_length):
            sub_x = ' '.join(x_[i:i+curr_length])
            if sub_x not in matches:
                matches[sub_x] = np.average([fuzz.ratio(sub_x,y_) for y_ in y],weights=weights)
#                matches[sub_x] = jaccard(sub_x,y)
    if len(matches) == 0:
        return None, x
    return matches, sorted(matches.items(), key=operator.itemgetter(1))[-1][0]

def get_sub_idx(x, y): 
    l1, l2 = len(x), len(y) 
    truncate = 0
    while truncate < 3:
        y_ = y[truncate:] 
        if len(y_) == 0:
            break
        for i in range(l1): 
            if x[i:i+len(y_)] == y_: 
                return i, i + len(y_) #- 1
        if truncate > 0:
            y_ = y[:-truncate] 
            for i in range(l1): 
                if x[i:i+len(y_)] == y_: 
                    return i, i + len(y_) #- 1
            y_ = y[truncate:-truncate]
            for i in range(l1): 
                if x[i:i+len(y_)] == y_: 
                    return i, i + len(y_) #- 1
        truncate += 1
    return 0,l1

def find_best_combinations(start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, valid_start= 0, valid_end=512):
    best = (valid_start, valid_end - 1)
    best_score = -9999
#    print(valid_end, start_top_index, end_top_index)
    for i in range(len(start_top_log_probs)):
        for j in range(end_top_log_probs.shape[0]):
            if valid_start <= start_top_index[i] < valid_end and valid_start <= end_top_index[j,i] < valid_end and start_top_index[i] < end_top_index[j,i]:
                score = start_top_log_probs[i] * end_top_log_probs[j,i]
                if score > best_score:
                    best = (start_top_index[i],end_top_index[j,i])
                    best_score = score
    return best

special_tokens = {"positive": "[POS]", "negative":"[NEG]", "neutral": "[NTR]"}

def convert_lines(tokenizer, df, max_sequence_length = 512):
    pad_token_idx = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    outputs = np.zeros((len(df), max_sequence_length))
    type_outputs = np.zeros((len(df), max_sequence_length))
    position_outputs = np.zeros((len(df), 2))
    extracted = []
    for idx, row in tqdm(df.iterrows(), total=len(df)): 
#        input_ids_0 = tokenizer.encode(special_tokens[row.sentiment], add_special_tokens=False) 
        input_ids_0 = tokenizer.encode(row.sentiment.strip(), add_special_tokens=False) 
        input_ids_1 = tokenizer.encode(row.text, add_special_tokens=False) 
        input_ids = [tokenizer.cls_token_id, ]+ input_ids_0 +  [tokenizer.sep_token_id,] + input_ids_1 + [tokenizer.sep_token_id, ]
        token_type_ids = [0,]*(len(input_ids_0) + 1) + [1,]*(len(input_ids_1) + 2)

        if len(input_ids) > max_sequence_length: 
#            input_ids = input_ids[:max_sequence_length//2] + input_ids[-max_sequence_length//2:] 
            input_ids = input_ids[:max_sequence_length]
            input_ids[-1] = tokenizer.sep_token_id
            token_type_ids = token_type_ids[:max_sequence_length]
        else:
            input_ids = input_ids + [pad_token_idx, ]*(max_sequence_length - len(input_ids))
            token_type_ids = token_type_ids + [pad_token_idx, ]*(max_sequence_length - len(token_type_ids))
        assert len(input_ids) == len(token_type_ids)
        outputs[idx,:max_sequence_length] = np.array(input_ids)
        type_outputs[idx,:] = token_type_ids
        selected_text = row.selected_text.strip()
        if " "+selected_text in row.text:
            input_ids_2 = tokenizer.encode(" "+selected_text, add_special_tokens=False)
        else:
            input_ids_2 = tokenizer.encode(selected_text, add_special_tokens=False)
        start_idx, end_idx = get_sub_idx(input_ids_1, input_ids_2)
        extracted.append(tokenizer.decode(input_ids_1[start_idx:end_idx]))
        position_outputs[idx,:] = [start_idx + len(input_ids_0) + 2, end_idx + len(input_ids_0) + 2]
#    df["extracted"] = extracted
#    df.to_csv("hmm.csv",index=False)
    return outputs, type_outputs, position_outputs

def convert_lines_xlnet(tokenizer, df, max_sequence_length = 512):
    pad_token_idx = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    outputs = np.zeros((len(df), max_sequence_length))
    type_outputs = np.zeros((len(df), max_sequence_length))
    position_outputs = np.zeros((len(df), 2))
    extracted = []
    for idx, row in tqdm(df.iterrows(), total=len(df)): 
#        input_ids_0 = tokenizer.encode(special_tokens[row.sentiment], add_special_tokens=False) 
        input_ids_0 = tokenizer.encode(row.sentiment.strip(), add_special_tokens=False) 
        input_ids_1 = tokenizer.encode(row.text, add_special_tokens=False) 
        input_ids =  input_ids_0 +  [tokenizer.sep_token_id, ] + input_ids_1 + [tokenizer.sep_token_id, ] + [tokenizer.cls_token_id, ]
        token_type_ids = [0,]*(len(input_ids_0) + 1) + [1,]*(len(input_ids_1) + 2)

        if len(input_ids) > max_sequence_length: 
#            input_ids = input_ids[:max_sequence_length//2] + input_ids[-max_sequence_length//2:] 
            input_ids = input_ids[:max_sequence_length]
            input_ids[-2] = tokenizer.sep_token_id
            input_ids[-1] = tokenizer.cls_token_id
            token_type_ids = token_type_ids[:max_sequence_length]
        else:
            input_ids = input_ids + [pad_token_idx, ]*(max_sequence_length - len(input_ids))
            token_type_ids = token_type_ids + [pad_token_idx, ]*(max_sequence_length - len(token_type_ids))
        assert len(input_ids) == len(token_type_ids)
        outputs[idx,:max_sequence_length] = np.array(input_ids)
        type_outputs[idx,:] = token_type_ids
        selected_text = row.selected_text.strip()
        if " "+selected_text in row.text:
            input_ids_2 = tokenizer.encode(" "+selected_text, add_special_tokens=False)
        else:
            input_ids_2 = tokenizer.encode(selected_text, add_special_tokens=False)
        start_idx, end_idx = get_sub_idx(input_ids_1, input_ids_2)
        extracted.append(tokenizer.decode(input_ids_1[start_idx:end_idx]))
        position_outputs[idx,:] = [start_idx + len(input_ids_0) + 1, end_idx + len(input_ids_0) + 1]
#    df["extracted"] = extracted
#    df.to_csv("extracted.csv",index=False)
    type_outputs[outputs == tokenizer.cls_token_id] = 2
    type_outputs[outputs == pad_token_idx] = 4
    return outputs, type_outputs, position_outputs

def convert_lines_cls(tokenizer, df, max_sequence_length = 512):
    pad_token_idx = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    outputs = np.zeros((len(df), max_sequence_length))
    type_outputs = np.zeros((len(df), max_sequence_length))
    for idx, row in tqdm(df.iterrows(), total=len(df)): 
        input_ids = tokenizer.encode(row.text, add_special_tokens=False) 
        input_ids = [tokenizer.cls_token_id, ]+ input_ids +  [tokenizer.sep_token_id, ] 
        token_type_ids = [0,]*(len(input_ids)) 

        if len(input_ids) > max_sequence_length: 
#            input_ids = input_ids[:max_sequence_length//2] + input_ids[-max_sequence_length//2:] 
            input_ids = input_ids[:max_sequence_length]
            input_ids[-1] = tokenizer.sep_token_id
            token_type_ids = token_type_ids[:max_sequence_length]
        else:
            input_ids = input_ids + [pad_token_idx, ]*(max_sequence_length - len(input_ids))
            token_type_ids = token_type_ids + [pad_token_idx, ]*(max_sequence_length - len(token_type_ids))
        if len(input_ids) != len(token_type_ids):
            print(input_ids, token_type_ids)
        outputs[idx,:max_sequence_length] = np.array(input_ids)
        type_outputs[idx,:] = token_type_ids
    return outputs, type_outputs


