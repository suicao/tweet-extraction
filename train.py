import os

import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import json
import numpy as np
import pickle

import os
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--train_full', action='store_true')
parser.add_argument('--holdout', action='store_true')
parser.add_argument('--stratified', action='store_true')
parser.add_argument('--extra_sentiment', action='store_true')
parser.add_argument('--pl', action='store_true')
parser.add_argument('--devices', default='7')
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--beam_size', type=int, default=5)
parser.add_argument('--max_sequence_length', type=int, default=128)
parser.add_argument('--accumulation_steps', type=int, default=1)

parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--seed', type=int, default=13)
#parser.add_argument('--pretrained_path', type=str, default="../zalo-bert/output/checkpoint-43000/pytorch_model.bin")
parser.add_argument('--pretrained_path', type=str, default=None)
parser.add_argument('--model', type=str, default="roberta")

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

from torch import nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import *
import torch
import torch.utils.data
import torch.nn.functional as F
from torch.optim import Adagrad, Adamax
from transformers.modeling_utils import * 
from scipy.stats import spearmanr
from utils import *
from models import *

SEED = args.seed + args.fold
EPOCHS = args.epochs 

lr=args.lr
batch_size = args.batch_size
accumulation_steps = args.accumulation_steps
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

if args.model == "roberta":
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSentimentExtraction.from_pretrained('roberta-base', output_hidden_states=True)
if args.model == "bertweet":
    tokenizer = BERTweetTokenizer(pretrained_path = './bertweet/')
    config = RobertaConfig.from_pretrained("./bertweet/config.json")
    model = RobertaForSentimentExtraction.from_pretrained('./bertweet/model.bin', config=config)
if args.model == "longformer":
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    model = LongformerForSentimentExtraction.from_pretrained('allenai/longformer-base-4096', output_hidden_states=True)
if args.model == "bart-large":
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    model = BartForSentimentExtraction.from_pretrained('facebook/bart-large', output_hidden_states=True)
if args.model == "bart":
    tokenizer = BartTokenizer.from_pretrained('./bart-configs')
    model = BartForSentimentExtraction.from_pretrained('./bart-configs', output_hidden_states=True)
elif args.model == "xlnet":
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    model = XLNetForSentimentExtraction.from_pretrained('xlnet-base-cased', output_hidden_states=True)
elif args.model == "xlmr":
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    model = XLMRobertaForSentimentExtraction.from_pretrained('xlm-roberta-base', output_hidden_states=True)
elif args.model == "xlmr-large":
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
    model = XLMRobertaForSentimentExtraction.from_pretrained('xlm-roberta-large', output_hidden_states=True)
elif args.model == "roberta-large":
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    model = RobertaForSentimentExtraction.from_pretrained('roberta-large', output_hidden_states=True)

model.cuda()

train_df = pd.read_csv("data/train.csv") if not args.holdout else pd.read_csv("data/train_holdout.csv")
if args.pl:
    print("Using pseudo labeling")
    test_df = pd.read_csv(f"./pseudo_labels/{args.model}_{args.fold}_{args.seed}.csv")
    train_df = pd.concat([train_df,test_df]).reset_index()
    train_df = train_df[["textID","text","selected_text","sentiment"]]
train_df.fillna("NaN",inplace=True)
if args.extra_sentiment:
    extra_df = pd.read_csv("data/tweet_dataset.csv").set_index("aux_id")
    for idx, row in tqdm(train_df.iterrows(),total=len(train_df)):
        if row.textID not in extra_df.index:
            print(row.textID)
            continue
    train_df.loc[idx, "sentiment"] = row.sentiment + " " + extra_df.loc[row.textID].sentiment

if os.path.exists(f"./data/X_train_{args.model}.npy") and (not args.pl):
    print("Loading cached input ids")
    X_train, X_type_train, X_pos_train, X_offset_train = \
        np.load(f"./data/X_train_{args.model}.npy"), np.load(f"./data/X_type_train_{args.model}.npy"), np.load(f"./data/X_pos_train_{args.model}.npy"), np.load(f"./data/X_offset_train_{args.model}.npy")
else:
    print("Creating input ids")
    if args.model not in ["xlnet","xlnet-large"]:
        X_train, X_type_train, X_pos_train, X_offset_train = convert_lines_v2(tokenizer, train_df, max_sequence_length=args.max_sequence_length)
    else:
        X_train, X_type_train, X_pos_train = convert_lines_xlnet(tokenizer, train_df, max_sequence_length=args.max_sequence_length)
    np.save(f"./data/X_train_{args.model}.npy", X_train) 
    np.save(f"./data/X_type_train_{args.model}.npy", X_type_train)
    np.save(f"./data/X_pos_train_{args.model}.npy", X_pos_train)
    np.save(f"./data/X_offset_train_{args.model}.npy", X_offset_train)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
   ]

num_train_optimization_steps = int(EPOCHS*len(train_df)/batch_size/accumulation_steps)

optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=False) 
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*num_train_optimization_steps, num_training_steps=num_train_optimization_steps) 
scheduler0 = get_constant_schedule(optimizer)

if not args.stratified:
    splits = list(KFold(n_splits=5, shuffle=True, random_state=args.seed).split(X_train, np.arange(len(X_train))))
else:
    print("Using stratified KFold")
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed).split(X_train, y_train))

for fold, (train_idx, val_idx) in enumerate(splits):
    if fold != args.fold:
        continue
    print("Training for fold {}".format(fold))
    if args.pretrained_path is not None:
        print(f"Loading pretrained checkpoint from {args.pretrained_path}")
        model.load_state_dict(torch.load(args.pretrained_path),strict=False)
    if args.train_full:
        train_idx = np.arange(len(X_train))
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train[train_idx],dtype=torch.long),torch.tensor(X_type_train[train_idx],dtype=torch.long),\
                torch.tensor(X_pos_train[train_idx, 0],dtype=torch.long), torch.tensor(X_pos_train[train_idx, 1],dtype=torch.long), torch.tensor(X_offset_train[train_idx],dtype=torch.long))
    valid_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train[val_idx],dtype=torch.long), torch.tensor(X_type_train[val_idx],dtype=torch.long),\
                torch.tensor(X_pos_train[val_idx, 0],dtype=torch.long), torch.tensor(X_pos_train[val_idx,1],dtype=torch.long),torch.tensor(X_offset_train[val_idx],dtype=torch.long))
    best_score = 0
    tq = tqdm(range(args.epochs + 1))
    model.freeze()
    frozen = True
    for epoch in tq:
        if epoch > 0 and frozen:
            model.unfreeze()
            frozen = False
            del scheduler0
            torch.cuda.empty_cache()

        val_preds = None
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        optimizer.zero_grad()
        pbar = tqdm(enumerate(train_loader),total=len(train_loader),leave=False)
        model.train()
        for i, items in pbar:
            x_batch, x_type_batch, x_start_batch, x_end_batch,x_offset_batch = items
            p_mask = torch.zeros(x_batch.shape,dtype=torch.float32)
            p_mask[x_batch == tokenizer.pad_token_id] = 1.0
            p_mask[x_batch == tokenizer.cls_token_id] = 1.0
            if args.model in ["xlnet","xlnet-large"]:
                attention_mask=(x_batch != tokenizer.pad_token_id).float().cuda()
            else:
                attention_mask=(x_batch != tokenizer.pad_token_id).cuda()
            for i_ in range(len(x_batch)):
                p_mask[i_, :x_offset_batch[i_]] = 1.0
            loss = model(input_ids=x_batch.cuda(), start_positions = x_start_batch.cuda(), end_positions = x_end_batch.cuda(), \
                                    attention_mask=attention_mask, token_type_ids=x_type_batch.cuda(),p_mask=p_mask.cuda()) #,cls_ids=y_batch)
            loss /= accumulation_steps
            loss = loss.mean()
            loss.backward()
            if i % accumulation_steps == 0 or i == len(pbar) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if not frozen:
                    scheduler.step()
            pbar.set_postfix(loss = loss.item()*accumulation_steps)
        torch.save(model.state_dict(),"models/{}_{}_{}.bin".format(args.model, fold,args.seed))
        if args.train_full or epoch < args.epochs:
            continue
        model.load_state_dict(torch.load(("models/{}_{}_{}.bin".format(args.model, fold,args.seed))))
        model.eval()
        true_texts = train_df.loc[val_idx].selected_text.values
        selected_texts = []
        start_end_idxs = []
        # with torch.no_grad():
        pbar = tqdm(enumerate(valid_loader),total=len(valid_loader),leave=False)
        for i, items in pbar:
            x_batch, x_type_batch, x_start_batch, x_end_batch,x_offset_batch = items
            p_mask = torch.zeros(x_batch.shape,dtype=torch.float32)
            p_mask[x_batch == tokenizer.pad_token_id] = 1.0
            p_mask[x_batch == tokenizer.cls_token_id] = 1.0
            for i_ in range(len(x_batch)):
                p_mask[i_, :x_offset_batch[i_]] = 1.0
            if args.model in ["xlnet","xlnet-large"]:
                attention_mask=(x_batch != tokenizer.pad_token_id).float().cuda()
            else:
                attention_mask=(x_batch != tokenizer.pad_token_id).cuda()
            start_top_log_probs, start_top_index, end_top_log_probs, end_top_index = model(input_ids=x_batch.cuda(), attention_mask= attention_mask, \
                                                                                            token_type_ids=x_type_batch.cuda(),p_mask = p_mask.cuda(), beam_size=args.beam_size)
            start_top_log_probs = start_top_log_probs.detach().cpu().numpy()
            end_top_log_probs = end_top_log_probs.detach().cpu().numpy()
            start_top_index = start_top_index.detach().cpu().numpy()
            end_top_index = end_top_index.detach().cpu().numpy()
            for i_, x in enumerate(x_batch):
                x = x.numpy()
                real_length = np.sum(x != tokenizer.pad_token_id)
                valid_start = x_offset_batch[i_]
                if args.model in ["xlnet","xlnet-large"]:
                    real_length -= 1
                best_start, best_end = find_best_combinations(start_top_log_probs[i_], start_top_index[i_], \
                                                                end_top_log_probs[i_].reshape(args.beam_size,args.beam_size), end_top_index[i_].reshape(args.beam_size,args.beam_size), \
                                                                valid_start = valid_start, valid_end = real_length)
                selected_text = tokenizer.decode([w for w in x[best_start:best_end] if w != tokenizer.pad_token_id], clean_up_tokenization_spaces=False)
                selected_texts.append(selected_text.strip())
        scores = []
        scores = [jaccard(str1,str2) for str1, str2 in zip(true_texts, selected_texts)]
        matched_texts = [y if y in x else fuzzy_match(x,y)[1] for x,y in zip(train_df.loc[val_idx].text.values, selected_texts)]
#        train_df.loc[val_idx, "selected_text"] = selected_texts
#        train_df.loc[val_idx, "matched_text"] = matched_texts
#        train_df.loc[val_idx][["textID","selected_text","matched_text"]].to_csv(f"preds_dieter/{args.model}_{fold}.csv",index=False)
        matched_scores = [jaccard(str1,str2) for str1, str2 in zip(true_texts, matched_texts)]
        bugged_preds = []
        bugged_labels = []
        for x,y,z in zip(train_df.loc[val_idx].text.values, true_texts, matched_texts):
            if "  " in x:
                bugged_labels.append(y)
                bugged_preds.append(z)
        bugged_scores = [jaccard(str1,str2) for str1, str2 in zip(bugged_labels, bugged_preds)]
        print(f"\nFold {fold}, Raw score = {np.mean(scores):.4f}, matched score = {np.mean(matched_scores):.4f}")
#        break
