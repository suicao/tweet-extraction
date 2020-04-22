import os

import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from torch import nn
import json
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import *
import torch
import torch.utils.data
import torch.nn.functional as F
from torch.optim import Adagrad, Adamax
from transformers.modeling_utils import * 
import argparse
from scipy.stats import spearmanr
from utils import *
from models import *

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--train_full', action='store_true')
parser.add_argument('--ignore_neutral', action='store_true')
parser.add_argument('--stratified', action='store_true')
parser.add_argument('--use_only', type=str, default='all')
parser.add_argument('--devices', default='7')
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--beam_size', type=int, default=5)
parser.add_argument('--max_sequence_length', type=int, default=128)
parser.add_argument('--accumulation_steps', type=int, default=1)

parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--stop_after', type=int, default=3)
parser.add_argument('--seed', type=int, default=13)
#parser.add_argument('--pretrained_path', type=str, default="../tweet-extraction/pretrained_models/roberta_0.bin")
parser.add_argument('--pretrained_path', type=str, default=None)
parser.add_argument('--model', type=str, default="roberta")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
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
if args.model == "roberta-squad":
    tokenizer = RobertaTokenizer.from_pretrained("deepset/roberta-base-squad2")
    model = RobertaForSentimentExtraction.from_pretrained("deepset/roberta-base-squad2", output_hidden_states=True)
if args.model == "bart":
    tokenizer = BartTokenizer.from_pretrained('bart-large')
    model = BartForSentimentExtraction.from_pretrained('bart-large', output_hidden_states=True)
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
elif args.model == "roberta-detector":
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large-openai-detector')
    model = RobertaForSentimentExtraction.from_pretrained('roberta-large-openai-detector', output_hidden_states=True)

model.cuda()

train_df = pd.read_csv("data/train.csv")
train_df.fillna("NaN",inplace=True)
#train_df.text = train_df.text.apply(lambda x: x.replace("`","'"))
#train_df.selected_text = train_df.selected_text.apply(lambda x: x.replace("`","'"))
#train_df["orig_selected_text"] = train_df.selected_text.values

print(f"Training on {args.use_only}")
if args.ignore_neutral or args.use_only not in ["all","neutral"]:
    train_df = train_df[train_df.sentiment != "neutral"].reset_index()
    #print(train_df.head())
if args.model not in ["xlnet","xlnet-large"]:
    X_train, X_type_train, X_pos_train = convert_lines(tokenizer, train_df, max_sequence_length=args.max_sequence_length)
else:
    X_train, X_type_train, X_pos_train = convert_lines_xlnet(tokenizer, train_df, max_sequence_length=args.max_sequence_length)

sentiment_dict = {"negative": 0, "neutral": 1, "positive": 2}
y_train = np.array([sentiment_dict[x] for x in train_df.sentiment.values])

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

num_train_optimization_steps = int(EPOCHS*len(train_df)/batch_size/accumulation_steps)

optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=False) 
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*num_train_optimization_steps, num_training_steps=num_train_optimization_steps)  # PyTorch scheduler
scheduler0 = get_constant_schedule(optimizer)
model = nn.DataParallel(model)

if args.model in ["bart"]:
    tsfm = model.module.model
if args.model in ["xlm","xlnet","gpt2","xlnet-large"]:
    tsfm = model.module.transformer
if args.model in ["bert","bert-mrpc","bert-large","bert-large-whole-word-masking"]:
    tsfm = model.module.bert
if args.model in ["albert"]:
    tsfm = model.module.albert
if args.model in ["roberta","xlmr","xlmr-large","roberta-detector","roberta-large","roberta-squad"]:
    tsfm = model.module.roberta

if not args.stratified:
    splits = list(KFold(n_splits=5, shuffle=True, random_state=args.seed).split(X_train, np.arange(len(X_train))))
else:
    print("Using stratified KFold")
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed).split(X_train, y_train))

for fold, (train_idx, val_idx) in enumerate(splits):
    print("Training for fold {}".format(fold))
    if args.pretrained_path is not None:
        print(f"Loading pretrained checkpoint from {args.pretrained_path}")
        model.load_state_dict(torch.load(args.pretrained_path),strict=False)
    if fold != args.fold:
        continue
    if args.train_full:
        train_idx = np.arange(len(X_train))
    if args.use_only != "all":
        train_idx = np.array([idx for idx in train_idx if train_df.loc[idx].sentiment == args.use_only])
        val_idx = np.array([idx for idx in val_idx if train_df.loc[idx].sentiment == args.use_only])
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train[train_idx],dtype=torch.long),torch.tensor(X_type_train[train_idx],dtype=torch.long),\
                torch.tensor(X_pos_train[train_idx, 0],dtype=torch.long), torch.tensor(X_pos_train[train_idx, 1],dtype=torch.long)) #,torch.tensor(y_train[train_idx],dtype=torch.long))
    valid_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train[val_idx],dtype=torch.long), torch.tensor(X_type_train[val_idx],dtype=torch.long),\
                torch.tensor(X_pos_train[val_idx, 0],dtype=torch.long), torch.tensor(X_pos_train[val_idx,1],dtype=torch.long))
    '''
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train[train_idx],dtype=torch.long),torch.tensor(X_type_train[train_idx],dtype=torch.long),\
                torch.tensor(X_start_train[train_idx],dtype=torch.float), torch.tensor(X_end_train[train_idx],dtype=torch.float)) #,torch.tensor(y_train[train_idx],dtype=torch.long))
    valid_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train[val_idx],dtype=torch.long), torch.tensor(X_type_train[val_idx],dtype=torch.long),\
                torch.tensor(X_start_train[val_idx],dtype=torch.float), torch.tensor(X_end_train[val_idx],dtype=torch.float))
    '''
    best_score = 0
    tq = tqdm(range(args.stop_after + 1))
    for child in tsfm.children():
        for param in child.parameters():
            param.requires_grad = False
    frozen = True
    for epoch in tq:
        if epoch > 0 and frozen:
            for child in tsfm.children():
                for param in child.parameters():
                    param.requires_grad = True
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
            x_batch, x_type_batch, x_start_batch, x_end_batch  = items
            p_mask = torch.zeros(x_batch.shape,dtype=torch.float32)
            p_mask[x_batch == tokenizer.pad_token_id] = 1.0
            p_mask[x_batch == tokenizer.cls_token_id] = 1.0

            if args.model in ["xlnet","xlnet-large"]:
                attention_mask=(x_batch != tokenizer.pad_token_id).float().cuda()
                p_mask[:,:2] = 1.0
            else:
                attention_mask=(x_batch != tokenizer.pad_token_id).cuda()
                p_mask[:,:3] = 1.0

            loss = model(input_ids=x_batch.cuda(), start_positions = x_start_batch.cuda(), end_positions = x_end_batch.cuda(), \
                                    attention_mask=attention_mask, token_type_ids=x_type_batch.cuda(),p_mask=None) #,cls_ids=y_batch)
            loss /= accumulation_steps
            loss = loss.mean()
            loss.backward()
            if i % accumulation_steps == 0 or i == len(pbar) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if not frozen:
                    scheduler.step()
            pbar.set_postfix(loss = loss.item()*accumulation_steps)
        torch.save(model.state_dict(),"models/{}_{}_{}.bin".format(args.model, fold,args.use_only))
        if args.train_full or epoch < args.stop_after:
            continue
        model.eval()
#        model.load_state_dict(torch.load("models/{}_{}_{}.bin".format(args.model, fold,args.use_only)))
        true_texts = train_df.loc[val_idx].selected_text.values
        selected_texts = []
        start_end_idxs = []
        # with torch.no_grad():
        pbar = tqdm(enumerate(valid_loader),total=len(valid_loader),leave=False)
        for i, items in pbar:
            x_batch, x_type_batch, x_start_batch, x_end_batch = items
            p_mask = torch.zeros(x_batch.shape,dtype=torch.float32)
            p_mask[x_batch == tokenizer.pad_token_id] = 1.0
            p_mask[x_batch == tokenizer.cls_token_id] = 1.0
            if args.model in ["xlnet","xlnet-large"]:
                attention_mask=(x_batch != tokenizer.pad_token_id).float().cuda()
                p_mask[:,:2] = 1.0
            else:
                attention_mask=(x_batch != tokenizer.pad_token_id).cuda()
                p_mask[:,:3] = 1.0
            start_top_log_probs, start_top_index, end_top_log_probs, end_top_index = model(input_ids=x_batch.cuda(), attention_mask= attention_mask, \
                                                                                            token_type_ids=x_type_batch.cuda(),p_mask = None, beam_size=args.beam_size)
            start_top_log_probs = start_top_log_probs.detach().cpu().numpy()
            end_top_log_probs = end_top_log_probs.detach().cpu().numpy()
            start_top_index = start_top_index.detach().cpu().numpy()
            end_top_index = end_top_index.detach().cpu().numpy()
            for i_, x in enumerate(x_batch):
                x = x.numpy()
                real_length = np.sum(x != tokenizer.pad_token_id)
                valid_start = 3
                if args.model in ["xlnet","xlnet-large"]:
                    real_length -= 1
                    valid_start = 2
                best_start, best_end = find_best_combinations(start_top_log_probs[i_], start_top_index[i_], \
                                                                end_top_log_probs[i_].reshape(args.beam_size,args.beam_size), end_top_index[i_].reshape(args.beam_size,args.beam_size), \
                                                                valid_start = valid_start, valid_end = real_length)
                selected_text = tokenizer.decode([w for w in x[best_start:best_end] if w != tokenizer.pad_token_id])
                selected_texts.append(selected_text.strip())
        scores = []
        count = 0
        scores = [jaccard(str1,str2) for str1, str2 in zip(true_texts, selected_texts)]
        matched_texts = [y if y in x else fuzzy_match(x,y)[1] for x,y in zip(train_df.loc[val_idx].text.values, selected_texts)]
        matched_scores = [jaccard(str1,str2) for str1, str2 in zip(true_texts, matched_texts)]
        print(f"\nRaw score = {np.mean(scores):.4f}, matched score = {np.mean(matched_scores):.4f}")
#        torch.save(model.state_dict(),"models/{}_{}_{}.bin".format(args.model, fold,args.use_only))
