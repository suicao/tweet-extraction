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
parser.add_argument('--pre', action='store_true')
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
    model = RobertaSentimentClassification.from_pretrained('roberta-base', output_hidden_states=True, num_labels=3 if not args.pre else 1)
model.cuda()

if not args.pre:
    train_df = pd.read_csv("data/train.csv")
    train_df.fillna("NaN",inplace=True)
    sent_dict = {"negative": 0, "neutral": 1, "positive": 2}
    y_train = train_df.sentiment.apply(lambda x: sent_dict[x]).values 
else:
    cols = ["sentiment", "ids", "date", "flag", "user", "text"]
    train_df = pd.read_csv("data/sent140.csv",names=cols,encoding="ISO-8859-1")
    train_df.fillna("NaN",inplace=True)
    y_train = train_df.sentiment.values // 4
if not args.pre:
    X_train, X_type_train = convert_lines_cls(tokenizer, train_df, max_sequence_length=args.max_sequence_length)
else:
    X_train, X_type_train = convert_lines_cls(tokenizer, train_df, max_sequence_length=args.max_sequence_length)
    np.save("data/pre_sent_X.npy",X_train)
    np.save("data/pre_sent_X_type.npy",X_type_train)
    
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

if args.model in ["bart"]:
    tsfm = model.model
if args.model in ["longformer"]:
    tsfm = model.longformer
if args.model in ["xlm","xlnet","gpt2","xlnet-large"]:
    tsfm = model.transformer
if args.model in ["bert","bert-mrpc","bert-large","bert-large-whole-word-masking"]:
    tsfm = model.bert
if args.model in ["albert"]:
    tsfm = model.albert
if args.model in ["roberta","xlmr","xlmr-large","roberta-detector","roberta-large","roberta-squad","distilroberta"]:
    tsfm = model.roberta

splits = list(KFold(n_splits=5 if not args.pre else 10, shuffle=True, random_state=args.seed).split(X_train, np.arange(len(X_train))))
for fold, (train_idx, val_idx) in enumerate(splits):
    if fold != args.fold:
        continue
    print("Training for fold {}".format(fold))
    if args.pretrained_path is not None:
        print(f"Loading from {args.pretrained_path}")
        state_dict = torch.load(args.pretrained_path)
        state_dict.pop("qa_outputs.weight",None)
        state_dict.pop("qa_outputs.bias",None)
        model.load_state_dict(state_dict,strict=False)
    if args.train_full:
        train_idx = np.arange(len(X_train))
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train[train_idx],dtype=torch.long),torch.tensor(X_type_train[train_idx],dtype=torch.long),\
                torch.tensor(y_train[train_idx],dtype=torch.long))
    valid_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train[val_idx],dtype=torch.long), torch.tensor(X_type_train[val_idx],dtype=torch.long),\
                torch.tensor(y_train[val_idx],dtype=torch.long))
    tq = tqdm(range(args.epochs + 1))
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
            x_batch, x_type_batch, y_batch = items
            if args.model in ["xlnet","xlnet-large"]:
                attention_mask=(x_batch != tokenizer.pad_token_id).float().cuda()
            else:
                attention_mask=(x_batch != tokenizer.pad_token_id).cuda()
            logits = model(input_ids=x_batch.cuda(), attention_mask=attention_mask, token_type_ids=x_type_batch.cuda()) #,cls_ids=y_batch)
            if not args.pre:
                loss = torch.nn.CrossEntropyLoss()(F.softmax(logits), y_batch.cuda())
            else:
                loss = F.binary_cross_entropy_with_logits(logits.view(-1).cuda(),y_batch.float().cuda())
            loss /= accumulation_steps
            loss = loss.mean()
            loss.backward()
            if i % accumulation_steps == 0 or i == len(pbar) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if not frozen:
                    scheduler.step()
            pbar.set_postfix(loss = loss.item()*accumulation_steps)
        torch.save(model.state_dict(),f"models/sent_{args.model}_{fold if not args.pre else 'pre'}_{args.seed}.bin")
        if args.train_full: #or epoch < args.epochs:
            continue
        model.load_state_dict(torch.load((f"models/sent_{args.model}_{fold if not args.pre else 'pre'}_{args.seed}.bin")))
        model.eval()
        pbar = tqdm(enumerate(valid_loader),total=len(valid_loader),leave=False)
        all_preds = None
        for i, items in pbar:
            x_batch, x_type_batch, y_batch = items
            if args.model in ["xlnet","xlnet-large"]:
                attention_mask=(x_batch != tokenizer.pad_token_id).float().cuda()
            else:
                attention_mask=(x_batch != tokenizer.pad_token_id).cuda()
            logits = model(input_ids=x_batch.cuda(), attention_mask= attention_mask, token_type_ids=x_type_batch.cuda())
            logits = logits.detach().cpu().numpy()
            all_preds = logits if all_preds is None else np.concatenate([all_preds, logits])
        if not args.pre:
            all_preds = np.argmax(all_preds,axis=1)
        else:
            all_preds = all_preds > 0.5
        score = f1_score(y_train[val_idx], all_preds, average="macro")
        score2 = f1_score(y_train[val_idx], all_preds, average="micro")
        print(f"Epoch {fold}, F1 macro = {score:.4f}, accuracy = {score2:.4f}")
