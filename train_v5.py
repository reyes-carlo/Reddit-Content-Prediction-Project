import argparse
import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.cuda.amp as amp
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    AutoModel,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup
)
from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm
import wandb

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr

# ----------------
# DistilBERT or BERT MODEL
# ----------------
class BertForTwoTargetRegression(nn.Module):
    def __init__(
        self,
        pretrained_encoder: AutoModel,
        meta_feat_dim: int,
        num_subs: int,
        sub_embed_dim: int = 8
    ):
        super().__init__()
        self.encoder = pretrained_encoder
        hidden_size = self.encoder.config.hidden_size
        # absolutely useless embedding that made it a lot harder than the one in the last hw assignment
        self.sub_embed = nn.Embedding(num_subs, sub_embed_dim)
        # not enough time, so got desperate and used distilbert, so code helps compatiability with distilbert
        drop_prob = getattr(self.encoder.config, "hidden_dropout_prob", self.encoder.config.dropout)
        self.dropout = nn.Dropout(drop_prob)
        # regressor to predict two targets, num_comments and upvotes
        self.regressor = nn.Linear(hidden_size + meta_feat_dim + sub_embed_dim, 2)

    def forward(self, input_ids, attention_mask, meta_feats, sub_ids):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        cls_emb = self.dropout(cls_emb)
        sub_emb = self.sub_embed(sub_ids)
        x = torch.cat([cls_emb, meta_feats, sub_emb], dim=-1)
        return self.regressor(x)

# ----------------
# META FEATURES (SEEMS TO BE ABSOLUTELY USELESS)
# ----------------
def pack_meta_and_sub(example):
    numeric = [
        example["num_sentences"],
        example["avg_token_len"],
        example["upper_ratio"],
        example["punct_count"],
        example["emoji_count"],
        example["sentiment_score"],
        example["readability_flesch"],
        example["title_len"],
        example["body_len"],
        example["is_question"],
    ]
    example["meta_feats"] = numeric
    return example

# ----------------
# 3) TOKENIZER FUNCTION (WITH TRUNCATION & max_length 256, BUT LETS COLLATOR DO THE PADDING)
# ----------------
def preprocess_function(examples):
    return tokenizer(
        examples["cleaned"],
        truncation=True,
        padding=False,
        max_length=256,
        return_token_type_ids=False,
    )

# ----------------
# TRAIN FUNCTION - MAIN SOURCE OF DEBUGGING ERRORS
# - INTRODUCED DEBUG & SUBSET SLICING OUT OF DESPERATION FOR TIME AND SANITY
# ----------------
def train(args):
    # wandb
    wandb.init(project=args.wandb_project, config=vars(args))

    # load from cache if available. otherwise, we create the tokenized datasets and split them
    if os.path.isdir(args.cache_dir) and os.listdir(os.path.join(args.cache_dir, 'train')):
        print(f"Loading cached datasets from {args.cache_dir}")
        tokenized = DatasetDict.load_from_disk(args.cache_dir)
    else:
        files = glob.glob(os.path.join(args.input_dir, "cleaned_*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No files matching cleaned_*.jsonl in {args.input_dir}")
        dfs = []
        for path in files:
            sub = os.path.basename(path).removeprefix("cleaned_").removesuffix(".jsonl")
            df  = pd.read_json(path, lines=True)
            df["subreddit"] = sub
            dfs.append(df)
        full_df = pd.concat(dfs, ignore_index=True)

        # subreddits to ids like 0, 1, 2, 3 since there are four subreddits
        subs = sorted(full_df["subreddit"].unique())
        sub2id = {s: i for i, s in enumerate(subs)}
        full_df["subreddit_id"] = full_df["subreddit"].map(sub2id).astype(int)

        # dataset and split for train/val/test
        dataset = Dataset.from_pandas(full_df, preserve_index=False)
        splits1 = dataset.train_test_split(test_size=args.test_size, seed=args.seed)
        train_valid, test_ds = splits1['train'], splits1['test']
        val_fraction = args.test_size / (1 - args.test_size)
        splits2 = train_valid.train_test_split(test_size=val_fraction, seed=args.seed)
        train_ds, val_ds = splits2['train'], splits2['test']
        dataset = DatasetDict({'train': train_ds, 'validation': val_ds, 'test': test_ds})

        # add meta features and tokenize the final result, and save it to save time when training multiple models
        dataset = dataset.map(pack_meta_and_sub, remove_columns=[])
        original_cols = dataset['train'].column_names
        keep_cols = {'upvotes', 'num_comments', 'meta_feats', 'subreddit_id'}
        remove_cols = [c for c in original_cols if c not in keep_cols]
        tokenized = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=remove_cols,
            load_from_cache_file=False
        )

        # save a lot of time
        os.makedirs(args.cache_dir, exist_ok=True)
        tokenized.save_to_disk(args.cache_dir)
        print(f"Cached tokenized datasets at {args.cache_dir}")

    # for time-sake, since at the current moment, we only have 15 hours left
    if args.subset_fraction < 1.0:
        for split in ["train", "validation", "test"]:
            ds = tokenized[split].shuffle(seed=args.seed)
            keep_n = int(len(ds) * args.subset_fraction)
            tokenized[split] = ds.select(range(keep_n))

    # for sanity sake, test if the model can run on n examples and 1 epoch
    if args.debug:
        for split in ['train', 'validation', 'test']:
            ds = tokenized[split]
            n  = min(args.debug_n, len(ds))
            tokenized[split] = ds.select(range(n))
        args.num_train_epochs = 1

    # DataLoader WITH WeightedRandomSampler since the datasets are imbalanced
    counts = tokenized['train']['subreddit_id']
    freq = {c: counts.count(c) for c in set(counts)}
    weights = [1.0 / freq[sid] for sid in counts]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    collator     = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(tokenized['train'], batch_size=args.batch_size,
                              sampler=sampler, collate_fn=collator)
    val_loader   = DataLoader(tokenized['validation'], batch_size=args.batch_size,
                              shuffle=False, collate_fn=collator)
    test_loader  = DataLoader(tokenized['test'], batch_size=args.batch_size,
                              shuffle=False, collate_fn=collator)

    # standard setup
    model = BertForTwoTargetRegression(
        pretrained_encoder=AutoModel.from_pretrained(args.model_name),
        meta_feat_dim=10,
        num_subs=len(freq),
        sub_embed_dim=args.sub_embed_dim
    ).to(args.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    loss_fn = nn.MSELoss()

    # i think does enables mixed precision
    scaler = amp.GradScaler()
    total_steps = args.num_train_epochs * len(train_loader)
    num_warmup = args.warmup_steps if args.warmup_steps is not None else int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup,
        num_training_steps=total_steps
    )

    global_step = 0

    # train
    for epoch in range(1, args.num_train_epochs + 1):
        # ---- TRAIN ----
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
            batch   = {k: torch.tensor(v).to(args.device) for k, v in batch.items()}
            targets = torch.stack([batch['upvotes'], batch['num_comments']], dim=1).float()

            optimizer.zero_grad()
            with amp.autocast():
                preds = model(
                    batch['input_ids'],
                    batch['attention_mask'],
                    batch['meta_feats'],
                    batch['subreddit_id']
                )
                loss = loss_fn(preds, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            if global_step % 100 == 0:
                wandb.log(
                    {
                        'train/loss': loss.item(),
                        'train/lr': scheduler.get_last_lr()[0]
                    },
                    step=global_step
                )

        # val
        model.eval()
        val_step = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                batch   = {k: torch.tensor(v).to(args.device) for k, v in batch.items()}
                preds   = model(batch['input_ids'], batch['attention_mask'],
                                batch['meta_feats'], batch['subreddit_id'])
                loss_val = loss_fn(preds, torch.stack([batch['upvotes'], batch['num_comments']], dim=1).float())

                val_step += 1
                if val_step % 100 == 0:
                    wandb.log({'val/loss': loss_val.item()}, step=global_step)

        all_preds = {0: [], 1: []}
        all_trues = {0: [], 1: []}
        with torch.no_grad():
            for batch in val_loader:
                batch   = {k: torch.tensor(v).to(args.device) for k, v in batch.items()}
                preds   = model(batch['input_ids'], batch['attention_mask'],
                                batch['meta_feats'], batch['subreddit_id'])
                preds_np = preds.cpu().numpy()
                trues_np = np.stack([batch['upvotes'].cpu().numpy(),
                                     batch['num_comments'].cpu().numpy()], axis=1)
                for i in [0, 1]:
                    all_preds[i].extend(preds_np[:, i].tolist())
                    all_trues[i].extend(trues_np[:, i].tolist())

        val_metrics = {}
        for idx, name in [(0, 'upvotes'), (1, 'comments')]:
            mse = mean_squared_error(all_trues[idx], all_preds[idx])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(all_trues[idx], all_preds[idx])
            rho, _ = spearmanr(all_trues[idx], all_preds[idx])
            val_metrics.update({
                f'val/mse_{name}': mse,
                f'val/rmse_{name}': rmse,
                f'val/mae_{name}': mae,
                f'val/spearman_{name}': rho
            })

        wandb.log({'epoch': epoch, **val_metrics}, step=global_step)
        avg_mse   = (val_metrics['val/mse_upvotes'] + val_metrics['val/mse_comments']) / 2
        avg_rmse  = (val_metrics['val/rmse_upvotes'] + val_metrics['val/rmse_comments']) / 2
        avg_mae   = (val_metrics['val/mae_upvotes'] + val_metrics['val/mae_comments']) / 2
        avg_rho   = (val_metrics['val/spearman_upvotes'] + val_metrics['val/spearman_comments']) / 2
        print(f"Epoch {epoch} — val MSE: {avg_mse:.4f}, val RMSE: {avg_rmse:.4f}"
              f", val MAE: {avg_mae:.4f}, val Spearman: {avg_rho:.4f}")

    # test
    model.eval()
    test_step = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test", leave=True):
            batch   = {k: torch.tensor(v).to(args.device) for k, v in batch.items()}
            preds   = model(batch['input_ids'], batch['attention_mask'],
                            batch['meta_feats'], batch['subreddit_id'])
            loss_test = loss_fn(preds, torch.stack([batch['upvotes'], batch['num_comments']], dim=1).float())

            test_step += 1
            if test_step % 100 == 0:
                wandb.log({'test/loss': loss_test.item()}, step=global_step)

    all_preds = {0: [], 1: []}
    all_trues = {0: [], 1: []}
    with torch.no_grad():
        for batch in test_loader:
            batch   = {k: torch.tensor(v).to(args.device) for k, v in batch.items()}
            preds   = model(batch['input_ids'], batch['attention_mask'],
                            batch['meta_feats'], batch['subreddit_id'])
            preds_np = preds.cpu().numpy()
            trues_np = np.stack([batch['upvotes'].cpu().numpy(),
                                 batch['num_comments'].cpu().numpy()], axis=1)
            for i in [0, 1]:
                all_preds[i].extend(preds_np[:, i].tolist())
                all_trues[i].extend(trues_np[:, i].tolist())

    test_metrics = {}
    for idx, name in [(0, 'upvotes'), (1, 'comments')]:
        mse  = mean_squared_error(all_trues[idx], all_preds[idx])
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(all_trues[idx], all_preds[idx])
        rho, _ = spearmanr(all_trues[idx], all_preds[idx])
        test_metrics.update({
            f'test/mse_{name}': mse,
            f'test/rmse_{name}': rmse,
            f'test/mae_{name}': mae,
            f'test/spearman_{name}': rho
        })

    wandb.log(test_metrics, step=global_step)
    avg_mse  = (test_metrics['test/mse_upvotes'] + test_metrics['test/mse_comments']) / 2
    avg_rmse = (test_metrics['test/rmse_upvotes'] + test_metrics['test/rmse_comments']) / 2
    avg_mae  = (test_metrics['test/mae_upvotes'] + test_metrics['test/mae_comments']) / 2
    avg_rho  = (test_metrics['test/spearman_upvotes'] + test_metrics['test/spearman_comments']) / 2
    print(f"Test — avg MSE: {avg_mse:.4f}, avg RMSE: {avg_rmse:.4f}"  
          f", avg MAE: {avg_mae:.4f}, avg Spearman: {avg_rho:.4f}")

    # save model for later
    os.makedirs(args.save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = os.path.join(args.save_dir, f"reddit_reg_model_{ts}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved model checkpoint to {ckpt_path}")
    wandb.save(ckpt_path)
    wandb.finish()

# ----------------
# 5) CLI
# ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=".")
    parser.add_argument("--cache_dir", type=str, default="dataset_cache")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--wandb_project", type=str, default="reddit_regression_new")
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--sub_embed_dim", type=int, default=8)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_n", type=int, default=100)
    parser.add_argument(
        "--subset_fraction",
        type=float,
        default=1.0,
        help="Fraction of each split to use (0.0–1.0) for faster training"
    )
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        return_token_type_ids=False
    )

    train(args)
