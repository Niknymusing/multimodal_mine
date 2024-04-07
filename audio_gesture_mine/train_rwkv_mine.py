
#### instantiate the model and start the pytorch lightning trainer

import os
import random
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


# Set Environment Variables
os.environ["RWKV_CTXLEN"] = '128'
os.environ["RWKV_HEAD_SIZE_A"] = '64' # Ensure this is consistent with head_size_a in args
os.environ["RWKV_FLOAT_MODE"] = 'bf16' # Change to bfloat16 to match CUDA kernel expectations
os.environ["RWKV_JIT_ON"] = "0"
os.environ["RWKV_T_MAX"] = "256"
#os.environ["RWKV_MY_TESTING"] = "x060" # Uncomment this if using the wkv6 CUDA kernel


import argparse
from argparse import Namespace
import pytorch_lightning as pl
from src.model import RWKV


# Configure args
args = Namespace(
    n_embd=128,
    vocab_size=100,  # Adjust to your actual vocabulary size
    n_layer=6,
    dim_att=128,
    dim_ffn=256,
    tiny_att_layer=-1,
    tiny_att_dim=-1,
    dropout=0,
    head_qk=0,
    layerwise_lr=1,
    my_pile_stage=0,
    weight_decay=0.01,
    ctx_len=128,
    lr_init=6e-4,
    accelerator='GPU',
    my_pos_emb=0,
    pre_ffn=0,
    head_size_a=64,  # Ensure this matches RWKV_HEAD_SIZE_A environment variable
    head_size = 64,
    n_head = 1,
    head_size_divisor=1,
    grad_cp=0,
    betas=(0.9, 0.999),
    adam_eps=1e-8,
    precision = 'bf16'  # Match precision with RWKV_FLOAT_MODE
)
model = RWKV(args)


# Load the checkpoint file
checkpoint_path = "/content/drive/MyDrive/rwkv_mine_training/checkpoints/epoch=9-step=115200.ckpt"
checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

# Update model state
model.load_state_dict(checkpoint['state_dict'])
# Instantiate the model
#model = RWKV(args)

model.train()
print(sum([p.numel() for p in model.parameters() if p.requires_grad == True])) #### count model parameters


# Configure the TensorBoardLogger
logger = TensorBoardLogger(
    save_dir='/content/drive/MyDrive/rwkv_mine_training',
    name='logs'
)



checkpoint_callback = ModelCheckpoint(
    dirpath='/content/drive/MyDrive/rwkv_mine_training/checkpoints',  # Specify your path here
    monitor='val_loss',  # Replace with your metric
    filename='{epoch}-{step}-{val_loss:.2f}',
    every_n_train_steps=300,
    save_top_k=1,  # Save the best checkpoint only
    mode='min',  # 'min' for minimizing the metric, 'max' for maximizing
)

# Trainer setup with callbacks and logger
trainer = pl.Trainer(
    accelerator='gpu',
    devices=-1,
    max_epochs=10,
    precision="bf16",
    callbacks=[checkpoint_callback],
    logger=logger
)


#print([list(next(iter(dataloader)))[i].shape for i in range(4) ])
trainer.fit(model, dataloader)