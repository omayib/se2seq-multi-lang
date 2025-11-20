from pathlib import Path

import torch

HIDDEN_SIZE = 512 # Increased from 128
DROPOUT_RATE = 0.1 # Increased from 0.1
NUM_EPOCHS = 300 # Increased from 150
NUM_LAYER = 3
SOS_TOKEN=0
EOS_TOKEN=1
PAD_TOKEN=2
TO_JV_TOKEN=3
TO_SU_TOKEN=4
TO_JV_TOKEN_STR = "<to_javanese>"
TO_SU_TOKEN_STR = "<to_sundanese>"
MAX_LENGTH=30
BATCH_SIZE=64
LEARNING_RATE=0.001
TRAINING_RATE=0.8
current_dir = Path(__file__).resolve().parent
# Path
OUTPUT_DIR="./../output"
NORMAL_CHECKPOINT="normal_seq2seq.pth"
ATTENTION_CHECKPOINT="attention_seq2seq.pth"
RESULT_DIR="/result"
DATA_RAW_DIR="./../data/raw"
DATA_PROCESSED_DIR="./../data/processed"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')