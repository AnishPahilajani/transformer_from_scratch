import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_weights_file_path, get_config, latest_weights_file_path

# Huggingface datasets and tokenizers
from datasets import load_dataset, Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel, BPE
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

import pandas as pd

torch.manual_seed(69)
#train_csv = "./transformer_from_scratch/train.csv"
train_csv = "./train.csv"
dev_csv = "./dev.csv"
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        # select token with max probability (because it is greedy search)
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)
        

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0) # removes the batch dimension

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    
    count = 0
    source_texts = []
    expected = []
    predicted = []
    ground_label = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            #print("val batch", batch)
            count += 1
            # encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            # encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)
            
            encoder_input = batch['encoder_input'].to(device) # (batch, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (Batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch, 1, 1, seq_len) hide only [PAD] tokens
            decoder_mask = batch['decoder_mask'].to(device) # (Batch, 1, seq_len, seq_len) hide [PAD] and subsequent tokens

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)
            
            threshold = 0.5
            binary_outputs = torch.where(proj_output < threshold, torch.zeros_like(proj_output), torch.ones_like(proj_output))
            
            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
            
            #model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            ground_truth = batch["label"].item()
            #model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(binary_outputs.item())
            ground_label.append(ground_truth)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'Question: ':>12}{source_text}")
            print_msg(f"{f'Option: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{binary_outputs.item()}")
            print_msg(f"{f'Ground truth: ':>12}{ground_truth}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    return predicted, ground_label
    # this in tensorboard related ignore for now
    # if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        # metric = torchmetrics.CharErrorRate()
        # cer = metric(predicted, ground_label)
        # writer.add_scalar('validation cer', cer, global_step)
        # writer.flush()

        # Compute the word error rate
        # metric = torchmetrics.WordErrorRate()
        # wer = metric(predicted, expected)
        # writer.add_scalar('validation wer', wer, global_step)
        # writer.flush()

        # Compute the BLEU metric
        # metric = torchmetrics.BLEUScore()
        # bleu = metric(predicted, expected)
        # writer.add_scalar('validation BLEU', bleu, global_step)
        # writer.flush()

def run_test(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer):
    model.eval()
    
    count = 0
    source_texts = []
    expected = []
    predicted = []
    ground_label = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            #print("val batch", batch)
            count += 1
            # encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            # encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)
            
            encoder_input = batch['encoder_input'].to(device) # (batch, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (Batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch, 1, 1, seq_len) hide only [PAD] tokens
            decoder_mask = batch['decoder_mask'].to(device) # (Batch, 1, seq_len, seq_len) hide [PAD] and subsequent tokens

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)
            
            threshold = 0.5
            binary_outputs = torch.where(proj_output < threshold, torch.zeros_like(proj_output), torch.ones_like(proj_output))
            
            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
            
            #model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            ground_truth = batch["label"].item()
            #model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(binary_outputs.item())
            ground_label.append(ground_truth)
            
            # Print the source, target and model output
            # print_msg('-'*console_width)
            # print_msg(f"{f'Question: ':>12}{source_text}")
            # print_msg(f"{f'Option: ':>12}{target_text}")
            # print_msg(f"{f'PREDICTED: ':>12}{binary_outputs.item()}")
            # print_msg(f"{f'Ground truth: ':>12}{ground_truth}")

            # if count == num_examples:
            #     print_msg('-'*console_width)
            #     break
    return predicted, ground_label


def get_all_sentences(ds):
    for item in ds:
        analysis = item['analysis']
        if not analysis:
            analysis = ''
        
        complete_analysis = item['complete analysis']
        if not complete_analysis:
            complete_analysis = ''
        
        explanation = item['explanation']
        if not explanation:
            explanation = ''
        yield item['question'] + ' ' + item['answer'] + ' ' + analysis + complete_analysis + explanation

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    # It only has the train split, so we divide it overselves
    # ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    # print(ds_raw)
    
    train = pd.read_csv(train_csv)
    dev = pd.read_csv(dev_csv)
    # Create a Hugging Face Dataset
    train_ds_raw = Dataset.from_pandas(train)
    val_ds_raw = Dataset.from_pandas(dev)
    print("Train data structure: ", train_ds_raw)
    print("Dev data structure: ", val_ds_raw)

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, train_ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, train_ds_raw, config['lang_tgt'])
    #tokenizer = get_or_build_tokenizer(config, ds_raw, config['lang_src'])

    # Keep 90% for training, 10% for validation
    # train_ds_size = int(0.9 * len(ds_raw))
    # val_ds_size = len(ds_raw) - train_ds_size
    # train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in train_ds_raw:
        src_ids = tokenizer_src.encode(item['question']).ids
        tgt_ids = tokenizer_tgt.encode(item['answer']).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    # val_dataloader = None
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model

def train_model(config):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)
    
    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    # we dont want model ctonsider pad tokens when calculating loss, so we ignore it this way
    #TODO Need to make this BCE loss
    #loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (batch, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (Batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch, 1, 1, seq_len) hide only [PAD] tokens
            decoder_mask = batch['decoder_mask'].to(device) # (Batch, 1, seq_len, seq_len) hide [PAD] and subsequent tokens

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (Batch, seq_len)
            
            
            threshold = 0.5
            binary_outputs = torch.where(proj_output < threshold, torch.zeros_like(proj_output), torch.ones_like(proj_output))
            binary_outputs.requires_grad = True

            # print(proj_output.shape)
            # print(batch['label'])
            # Compute the loss using a simple cross entropy
            # (batch, seq_len, tgt_vocab_size) -> (batch * seq_len, tgt_vocab_size)
            # loss = loss_fn(proj_output.view(-1), label.view(-1).float())
            loss = loss_fn(binary_outputs, label.float())
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
    print("RUNNING ON ALL VAL DATA")
    pred, gr_lab = run_test(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
    
    t = len(pred)
    c = 0
    for i in range(len(pred)):
        if pred[i] == gr_lab[i]:
            c+= 1
    print(f"Validation accurace: {c/t*100}")
        
        

if __name__ == '__main__':
    #warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)