import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation
from tqdm import tqdm

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import WandbLogger
from nlp import load_metric
import sys
import scipy.special

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import LoraConfig, get_peft_model
import wandb

# hyperparameters and the like
model_name = "meta-llama/Llama-3.1-8B"
lora_config = LoraConfig(
        r = 32,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_rslora=False
)
num_epochs = 5
negation_terms = ["negative", "free", "clear", "resolved", "normal", "improved", "stable", "absence", "remission", "denied", "rule", "ruled", "present", "evidence", "found", "unlikely", "have", "has", "contribute", "no", "not", "denies", "denied", "without", "never", "none", "neither"]
medical_terms_file = "summary_medical_concepts.txt"
Openi_dataset = "Openi_with_terms.jsonl"
output_length = 512
input_length = 2048
prompt = "You are an expert medical professional. Summarize the radiology report findings into an impression with minimal text"
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

lambda_medical=0.0021
lambda_negation=0.0021
adam_epsilon=1e-7
#learning_rate=0.00006   
learning_rate=0.001
train_batch_size=2
eval_batch_size=8
output_dir="llama3.1-8B-Med-Neg-Lora"

YOUR_API_KEY = ''
os.environ["WANDB_API_KEY"] = YOUR_API_KEY
wandb_logger = WandbLogger(project='MQA_Bart')




def set_seed(seed):
    random.randint(0, 4)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


model = AutoModelForCausalLM.from_pretrained(model_name)


class LlamaFineTuner(pl.LightningModule):
    def __init__(self):
        super(LlamaFineTuner, self).__init__()
        self.model = get_peft_model(model, lora_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        self.training_data = Resource(tokenizer=self.tokenizer, type_path=None, num_samples=None, input_length=input_length, output_length=output_length)



    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))
    

    def return_token_ids(self, t):
        ids = self.tokenizer.batch_encode_plus([t], truncation=True, return_tensors="pt")['input_ids'][0]
        return ids


    def is_logger(self):
        return self.trainer.proc_rank <= 0
    

    def parse_score(self, result):
        return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}


    def forward(
      self, input_ids, attention_mask=None, labels=None
  ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
    )
    


    def _step(self, batch, training_mode=False):
        labels = batch["labels"]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100
        
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=labels
        )

        effective_batch_size = outputs.logits.size()[0]

        medical_loss = torch.tensor(0.0).type_as(outputs[0]) 
        negation_loss = torch.tensor(0.0).type_as(outputs[0])

        for i in range(effective_batch_size):
            average_logits = torch.mean(outputs.logits[i], 0)
            idx = batch["id"][i].item()

            medical_terms = self.training_data[idx]['medical_terms']#gets the medical terms
            position_list = self.training_data[idx]['position_list']#gets the positions of the medical terms
            neg_uni = self.training_data[idx]['neg_uni']# gets the negations terms all for a row of data
            source = labels[i]#gets the labels for that row

            # update negation_loss
            if len(neg_uni) > 0:
                for term in neg_uni: #for each negation term
                    id_comb = None
                    # check membership first
                    if term in neg_unigrams_ids:
                        id_comb = neg_unigrams_ids[term]  # gets the token ids for that negation term
                    else:
                        id_comb = self.return_token_ids(term) #manually tokenizes the term and returns the ids

                    for j in range(id_comb.size()[0]):
                        neg_id = id_comb[j].item() #gets a specific token
                        presence_neg = (source == neg_id).nonzero(as_tuple=True)[0].tolist()  #returns the list of positions where the token appears in the source

                        # if there are no positions, continue
                        if len(presence_neg) == 0: 
                            continue

                        for p in presence_neg: #for each position....
                            negation_loss += average_logits[neg_id] 


            #update medical loss
            if len(medical_terms) > 0:
                for m in range(len(medical_terms)):#we iterate this way to get the index...
                    id_comb = None
                    if position_list[m] == 1:
                        if medical_terms[m] in medical_term_ids_mid:
                            id_comb = medical_term_ids_mid[medical_terms[m]]
                        else:
                            id_comb = self.return_token_ids(medical_terms[m])
                    elif position_list[m] == 0:
                        if medical_terms[m] in medical_term_ids_begin:
                            id_comb = medical_term_ids_begin[medical_terms[m]]
                        else:
                            id_comb = self.return_token_ids(medical_terms[m])
                    elif position_list[m] == 2:
                        if (medical_terms[m] in medical_term_ids_mid) and (medical_terms[m] in medical_term_ids_begin):
                            id_comb = torch.unique(torch.cat((medical_term_ids_mid[medical_terms[m]], medical_term_ids_begin[medical_terms[m]])))
                        else:
                            id_comb = self.return_token_ids(medical_terms[m])

                    for j in range(id_comb.size()[0]):
                        vocab_id = id_comb[j].item()
                        presence_vocab = (source == vocab_id).nonzero(as_tuple=True)[0].tolist()

                        # if there are no positions, continue
                        if len(presence_vocab) == 0: 
                            continue

                        for p in presence_vocab:
                            medical_loss += average_logits[vocab_id]
                            #for good measure, adds the average logits for upto the 2 previous tokens
                            if p - 1 >= 0: 
                                medical_loss += average_logits[source[p-1]]
                            if p - 2 >= 0: 
                                medical_loss += average_logits[source[p-2]]   


        loss = outputs[0]#gets the loss
        print(f"Initial loss: {loss.item()}, Medical loss: {medical_loss.item()}, Negation loss: {negation_loss.item()}")

        loss += (lambda_medical * medical_loss) / effective_batch_size
        loss += (lambda_negation * negation_loss) / effective_batch_size #ADDS the loss in order to penalize over rewarding it
        
        print(f"Final loss: {loss.item()}")

        return loss
    
    
    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)

    
    def _generative_step(self, batch):
        t0 = time.time()

        generated_ids = self.model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=True,
            max_length=output_length, 
            num_beams=5,
            repetition_penalty=1.5,
            early_stopping=True
        )

        preds = self.ids_to_clean_text(generated_ids)
        targets = self.ids_to_clean_text(batch["labels"])

        gen_time = (time.time() - t0) / batch["input_ids"].shape[0]

        loss = self._step(batch)
        base_metrics = {'val_loss': loss}

        summ_len = np.mean(self.lmap(len, generated_ids))

        base_metrics.update(
            gen_time=torch.tensor(gen_time).to(loss.device), 
            gen_len=torch.tensor(summ_len).to(loss.device), 
            preds=preds, 
            target=targets
        )

        return base_metrics

    
    def training_step(self, batch, batch_idx):
        loss = self._step(batch, training_mode=True)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}
    

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}
    

    #no need validation_step

    #no need validation_epoch_end


    def configure_optimizers(self):
        trainable_params = [p for n, p in self.model.named_parameters() if p.requires_grad]

        optimizer = AdamW(
            trainable_params,
            lr=self.learning_rate,
            eps=adam_epsilon
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min((step+1)/100, 1.0)   # simple warmup
        )

        self.opt = optimizer

        return {
            "optimizer": [optimizer],
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }

    
    #no need configure optimizer step

    #no need get_tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(
            tokenizer=self.tokenizer,
            type_path=None,
            num_samples=None,
            input_length=input_length,
            output_length=output_length
        )
        return DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=2
        )

    
    #no need val_dataloder

    

class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))


class Resource(Dataset):
    def __init__(self, tokenizer, type_path, num_samples, input_length, output_length, print_text=True):
        file = Openi_dataset
        dataset_list = []
        count = 0#this basically is the index, starting from 0, obviously
        with open(file, 'r') as input:
            for jsonObj in input:
                patientDict= json.loads(jsonObj) #takes each line as a JSON object
                d = {} #creates a dictionary

                d["id"] = count
                d["text"] = patientDict["article"]
                d["headline"] = patientDict["summary"]
                d["medical_terms"] = patientDict["medical_terms"]

                # encode the position of each medical term 
                position_list = []
                for m in d["medical_terms"]:#for each medical term...
                    test_str = d["headline"]#gets the target sequence
                    res = [i for i in range(len(test_str)) if test_str.startswith(m, i)]#this returns all the positions where that specific medical terms STARTS in the target sequence, it gives it as a list of integers
                    if len(res) > 1 and res[0] == 0:#if the size of res is more than one (more than 1 appearances) and the first appearance is at index 0 (at the beginning)....
                        position_list.append(2)#assigned to position 2
                    elif len(res) > 0 and res[0] == 0:#if the size of res is more than 0 (1 appearance) and the appearance is at index 0 (at the beginning)....
                        position_list.append(0)#assigned to position 0
                    elif len(res) > 0 and res[0] > 0:#if the size of res is more than 0 (1 appearance) and the appearance is after index 0 (NOT at the beginning)....
                        position_list.append(1)#assigned to position 1

                d["position_list"] = position_list#this position list basically doesn't tell what MEDICAL TERM it is, it just says if it's 0, 1 or 2...

                d["neg_uni"] = patientDict["negation_terms"]#negation terms
                dataset_list.append(d)#each object is basically a row
                count += 1#update count

                self.dataset = dataset_list
                if num_samples:
                    self.dataset = self.dataset[:num_samples]
                self.input_length = input_length
                self.tokenizer = tokenizer
                self.output_length = output_length
                self.print_text = print_text


    def __len__(self):
        return len(self.dataset)#returns the length of dataset_list (2735)
    

    def convert_to_features(self, example_batch):
        # Tokenize contexts and questions (as pairs of inputs)
        
        if self.print_text:
            print("Input Text: ", example_batch['text'])
        
        input_ = example_batch['text']
        target_ = example_batch['headline']

        text = alpaca_prompt.format(prompt, input_, target_) + tokenizer.eos_token
        label_text = target_ + tokenizer.eos_token

        encoded = self.tokenizer.batch_encode_plus([text], max_length=self.output_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
        tokenized_label = self.tokenizer.batch_encode_plus([label_text], max_length=self.output_length,
                                                           padding='max_length', truncation=True, return_tensors="pt")
        
        return encoded, tokenized_label
    

    def __getitem__(self, index):
        encoded, tokenized_label = self.convert_to_features(self.dataset[index])

        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()

        medical_terms = self.dataset[index]["medical_terms"]        
        neg_uni = self.dataset[index]["neg_uni"]
        position_list = self.dataset[index]["position_list"]
        id = self.dataset[index]["id"]

        masked_labels = input_ids.clone()
        label_length = (tokenized_label["attention_mask"].squeeze() ==1).sum()#gets length of the tokenized labels alone by counting the number of 1s

        total_seq_length = attention_mask.sum()#does the same for the full attention mask

        mask_until = total_seq_length - label_length #gets the index to mask until

        masked_labels[:mask_until] = -100 #mask the sequence where only the target sequence is visible


        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": masked_labels,
            "medical_terms": medical_terms,
            "position_list": position_list,
            "neg_uni": neg_uni,
            "id": id,
        }


class OwnData(Dataset):
    def __init__(self, tokenizer, type_path, num_samples, input_length, output_length, print_text=True):  
        file = Openi_dataset
        dataset_list = []
        count = 0#this basically is the index, starting from 0, obviously
        with open(file, 'r') as input:
            for jsonObj in input:
                patientDict= json.loads(jsonObj) #takes each line as a JSON object
                d = {} #creates a dictionary

                d["id"] = count
                d["text"] = patientDict["article"]
                d["headline"] = patientDict["summary"]

                dataset_list.append(d) #each object is basically a row
                count += 1 #update count

                self.dataset = dataset_list
                if num_samples:
                    self.dataset = self.dataset[:num_samples]
                self.input_length = input_length
                self.tokenizer = tokenizer
                self.output_length = output_length
                self.print_text = print_text


    def __len__(self):
        return len(self.dataset)#returns the length of dataset_list (2735)
    
    def convert_to_features(self, example_batch):
        # Tokenize contexts and questions (as pairs of inputs)
        
        if self.print_text:
            print("Input Text: ", example_batch['text'])
        
        input_ = example_batch['text']
        target_ = example_batch['headline']

        text = alpaca_prompt.format(prompt, input_, target_) + tokenizer.eos_token
        label_text = target_ + tokenizer.eos_token

        encoded = self.tokenizer.batch_encode_plus([text], max_length=self.output_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
        tokenized_label = self.tokenizer.batch_encode_plus([label_text], max_length=self.output_length,
                                                           padding='max_length', truncation=True, return_tensors="pt")
        
        return encoded, tokenized_label
    

    def __getitem__(self, index):
        encoded, tokenized_label = self.convert_to_features(self.dataset[index])

        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()

        id = self.dataset[index]["id"]

        masked_labels = input_ids.clone()
        label_length = (tokenized_label["attention_mask"].squeeze() ==1).sum()#gets length of the tokenized labels alone by counting the number of 1s

        total_seq_length = attention_mask.sum()#does the same for the full attention mask

        mask_until = total_seq_length - label_length #gets the index to mask until

        masked_labels[:mask_until] = -100 #mask the sequence where only the target sequence is visible


        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": masked_labels,
            "id": id,
        }
    

set_seed(42)


#here we go through the medical terms and tokenize it to create a dictionary. We do the same thing for negation terms as well
medical_term_ids_begin = {}
medical_term_ids_mid = {}
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

with open(medical_terms_file, 'r', encoding='utf8') as f:
    custom_noun = f.readlines()
    for i in range(len(custom_noun)):
        medical_term = custom_noun[i].replace('\n', '')
        ids = tokenizer.batch_encode_plus([medical_term], truncation=True, return_tensors="pt")['input_ids'][0]
        medical_term_ids_begin[medical_term] = ids
        #tokenizes the list of medical terms and adds them to the dictionary where the tokens are sorted under their term

        ids = tokenizer.batch_encode_plus([" " + medical_term], truncation=True, return_tensors="pt")['input_ids'][0]
        medical_term_ids_mid[medical_term] = ids
        print("Added medical term: ", medical_term)
        #tokenizes again just like above but this time the medical terms start with a space
print("Finished reading medical_term_file.txt !")


neg_unigrams = negation_terms
neg_unigrams_ids = {}
for e in neg_unigrams:
    ids = tokenizer.batch_encode_plus([e], truncation=True, return_tensors="pt")['input_ids'][0]
    
    neg_unigrams_ids[e] = ids
    #does the same as with the second group of medical terms
    print("Added negation term: ", e)
print("Finished construction of neg_unigrams_ids!")


logger = logging.getLogger(__name__)
args_dict = dict(
    output_dir=output_dir, # path to save the checkpoints
    model_name_or_path=model_name,
    tokenizer_name_or_path=model_name,
    max_input_length=input_length,
    max_output_length=output_length,
    freeze_encoder=False,
    freeze_embeds=False,
    learning_rate=learning_rate,
    weight_decay=0.0,
    adam_epsilon=adam_epsilon,
    warmup_steps=0,
    train_batch_size=train_batch_size,
    eval_batch_size=eval_batch_size,
    num_train_epochs=num_epochs,
    gradient_accumulation_steps=16,
    n_gpu=2,
    resume_from_checkpoint=None, 
    val_check_interval = 0.05, 
    n_val=1000,
    n_train=-1,
    n_test=-1,
    early_stop_callback=False,
    fp_16=True, # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
    tau=1.0,
    lambda_medical=lambda_medical,
    lambda_negation=lambda_negation
)


args = argparse.Namespace(**args_dict)

## Define Checkpoint function
checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=1)


train_params = dict(
    accelerator="gpu",
    devices=args.n_gpu if args.n_gpu>0 else None,
    max_epochs=args.num_train_epochs,
    early_stop_callback=False,
    accumulate_grad_batches=args.gradient_accumulation_steps,
    precision= 16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    resume_from_checkpoint=args.resume_from_checkpoint,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=checkpoint_callback,
    val_check_interval=args.val_check_interval,
    callbacks=[LoggingCallback()],
    logger=wandb_logger,
    sync_batchnorm=True,
    accelerator='dp'
)


def get_dataset(tokenizer, type_path, num_samples, input_length, output_length):
    return OwnData(tokenizer=tokenizer, type_path=None, num_samples=None, input_length=input_length, output_length=output_length)


model = LlamaFineTuner(args)

trainer = pl.Trainer(**train_params)
print ("Training model")
trainer.fit(model)


print ("training finished")
