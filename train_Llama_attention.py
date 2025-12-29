# import multiprocessing as mp
# mp.set_start_method("spawn", force=True)

import argparse
#import glob
import os
import json
import time
import logging
import random
import re
#from itertools import chain
#from string import punctuation
from tqdm import tqdm

#import nltk
#nltk.download('punkt')
#from nltk.tokenize import sent_tokenize

#import pandas as pdn
import numpy as np
import torch
#import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
#from nlp import load_metric
#import sys
#import scipy.special

from transformers import (
    get_linear_schedule_with_warmup,
    AutoModelForCausalLM,
    AutoTokenizer
)
from torch.optim import AdamW

from peft import LoraConfig, get_peft_model
# push merged model to Hugging Face Hub
from huggingface_hub import login, upload_folder
#import wandb



# hyperparameters and the like
model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
#model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
#model_name = "meta-llama/Llama-3.1-8B"

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
medical_terms_file = "/kaggle/input/summary-medical-concepts/summary_medical_concepts.txt"
Openi_dataset = "/kaggle/input/openi-with-terms/Openi_with_terms.jsonl"
output_length = 512
input_length = 1024
prompt = "You are an expert medical professional. Summarize the radiology report findings into an impression with minimal text"
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

lambda_medical=0.1
lambda_negation=0.06
adam_epsilon=1e-7
#learning_rate=0.00006   
learning_rate=0.001
train_batch_size=1
eval_batch_size=8
output_dir="/kaggle/working/llama3.1-8B-WeightedLoss"
repo_id = "Chilliwiddit/llama3.1-8B-WeightedLoss"
hf_token = "" 

tb_logger = TensorBoardLogger(save_dir="logs/", name="my_model")





class LlamaFineTuner(pl.LightningModule):
    def __init__(self):
        pl.seed_everything(42)

        super(LlamaFineTuner, self).__init__()
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16
        )

        
        self.model = get_peft_model(base_model, lora_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            print("Adding pad token...")
            self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            print("Added pad token")
        
        #testing code
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.use_cache = False

        self.training_data = Resource(tokenizer=self.tokenizer, type_path=None, num_samples=None, input_length=input_length, output_length=output_length)


        with open(medical_terms_file) as f:
            terms = [ln.strip() for ln in f if ln.strip()]
        med_ids = []
        for t in terms:
            med_ids.extend(self.tokenizer(t, add_special_tokens=False)["input_ids"])
            med_ids.extend(self.tokenizer(" " + t, add_special_tokens=False)["input_ids"])
            print(f"Processed medical term: {t}")
        self.register_buffer("medical_vocab_ids",
                            torch.tensor(sorted(set(med_ids)), dtype=torch.long))
        print("Finished reading medical_term_file.txt !")


        neg_ids = []
        for t in negation_terms:
            neg_ids.extend(self.tokenizer(t, add_special_tokens=False)["input_ids"])
            print(f"Processed negation term: {t}")
        self.register_buffer("negation_vocab_ids",
                            torch.tensor(sorted(set(neg_ids)), dtype=torch.long))
        print("Finished construction of neg_unigrams_ids!")



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

        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        logits = outputs.logits  # [B, T, V]
        labels = batch["labels"] # [B, T], -100 masked

        # SHIFT Logits and Labels (CRITICAL STEP)
        # We predict the NEXT token, so we remove the last logit and the first label.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Define V (vocab size) based on shifted logits
        V = shift_logits.size(-1)

        loss_tok = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, V), 
            shift_labels.view(-1),
            reduction="none", 
            ignore_index=-100
        ).view_as(shift_labels)  # [B, T]


        # build medical mask from **label ids**
        # medical_vocab_ids: 1D LongTensor buffer registered in __init__    
        with torch.no_grad():
            med_mask = torch.isin(shift_labels, self.medical_vocab_ids) & (shift_labels != -100)
            neg_mask = torch.isin(shift_labels, self.negation_vocab_ids) & (shift_labels != -100)
            mask = (shift_labels != -100).float()

        
        #training code
        print("Number of -100 in labels:", (labels == -100).sum().item())
        print("Is there at least one medical mask token:", med_mask.any().item())

                
        test_weight = torch.ones_like(loss_tok)
        test_loss = ((loss_tok * test_weight) * mask).sum() / mask.sum()
        print("Test loss (no weighting):", test_loss.item())

        weights = torch.ones_like(loss_tok)

        weights = weights + (med_mask.float() * lambda_medical)

        weights = weights + (neg_mask.float() * lambda_negation)

        loss = ((loss_tok * weights) * mask).sum() / mask.sum()
        print("Weighted loss:", loss.item())
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
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

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
   

    #no need validation_step

    #no need validation_epoch_end


    def configure_optimizers(self):
        trainable_params = [p for n, p in self.model.named_parameters() if p.requires_grad]

        optimizer = AdamW(
            trainable_params,
            lr=learning_rate,
            eps=adam_epsilon
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min((step+1)/100, 1.0)   # simple warmup
        )

        self.opt = optimizer

        return {
            "optimizer": optimizer,
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
            shuffle=False,
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
            output_test_results_file = os.path.join(output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))


class Resource(Dataset):
    def __init__(self, tokenizer, type_path, num_samples, input_length, output_length, print_text=False):
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


class OwnData(Dataset):
    def __init__(self, tokenizer, type_path, num_samples, input_length, output_length, print_text=False):  
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
    


tokenizer = AutoTokenizer.from_pretrained(model_name)


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
    n_gpu=1,
    resume_from_checkpoint=None, 
    val_check_interval = 0.05, 
    n_val=1000,
    n_train=-1,
    n_test=-1,
    early_stop_callback=False,
    fp_16=True, # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=0.5, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
    tau=1.0,
    lambda_medical=lambda_medical,
    lambda_negation=lambda_negation
)


args = argparse.Namespace(**args_dict)

## Define Checkpoint function
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=args.output_dir, 
    filename="checkpoint-epoch-{epoch}", 
    every_n_epochs=3, 
    save_top_k=1, 
    save_last=True
)


train_params = dict(
    accelerator="gpu",
    devices=args.n_gpu if args.n_gpu>0 else None,
    #strategy="ddp_spawn",
    precision="16-mixed",
    max_epochs=args.num_train_epochs,
    callbacks=[LoggingCallback(), checkpoint_callback],
    accumulate_grad_batches=args.gradient_accumulation_steps,
    #precision= 16 if args.fp_16 else 32,
    gradient_clip_val=args.max_grad_norm,
    #checkpoint_callback=checkpoint_callback,
    val_check_interval=args.val_check_interval,
    logger=tb_logger,
    sync_batchnorm=True,
    fast_dev_run=False, 
)


def get_dataset(tokenizer, type_path, num_samples, input_length, output_length):
    return OwnData(tokenizer=tokenizer, type_path=None, num_samples=None, input_length=input_length, output_length=output_length)


model = LlamaFineTuner()

trainer = pl.Trainer(**train_params)


print ("Training model")
trainer.fit(model)


print ("training finished")



if hf_token:
    login(token=hf_token)

final_output_dir = "/kaggle/working/llama3.1-8B-WeightedLoss-Final"


print(f"Saving tokenizer to {final_output_dir}...")
model.tokenizer.save_pretrained(final_output_dir)

print("Saving PEFT Adapters...")
peft_model = model.model 
peft_model.save_pretrained(final_output_dir)

print("Uploading clean folder to Hugging Face...")
upload_folder(
    repo_id=repo_id, 
    folder_path=final_output_dir, 
    repo_type="model", 
    token=hf_token
)

print("Model push complete.")
