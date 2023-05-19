# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""T5 finetuning runner."""

import csv
import os, json
import logging
import argparse
import random
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import T5Tokenizer
from transformers import T5ForConditionalGeneration
from transformers import get_linear_schedule_with_warmup, AdamW

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = ".."

GOALRECOGNITION_PATH = ROOT_DIR + "/datasets/goalrecognition/blocksworld/"
TEST_1_PATH = ROOT_DIR + "/datasets/goalrecognition/blocksworld/goalrecognition-blocksworld-5-3k.jsonl"
TEST_2_PATH = ROOT_DIR + "/datasets/generalization/GE1/ge1-goalrecognition-blocksworld-10-3-3k.jsonl"
TEST_3_PATH = ROOT_DIR + "/datasets/generalization/GE3/ge3-goalrecognition-blocksworld-5-3-3k.jsonl"

def load_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    return data

class RacinExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, pid, state=None, observations=None, query=None, label=None):
        self.pid = pid
        self.state = state
        self.observations = observations
        self.query = query
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_id = label_id

class RacinProcessor(object):
    def get_train_examples(self, data_dir):
        racin_path = data_dir + "goalrecognition-blocksworld-5-12k.jsonl"
        racin = load_jsonl(racin_path)
        examples = []
        count = 0
        for r in racin:
            count += 1
            if count <= 10000:
                examples.append(
                    RacinExample(pid=r['problem_id'], state=r['state'], observations=r['observations'], query=r['query'], label=r['label']))
        
        print("train size:", len(examples))
        return examples     
                    
    def get_dev_examples(self, data_dir):
        racin_path = data_dir + "goalrecognition-blocksworld-5-12k.jsonl"
        racin = load_jsonl(racin_path)
        examples = []
        count = 0
        for r in racin:
            count += 1
            if count > 10000:
                examples.append(
                    RacinExample(pid=r['problem_id'], state=r['state'], observations=r['observations'], query=r['query'], label=r['label']))
        
        print("dev size:", len(examples))
        return examples    
    
    def get_test_examples(self, data_dir):
        racin = load_jsonl(data_dir)
        examples = []
        for r in racin:
            examples.append(
                RacinExample(pid=r['problem_id'], state=r['state'], observations=r['observations'], query=r['query'], label=r['label']))
            
        return examples  
    
    def get_labels(self):
        """See base class."""
        return [0, 1]
        

    
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}
    max_len = 0
    features = []
    
    count = 0
    for (ex_index, example) in enumerate(examples):    
        count += 1
        StateAndAction = tokenizer.tokenize(example.state + " " + example.observations)

        query = tokenizer.tokenize(example.query)

        tokens = StateAndAction + query
    
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        if len(input_ids) > max_len:
            max_len = len(input_ids)
            
        input_mask = [1] * len(input_ids)
        
        input_ids += [0] * (max_seq_length - len(input_ids))
        input_mask += [0] * (max_seq_length - len(input_mask))

        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        
        label_id = label_map[example.label]
        
        label_text = " "
        if label_id == 0:
            label_text = "No"
        elif label_id == 1:
            label_text = "Yes"
            
        label_token = tokenizer.tokenize(label_text)

        label_ids = tokenizer.convert_tokens_to_ids(label_token)
        
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              label_id=label_ids,))
    print(max_len)
    return features

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def get_test_dataloader(data_dir, label_list, args, tokenizer, processor):
    test_examples = processor.get_test_examples(data_dir)
    test_features = convert_examples_to_features(
        test_examples, label_list, args.max_seq_length, tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)

    test_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)

    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.train_batch_size)
    
    return test_dataloader

def do_evaluation(model, eval_dataloader, args, device, tokenizer, is_training=False):
    if is_training:
        eval_flag='train'
    else:
        eval_flag='eval'
        
    model.eval()
    
    #tokenizer.padding_side = "left"
    #tokenizer.pad_token = tokenizer.eos_token  # to avoid an error
    
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    logits_all=None
    
    total = 0
    correct = 0
    with torch.no_grad():
        for input_ids, input_mask, label_ids in eval_dataloader:
            
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            label_ids = label_ids.to(device)
            
            output_sequences  = model.generate(input_ids=input_ids, attention_mask=input_mask, do_sample=False, max_length=2, min_length=1)
            
            preds = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

            for i in range(len(labels)):
                total += 1
                if labels[i] == preds[i]:
                    correct += 1

    eval_accuracy = correct / total
    print(total)
    result = {'seed': args.seed,
              f'{eval_flag}_accuracy': eval_accuracy,}
    logger.info("  %s = %s", f'{eval_flag}_accuracy', str(result[f'{eval_flag}_accuracy']))
    logger.info(f"***** {eval_flag} results *****")
    for key in sorted(result.keys()):
         logger.info("  %s = %s", key, str(result[key]))
    model.zero_grad()
    return logits_all, eval_accuracy
            
            
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--t5_model_dir", default="~/.cache/t5-base", type=str,
                        help="t5 pre-trained model selected in the list: t5-base, t5-large")
    parser.add_argument("--task_name",
                        default="racin",
                        type=str,
                        help="The name of the task to train.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run test on the test set.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=5,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.06,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--train_example_size",
                        default=2000,
                        type=int,
                        help="The size of training set.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--gpuid', type=int, default=-1,help='The gpu id to use')
    args = parser.parse_args()

    processors = {
        "racin": RacinProcessor,
    }

    num_labels_task = {
        "racin": 2,
    }

    device = 'cpu'
    
    if torch.cuda.is_available():
        if args.gpuid >= 0 and args.gpuid <= torch.cuda.device_count()-1:
            device = 'cuda:' + str(args.gpuid)
        else:
            gpuid = torch.cuda.device_count()-1
            device = 'cuda:' + str(gpuid)
    else:
        device = 'cpu'

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    tokenizer = T5Tokenizer.from_pretrained(args.t5_model_dir)

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(GOALRECOGNITION_PATH)
        
        random.shuffle(train_examples)
        
        num_train_steps = int(len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
        num_warmup_steps = int(args.warmup_proportion * num_train_steps)

    if args.do_eval:
        eval_examples = processor.get_dev_examples(GOALRECOGNITION_PATH)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
        
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.train_batch_size)
        
    if args.do_test:
        test_dataloader_1 = get_test_dataloader(TEST_1_PATH, label_list, args, tokenizer, processor)
        test_dataloader_2 = get_test_dataloader(TEST_2_PATH, label_list, args, tokenizer, processor)
        test_dataloader_3 = get_test_dataloader(TEST_3_PATH, label_list, args, tokenizer, processor)

    # Prepare model
    model = T5ForConditionalGeneration.from_pretrained(args.t5_model_dir, num_labels = num_labels)

    model.to(device)
    best_eval_acc=0.0
    if args.do_train:
        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)

        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)        

        train_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
        
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, label_ids = batch
                loss = model(input_ids=input_ids, attention_mask=input_mask, labels=label_ids).loss
                loss = loss.mean()
                
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                
                loss.backward()
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    model.zero_grad()
                    scheduler.step()
                    
            logits_all, eval_accuracy = do_evaluation(model, eval_dataloader, args, device, tokenizer, is_training=False)
            print('epoch: {}, eval_acc: {}'.format(epoch, eval_accuracy))
            if best_eval_acc < eval_accuracy:
                best_eval_acc = eval_accuracy

        print('Best eval acc:', best_eval_acc)
    if args.do_test:
            logits_all, test_accuracy_1 = do_evaluation(model, test_dataloader_1, args, device, tokenizer, is_training=False)
            print('test_1_acc: {}'.format(test_accuracy_1))
            logits_all, test_accuracy_2 = do_evaluation(model, test_dataloader_2, args, device, tokenizer, is_training=False)
            print('test_2_acc: {}'.format(test_accuracy_2))
            logits_all, test_accuracy_3 = do_evaluation(model, test_dataloader_3, args, device, tokenizer, is_training=False)
            print('test_3_acc: {}'.format(test_accuracy_3))

            result = "GE_1_2_3_goalrecognition_t5.csv"
            isExist = True
            if not os.path.exists(result):
                isExist = False
            with open(result, "a+") as csvfile:
                writer = csv.writer(csvfile)
                if not isExist:
                    writer.writerow([ "Dev Acc", "Acc on MIX123", "Acc on GE1", "Acc on GE3", "seed"])
                writer.writerow([str(eval_accuracy), str(test_accuracy_1), str(test_accuracy_2), str(test_accuracy_3), str(args.seed)])
            

if __name__ == "__main__":
    main()
