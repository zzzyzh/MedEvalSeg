import torch
import os
import json
import gc

from PIL import Image
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm

from ..utils import save_json,extract,judge_multi_choice
from ..base_dataset import BaseDataset

from ..question_formats import get_multiple_choice_prompt

class MedXpertQA(BaseDataset):
    def __init__(self,model,dataset_path,output_path,split = "MM"):
        self.split = split
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path if dataset_path else "TsinghuaC3I/MedXpertQA"
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))
    

    def load_data(self):
        dataset_path = self.dataset_path
        dataset = load_dataset(dataset_path,self.split)["test"]
        for idx,sample in tqdm(enumerate(dataset)):
            if idx % self.num_chunks == self.chunk_idx:
                sample = self.construct_messages(sample)
                self.samples.append(sample)
        return self.samples

    def construct_messages(self,sample):
        choices = []
        question = sample["question"]
        options = sample["options"]
        answer = sample["label"]
        for option,label in options.items():
            choice = f"{option}. {label}"
            choices.append(choice)
                
        is_reasoning = True if os.environ.get("REASONING","False") == "True" else False
        prompt = get_multiple_choice_prompt(question,choices,is_reasoning)
            
        if "images" in sample:
            images = sample["images"]
            images = [os.path.join(self.dataset_path,"images",image) for image in images]
            images = [Image.open(image) for image in images]
            messages = {"prompt":prompt,"images":images}
            del sample["images"]
        else:
            messages = {"prompt":prompt}
        sample["messages"] = messages
        sample["choices"] = choices
        sample["answer"] = answer
        del sample["label"]
        del sample["options"]
        return sample


    def cal_metrics(self,out_samples):
        total = 0
        right = 0
        total_question_type = defaultdict(int)
        right_question_type = defaultdict(int)
        right_task_type =  defaultdict(int)
        total_task_type = defaultdict(int)
        for i,sample in enumerate(out_samples):
            response = sample["response"]
            choices = sample["choices"]
            answer = sample["answer"]
            question_type = sample["question_type"]
            medical_task = sample["medical_task"]

            total_question_type[question_type] += 1
            total_task_type[medical_task] += 1

            total += 1
            response = extract(response,"answer")
            correct = judge_multi_choice(choices,answer,response)
            out_samples[i]["correct"] = correct
            if correct:
                right_question_type[question_type] += 1
                right_task_type[medical_task] += 1
                right += 1
        
        metrics = {"total":total,"right":right,"acc":right/total}
        question_type_metrics = {}
        for key,value in total_question_type.items():
            right_cnt = right_question_type[key]
            question_type_metrics[key] = {"total":value,"right":right_cnt,"acc":right_cnt/value}

        task_type_metrics = {}
        for key,value in total_task_type.items():
            right_cnt = right_task_type[key]
            task_type_metrics[key] = {"total":value,"right":right_cnt,"acc":right_cnt/value}
        
        metrics = {"total metrics":metrics,"question type metrics":question_type_metrics,"task type metrics":task_type_metrics}
        return metrics,out_samples


                