import torch
import os
import json
import gc
import csv

from PIL import Image
from datasets import load_dataset,load_from_disk
from collections import defaultdict
from tqdm import tqdm

from ..utils import save_json,judge_multi_choice
from ..base_dataset import BaseDataset

from ..question_formats import get_multiple_choice_prompt

class MedFrameQA(BaseDataset):
    def __init__(self,model,dataset_path,output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path if dataset_path else "SuhaoYu1020/MedFrameQA"
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))


    def run(self,samples,model,batch_size = 500):
      out_samples = []
      with torch.no_grad():
          messages_list = []
          current_messages = []
          current_samples = []
          for sample in tqdm(samples):
              messages = sample["messages"]
              current_messages.append(messages)
              current_samples.append(sample)
              if len(current_messages) >= batch_size:
                  messages_list.append([current_messages,current_samples])
                  current_messages = []
                  current_samples = []
          if current_messages:
              messages_list.append([current_messages,current_samples])
          
          for current_messages,current_samples in tqdm(messages_list):
              outputs = model.generate_outputs(current_messages)
              try:
                  for sample,response in zip(current_samples,outputs):
                      del sample["messages"]
                      sample["response"] = response
                      out_samples.append(sample)   
              except Exception as e:
                  from pdb import set_trace;set_trace()
                  print(e)
              gc.collect()
      return out_samples
    
    def load_data(self):
        dataset = load_dataset(self.dataset_path)["test"]
        for idx,sample in tqdm(enumerate(dataset)):
            if idx % self.num_chunks == self.chunk_idx:
                sample = self.construct_messages(sample)
                self.samples.append(sample)
        return self.samples


    def construct_messages(self,sample):

        question = sample["question"]
        choices = sample["options"]
        answer = sample["correct_answer"]
        image_1 = sample["image_1"]
        image_2 = sample["image_2"]
        image_3 = sample["image_3"]
        image_4 = sample["image_4"]
        image_5 = sample["image_5"]

        choices = [f"{chr(65+i)}.{choices[i]}" for i in range(len(choices))]

        images = [image_1,image_2,image_3,image_4,image_5]
        images = [image for image in images if image is not None]

        is_reasoning = True if os.environ.get("REASONING","False") == "True" else False
        prompt = get_multiple_choice_prompt(question,choices,is_reasoning)

        messages = {"prompt":prompt,"images":images}
        sample["messages"] = messages
        sample["choices"] = choices
        sample["answer"] = answer
        del sample["options"]
        del sample["correct_answer"]
        del sample["image_1"]
        del sample["image_2"]
        del sample["image_3"]
        del sample["image_4"]
        del sample["image_5"]
        return sample


    def cal_metrics(self,out_samples):
        total = 0
        right = 0
        total_modality = defaultdict(int)
        right_modality = defaultdict(int)
        for i,sample in enumerate(out_samples):
            response = sample["response"]
            choices = sample["choices"]
            answer = sample["answer"]

            correct = judge_multi_choice(choices,answer,response)
            out_samples[i]["correct"] = correct
            if correct:
                right += 1
            total += 1

        metrics = {"total metrics":{"total":total,"right":right,"acc":right/total}}
        return metrics,out_samples


                