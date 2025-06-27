import json
import math
import sys
import copy
import re
import os
import json
import difflib
import asyncio
import random


from tqdm.asyncio import tqdm_asyncio
from nltk.translate.meteor_score import single_meteor_score
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from mathruler.grader import extract_boxed_content

from collections import defaultdict, Counter
from openai import AzureOpenAI, OpenAI,AsyncAzureOpenAI,AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
)


def tokenize(text):
    text = text.lower().replace(".", " .").split(" ")
    return text

def bleu(pred,target,n):
    weights=[1/n for _ in range(n)]
    tokenized_target = tokenize(target)
    tokenized_pred = tokenize(pred)
    return sentence_bleu([tokenized_target], tokenized_pred, weights=weights)

def rouge(pred,target):
    rouge_scorer = Rouge()
    rouge_scores = rouge_scorer.get_scores(pred.lower(), target.lower())
    return rouge_scores



def get_compare_messages(question,response,answer):
    prompt = f"""
Your task is to determine whether the user's answer is correct based on the provided questions and standard answers (for example, if the user expresses a similar meaning to the standard answer, or another interpretation of the standard answer, it is considered correct.)

The question is: {question}

The standard answer: {answer}

The user's answer: {response}

Please strictly follow the following format for output(0 represents correct, 1 represents incorrect):
<think>{{your concise think step}}</think>
<judge>{{0/1}}</judge>

for example:
<think>The standard answer is right, and the user's answer is right frontal lobe, they express the same meaning, so it is correct.</think>
<judge>0</judge>
    """
    messages = [{"role":"user","content":prompt}]
    return messages


def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()

def find_most_similar_index(str_list, target_str):
    """
    Given a list of strings and a target string, returns the index of the most similar string in the list.
    """
    # Initialize variables to keep track of the most similar string and its index
    most_similar_str = None
    most_similar_index = 0
    highest_similarity = 0
    
    # Iterate through each string in the list
    for i, str in enumerate(str_list):
        # Calculate the similarity between the current string and the target string
        similarity = str_similarity(str, target_str)
        
        # If the current string is more similar than the previous most similar string, update the variables
        if similarity > highest_similarity:
            most_similar_str = str
            most_similar_index = i
            highest_similarity = similarity
    
    # Return the index of the most similar string
    return most_similar_index

def judge_multi_choice(choices,answer,response,alphas = None):
    response = response.lower()
    if response.split("\n\n")[0] in [chr(ord('a') + i) for i in range(len(choices))]:
        response = response.split("\n\n")[0]
    elif response.split("\n\n")[-1].split(".")[0] in [chr(ord('a') + i) for i in range(len(choices))]:
        response = response.split("\n\n")[-1].split(".")[0]
    
    response = parse_response(response)
    alphas = [chr(ord('a') + i) for i in range(len(choices))]
    choices = [choice.lower() for choice in choices]
    flag = False
    response = response.strip().lower()
    response = response.replace("\n","")
    split_response = response.split(".")[0]
    split_response = split_response.split(":")[-1]
    answer = answer.strip().lower()
    
    if len(split_response) > 300:
        flag = False
    # letter,letter.  choice,choice
    if split_response == answer:
        flag = True
    
    # letter,choice
    elif split_response in alphas:
        if choices[ord(split_response)-ord("a")]== answer:
            flag = True
    
    # choice letter
    elif split_response in choices:
        if answer in alphas and split_response == choices[ord(answer)-ord("a")]:
            flag = True
    # unparsed
    else:
        index = find_most_similar_index(choices,response)
        if alphas[index] == answer or choices[index] == answer:
            flag = True
    return flag


def parse_response(response):
    response = response.lower()
    if "boxed" in response:
        response = extract_boxed_content(response)
    elif "<answer>" in response:
        response = extract(response,"answer")
    answer_patterns = [
        "**answer**:",
        "**answer**",
        "*answer*:",
        "**answer:**",
        "answer is",
        "answer:",
        "答案:",
        "final answer",
        "final answer is"
    ]
    for answer_pattern in answer_patterns:
        if answer_pattern in response:
            response = response.split(answer_pattern)[-1]
    
    return response


def judge_close_end_vqa(answer,response):
    answer = answer.lower()
    response = parse_response(response)
    response = response.replace("\n","").replace(".","")
    if response == answer:
        return True
    else:
        return False

def judge_judgement(answer,response):
    answer = answer.lower()
    response = parse_response(response)
    response = response.replace("\n","").replace(".","")
    if ('yes' in response) ^ ('no' in response):
        if answer in response:
            return True
    return False


def judge_open_end_vqa(answer,response):
    answer = answer.lower()
    response = parse_response(response)
    bleu1 = bleu(response,answer,1)
    bleu2 = bleu(response,answer,2)
    bleu3 = bleu(response,answer,3)
    bleu4 = bleu(response,answer,4)

    em = response == answer
    rouge_scores = rouge(response,answer)
    rouge_1 = rouge_scores[0]["rouge-1"]["f"]
    rouge_2 = rouge_scores[0]["rouge-2"]["f"]
    rouge_l = rouge_scores[0]["rouge-l"]["f"]

    precision,recall,f1 = calculate_f1(response,answer)


    return {
        "em" : em,
        "bleu1" : bleu1,
        "bleu2" : bleu2,
        "bleu3" : bleu3,
        "bleu4" : bleu4,
        "rouge1" : rouge_1,
        "rouge2" : rouge_2,
        "rougel" :  rouge_l,
        "precision": precision,
        "recall": recall,
        "f1" :f1         
    }


def calculate_f1(prediction, ground_truth):
    prediction_tokens = set(prediction.lower().split())
    ground_truth_tokens = set(ground_truth.lower().split())
    
    common = prediction_tokens & ground_truth_tokens
    
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return 0
    
    precision = len(common) / len(prediction_tokens)
    recall = len(common) / len(ground_truth_tokens)
    
    if precision + recall == 0:
        return 0,0,0
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1,precision,recall





def get_content_between_a_b(start_tag, end_tag, text):
    extracted_text = ""
    start_index = text.find(start_tag)
    while start_index != -1:
        end_index = text.find(end_tag, start_index + len(start_tag))
        if end_index != -1:
            extracted_text += text[start_index + len(start_tag) : end_index] + " "
            start_index = text.find(start_tag, end_index + len(end_tag))
        else:
            break

    return extracted_text.strip()


def extract(text, type,hard = True):
    if text:
        target_str = get_content_between_a_b(f"<{type}>", f"</{type}>", text)
        if target_str:
            return target_str
        elif hard:
            return text
        else:
            return ""
    else:
        return ""

# DATA SAVING
def save_json(filename, ds):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(ds, f, indent=4, ensure_ascii=False)

class fake_response:
    def __init__(self,usage):
        self.usage = usage

def before_retry_fn(retry_state):
    if retry_state.attempt_number > 1:
        print(f"Retrying API call. Attempt #{retry_state.attempt_number}, f{retry_state}")

async def deal_tasks(tasks, max_concurrent_tasks=500):
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    results = []

    async def sem_task(task):
        async with semaphore:
            return await task  

    sem_tasks = [sem_task(task) for task in tasks]

    for coro in tqdm_asyncio.as_completed(sem_tasks, total=len(sem_tasks)):
        result = await coro
        results.append(result)

    return results


class openai_llm:
    def __init__(self,model = None):
        if model is None:
            model = os.environ.get("judge_gpt_model","gpt-4.1-2025-04-14")

        self.model = model

        api_key = os.environ["openai_api_key"]

        self.client = OpenAI(
            api_key=api_key,
            base_url="http://38.97.60.202:9088/v1"
            )
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url="http://38.97.60.202:9088/v1"
            )
    
    @retry(wait=wait_fixed(10), stop=stop_after_attempt(1000), before=before_retry_fn)
    def response(self,messages,**kwargs):
        response = self.client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=messages,
            n = kwargs.get("n", 1),
            temperature= kwargs.get("temperature", 0),
            max_tokens=kwargs.get("max_tokens", 4000),
            timeout=kwargs.get("timeout", 180)
        )
        return response.choices[0].message.content
    

    
    @retry(wait=wait_fixed(10), stop=stop_after_attempt(1000), before=before_retry_fn)
    async def response_async(self,messages,**kwargs):
        response = await self.async_client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=messages,
            n = kwargs.get("n", 1),
            temperature= kwargs.get("temperature", 0),
            max_tokens=kwargs.get("max_tokens", 4096),
            timeout=kwargs.get("timeout", 180)
        )      
        return response.choices[0].message.content
    
    def generate_output(self,messages,**kwargs):
        try:
            response = self.response(messages,**kwargs)
        except Exception as e:
            response = None
            print(f"get {kwargs.get('model', self.model)} response failed: {e}")
        return response
    
    async def generate_output_async(self,idx, messages,**kwargs):
        try:
            response = await self.response_async(messages,**kwargs)
        except Exception as e:
            response = None
            print(f"get {kwargs.get('model', self.model)} response failed: {e}")
        return idx,response
    
    def generate_outputs(self,messages,**kwargs):
        tasks = [self.generate_output_async(i,messages[i],**kwargs) for i in range(len(messages))]
        results = asyncio.run(deal_tasks(tasks))
        results = sorted(results, key=lambda x: x[0])
        results = [x[1] for x in results]
        return results

judger = openai_llm()
