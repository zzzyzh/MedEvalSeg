import torch
import os
import json

from tqdm import tqdm
import gc

from .utils import save_json

class BaseDataset:
  def __init__(self):
    self.chunk_idx = int(os.environ.get("chunk_idx",0))
    self.num_chunks = int(os.environ.get("num_chunks",1))


  def run(self,samples,model,batch_size = 2000):
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

  def cal_matrics(self):
    pass

  def init_dataset(self):
    pass

  def construct_messages(self):
    pass


  def eval(self):
      model = self.model
      dataset_path = self.dataset_path
      output_path = self.output_path
      num_chunks = self.num_chunks
      chunk_idx = self.chunk_idx
      if num_chunks == 1:
          results_path = os.path.join(output_path,"results.json")
          matric_path = os.path.join(output_path,"metrics.json")
          out_samples = self.run(self.samples,model)
          save_json(results_path,out_samples)

          metrics,out_samples = self.cal_metrics(out_samples)
          save_json(matric_path,metrics)
          save_json(results_path,out_samples)
          return metrics


      elif num_chunks > 1:
        results_path = os.path.join(output_path,f"results_{chunk_idx}.json")
        final_results_path = os.path.join(output_path,"results.json")
        out_samples = self.run(self.samples,model)
        save_json(results_path,out_samples)

        total_results_path = os.listdir(output_path)
        total_results_path = [result for result in total_results_path if result.startswith("results_")]
        if len(total_results_path) == num_chunks:
            total_results = []
            for result in total_results_path:
                results_path = os.path.join(output_path,result)
                with open(results_path,"r") as f:
                    total_results.extend(json.load(f))

            save_json(final_results_path,total_results)
            metrics,out_samples = self.cal_metrics(total_results)
            matric_path = os.path.join(output_path,"metrics.json")
            save_json(matric_path,metrics)
            save_json(final_results_path,out_samples)
            return metrics
        else:
            return None
      else:
          raise ValueError("num_chunks must be greater than 0")
