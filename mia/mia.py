import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper.inference import Inference
import pandas as pd

class MIA:
    def __init__(self, config):
        self.inference = Inference(config)
        self.dataset_loc = "./dataset-cache/"
        self.member_log = "member_logs.out"
        self.non_member_log = "non_member_logs.out"
        pass
    
    def log(self, EXAMPLE, PROMPT, OUTPUT, MEMBER, LOGPROB):
        if(MEMBER):
            with open(self.member_log, "a") as f:
                f.write(f"{EXAMPLE},{PROMPT},{MEMBER},{LOGPROB}\n")
        else:
            with open(self.non_member_log, "a") as f:
                f.write(f"{EXAMPLE},{PROMPT},{MEMBER},{LOGPROB}\n")
        
    def attack(self, datasets = ["agnews", "dbpedia", "sst2", "trec"]):
        for dataset in datasets:
            dataset_path = self.dataset_loc + dataset + ".csv"
            df = pd.read_csv(dataset_path)
            
            df = df.sample(n=1000, random_state=42)
            
            # Shuffle the dataset
            df = df.sample(frac=1, random_state=42)
            
            # Query and get the confidence / logprob of the model - SAME MEMBER
            INSTRUCTION = "Output only one word. Given these examples below, predict the label of the given query:"
            for _, row in df.iterrows():
                EXAMPLE = f"EXAMPLE:\nQuery: {row["text"]}, Label: {row["label"]}"
                # Same member
                PROMPT = f"Query: {row["text"]}, Label: "
                
                output, logprob = self.inference.query(INSTRUCTION, EXAMPLE, PROMPT)
                # Log if output is correct
                if(output == row["label"]):
                    self.log(EXAMPLE, PROMPT, output, True, logprob)
                    
                # Non-member
                random_row = df.sample(n=1, random_state=42).iloc[0]
                PROMPT = f"Query: {random_row['text']}, Label: "
                output, logprob = self.inference.query(INSTRUCTION, EXAMPLE, PROMPT)
                # Log if output is correct
                if(output == row["label"]):
                    self.log(EXAMPLE, PROMPT, output, False, logprob)
                    
                # Log to stdout
                print("Completed an MIA Iteration.")