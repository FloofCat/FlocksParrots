import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper.inference import Inference
import pandas as pd
from .teacher import TeacherEnsemble
from .gnmax import GNMax

class PPATE:
    def __init__(self, config):
        self.inference = Inference(config)
        self.config = config["promptpate"]
        
        self.n_teachers = self.config["num_teachers"]
        self.examples_per_teacher = self.config["examples_per_teacher"]
        self.sigma_1 = self.config["sigma_1"]
        self.sigma_2 = self.config["sigma_2"]
        self.threshold = self.config["threshold"]
        self.target_eps = self.config["target_eps"]
        self.delta = self.config["delta"]
        self.dataset_loc = "./dataset-cache/"
        self.candid_prompts = self.config["candid_prompts"]
        self.infer_dataset = self.config["infer_dataset"]
        self.gnmax = GNMax(self.sigma_1, self.sigma_2, self.threshold, self.delta)
        
        self.private_data, self.public_data = self.load_dataset(self.config["dataset_path"])
        self.load_iid_ood()
    def load_dataset(self, dataset_path):
        df = pd.read_csv(self.dataset_loc + dataset_path + ".csv")
        
        public_data = []
        private_data = []
        
        # Put 100 samples in public_data
        for index, row in df.iterrows():
            if index < 100:
                public_data.append({"input": row["text"], "label": row["label"]})
            else:
                private_data.append({"input": row["text"], "label": row["label"]})
        
        private_data = private_data[:100]
        
        return private_data, public_data
        
    def load_iid_ood(self):
        self.infer_dataset = pd.read_csv(self.dataset_loc + self.infer_dataset + ".csv")
        
        self.infer_dataset = self.infer_dataset.sample(frac=1).reset_index(drop=True)
        
        self.infer_dataset = self.infer_dataset[:500]
    
    def _train(self, private_data, public_data):
        self.teacher = TeacherEnsemble(self.n_teachers, self.examples_per_teacher, self.inference, self.target_eps, self.gnmax)
        self.teacher.create_teacher_prompts(private_data)
        
        labeled_data = self.teacher.label_public_data(public_data)
        
        best_prompt, val_acc = self.teacher.create_student_prompt(labeled_data, 0.2, self.candid_prompts)
        
        return best_prompt, val_acc
    
    def train(self):
        best_prompt, val_acc = self._train(self.private_data, self.public_data)
        print("Best Prompt:", best_prompt)
        print("Validation Accuracy:", val_acc)
        
        return best_prompt, val_acc
    
    def _predict(self, query, best_prompt):
        print("Predicting...")
        predictions, _ = self.inference.query("Output only one word. Given these examples below, predict the label of the given query:", best_prompt, query)
        return predictions

    def predict(self, best_prompt):
        for i, query in enumerate(self.infer_dataset["text"]):
            predictions = self._predict(query, best_prompt)
            print("Query Number:", i+1)
            print("Query:", query)
            print("Predictions:", predictions)
            print("---------------------------")