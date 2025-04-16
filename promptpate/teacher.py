import queue
import numpy as np
from tqdm import tqdm
from math import sqrt, log

class TeacherEnsemble:
    def __init__(self, num_teachers, examples_per_teacher, inference, target_eps, gnmax):
        self.num_teachers = num_teachers
        self.examples_per_teacher = examples_per_teacher
        self.inference = inference
        self.target_eps = target_eps
        self.gnmax = gnmax
        
        self.teacher_prompts = []
        self.privacy_cost = 0.0
        self.queries_answered = 0
    
    def compute_eps_per_query(self, sigma, delta):
        return sqrt(2 * log(1.25 / delta)) / sigma
    
    def create_teacher_prompts(self, private_data):
        if len(private_data) < self.num_teachers:
            self.num_teachers = len(private_data)
            print(f"Number of teachers adjusted to {self.num_teachers}")
        
        np.random.shuffle(private_data)
        
        for i in range(self.num_teachers):
            example = private_data[i]
            self.teacher_prompts.append(f"Query: {example["input"]}, Label: {example["label"]}")
        print("Teacher prompts created successfully.")

    def get_teacher_votes(self, query):
        votes = []
        for prompt in tqdm(self.teacher_prompts):
            prediction, _ = self.inference.query("Output only one word. Given these examples below, predict the label of the given query:", prompt, query)
            votes.append(prediction)
        
        unique_votes, vote_counts = np.unique(votes, return_counts=True)
        return unique_votes, vote_counts
        
    def label_public_data(self, public_data):
        labeled_data = []
        for query in tqdm(public_data):
            if(self.privacy_cost >= self.target_eps):
                print(f"Privacy budget exhausted. Stopping at {self.queries_answered} queries.")
                break
            
            classes, vote_counts = self.get_teacher_votes(query)
            label, passed = self.gnmax.noisy_max(classes, vote_counts)
            
            if passed:
                labeled_data.append({"input": query, "label": label})
                self.queries_answered += 1
                
                # Papernot simplified the privacy cost calculation as shown in the Appendix Theorem 1
                self.privacy_cost += self.compute_eps_per_query(self.gnmax.sigma1, self.gnmax.delta)
                
        print(f"Public data created with privacy cost {self.privacy_cost} and {self.queries_answered} queries.")
        return labeled_data
        
    def create_student_prompt(self, labeled_data, validation_split, num_candidates):
        np.random.shuffle(labeled_data)
        val_size = max(1, int(len(labeled_data) * validation_split))
        val_data = labeled_data[:val_size]
        candidate_data = labeled_data[val_size:]
        
        if len(candidate_data) < num_candidates:
            print(f"Not enough candidate examples. Using {len(candidate_data)} candidates.")
            num_candidates = len(candidate_data)
            
        # Create candidate prompts
        candidate_prompts = []
        for i in range(num_candidates):
            example = candidate_data[i]
            candidate_prompts.append(f"Query: {example["input"]}, Label: {example["label"]}")
                        
        # Evaluate each prompt on validation data
        best_accuracy = -1
        best_prompt = None
        
        for i, prompt in enumerate(candidate_prompts):
            correct = 0
            for query, true_label in val_data:
                prediction, _ = self.inference.query("Output only one word. Given these examples below, predict the label of the given query:", prompt, query)
                if prediction == true_label:
                    correct += 1
                    
            accuracy = correct / len(val_data)
            print(f"Prompt {i+1} validation accuracy: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_prompt = prompt
                
        return best_prompt, best_accuracy