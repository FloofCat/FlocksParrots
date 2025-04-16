import numpy as np

class GNMax:
    def __init__(self, sigma1, sigma2, threshold, delta):
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.threshold = threshold
        self.delta = delta
    
    def noisy_max(self, classes, vote_counts):
        noisy_counts = vote_counts + np.random.normal(0, self.sigma1, size=len(vote_counts))
        
        max_idx = np.argmax(noisy_counts)
        max_count = vote_counts[max_idx]
        
        threshold_noise = np.random.normal(0, self.sigma2)
        passed = (max_count + threshold_noise) > self.threshold
        
        if passed:
            return classes[max_idx], True
        else:
            return None, False
        