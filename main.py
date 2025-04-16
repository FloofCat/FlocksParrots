import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mia.mia import MIA
from helper.config import Config
from promptdpsgd.pdpsgd import PDPSGD
from promptpate.ppate import PPATE

def main():
    config = Config("config.json").config
    
    # Check the current experiment
    experiment = config["current_run"]
    if experiment == "mia":
        mia = MIA(config)
        mia.attack()
    elif experiment == "promptdpsgd":
        pdpsgd = PDPSGD(config)
        pdpsgd.train()
        pdpsgd.inference()
    elif experiment == "promptpate":
        ppate = PPATE(config)
        best_prompt, val_acc = ppate.train()
        ppate.predict(best_prompt)
    else:
        raise ValueError("Invalid experiment")
    
    print("Experiment completed successfully.")
    
main()
    