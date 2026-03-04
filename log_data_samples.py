import wandb 
import numpy as np 
from src.utils.data_loader import load_dataset 

wandb.init(project = "DA6401_MLP", name="data_exploration")

X_train, _ , y_train, _ =  load_dataset("mnist") 

table = wandb.Table(columns=["image","label"]) 

for digit in range(10): 
    indices = np.where(y_train == digit)[0][:5]
    for idx in indices: 
        img = X_train[idx].reshape(28,28) 
        table.add_data(wandb.Image(img),digit) 
wandb.log({"Sample Images": table}) 
wandb.finish()
