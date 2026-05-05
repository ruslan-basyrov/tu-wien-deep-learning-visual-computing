import torch
import wandb
import os

class WandBLogger:

    def __init__(self, enabled=True, 
                 model: torch.nn.modules=None, 
                 run_name: str=None,
                 config = None) -> None:
        
        self.enabled = enabled

        if self.enabled:
            api_key = os.getenv("WANDB_API_KEY")
            if api_key:
                wandb.login(key=api_key)

            wandb.init(
                entity=os.getenv("WANDB_ENTITY", "ruslanbasyrov-tu-wien"),
                project=os.getenv("WANDB_PROJECT", "dlvs-assignment-1"),
                group=os.getenv("WANDB_GROUP", "experiments"),
                name=run_name,
                config=config,
            )

            if model is not None:
                self.watch(model)         
            
    def watch(self, model, log_freq: int=1):
        wandb.watch(model, log="all", log_freq=log_freq)
            

    def log(self, log_dict: dict, commit=True, step=None):
        if self.enabled:
            if step:
                wandb.log(log_dict, commit=commit, step=step)
            else:
                wandb.log(log_dict, commit=commit)
 

    def finish(self):
        if self.enabled:
            wandb.finish()
