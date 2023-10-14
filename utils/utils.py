import logging

from tqdm import tqdm
from transformers import ProgressCallback


class myProgressCallback(ProgressCallback):
    def __init__(self):
        logging.basicConfig(filename="/data/hanzhi/output/logfile", level=logging.INFO)
        super().__init__()


    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.training_bar = tqdm(total=state.max_steps, dynamic_ncols=True)
        self.current_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.training_bar.update(state.global_step - self.current_step)
            self.current_step = state.global_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs.keys():
            self.training_bar.set_description(f"Training Epoch: {int(logs['epoch'])}, (loss: {logs['loss']})")
        logging.info(state)
        logging.info(logs)
        print(logs)