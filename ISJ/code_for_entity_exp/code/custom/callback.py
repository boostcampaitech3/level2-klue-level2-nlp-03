from transformers.integrations import WandbCallback
from transformers.trainer_callback import TrainerState,TrainerCallback,TrainerControl, CallbackHandler
import wandb

def rewrite_logs(d):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    test_prefix = "test_"
    test_prefix_len = len(test_prefix)
    for k, v in d.items():
        if k.startswith(eval_prefix):
            new_d["eval/" + k[eval_prefix_len:]] = v
        elif k.startswith(test_prefix):
            new_d["test/" + k[test_prefix_len:]] = v
        else:
            new_d["train/" + k] = v
    return new_d

class customWandbCallback(WandbCallback):
    def __init__(self):
        " Training, Eval 때 confusion matrix는 Figure class라서 예외처리 추가"
        super(customWandbCallback, self).__init__()

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if self._wandb is None:
            return

        if not self._initialized:
            self.setup(args, state, model)

        if state.is_world_process_zero:
            logs = rewrite_logs(logs)
            # 이렇게 안하면 자꾸 wandb의 train/loss, learning rate이 train/global_step에 먹힌다;
            if len(logs.keys()) == 3:
                self._wandb.log({**logs, "train/global_step": state.global_step})
            else:
                for metric, vals in logs.items():
                    if isinstance(vals, float) or isinstance(vals, int):
                        self._wandb.log({metric:vals})
                    else:
                        self._wandb.log({metric: [self._wandb.Image(vals)]})
                    # self._wandb.log({**logs, "train/global_step": state.global_step})
            # self._wandb.log({
            #     "train/global_step": state.global_step
            # })


## Not yet using

# class customTrainerState(TrainerState):
#     def __init__(self):
#         super(customTrainerState, self).__init__()
#
#     def save_to_json(self, json_path:str):
#
#         """Save the content of this instance in JSON format inside `json_path`."""
#         json_string = json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"
#         with open(json_path, "w", encoding="utf-8") as f:
#             f.write(json_string)
#
# class customTrainerControl(TrainerControl):
#     def __init__(self):
#         super(customTrainerControl, self).__init__()
#
#
# class customTrainerCallback(CallbackHandler):
#     def __init__(self,*args, **kwargs):
#         super(customTrainerCallback, self).__init__(*args, **kwargs)
#
#
#     def on_save(self, args, state,  control):
#         control.should_save = False
#         return self.call_event("on_save", args, state, control)
