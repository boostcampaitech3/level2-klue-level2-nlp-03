from transformers import Trainer, TrainingArguments
# from custom.callback import customTrainerCallback
from loss import get_loss
from load_data import get_cls_list

class customTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(customTrainer, self).__init__(*args, **kwargs)

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):

        if self.control.should_log:
            # if is_torch_tpu_available():
            #     xm.mark_step()
            logs  = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None

        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            # metrics에 confusion matrix를 추가한 값들이 self.state.log_history에 추가되었음
            # 이 부분이 self._save_checkpoint에서 self.state를 저장하기 때문에 figure 값은 제거하려함.
            # history가 list로 관리됨 ->  train, eval 순서로 append 때문에 eval은 마지막 요소에서 pop
            self.state.log_history[-1].pop('eval_cm_ratio')
            self.state.log_history[-1].pop('eval_cm_samples')
            # for history in self.state.log_history:
            #     history
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """

        # 여기서 뭔가 인풋을 넣어주면 될 것 같다..!

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss



class customTrainer2(Trainer):
    def __init__(self, add_args,cls_list=None, *args, **kwargs):
        super(customTrainer2, self).__init__(*args, **kwargs)

        if add_args.loss_fn=='ldamloss':
            self.label_smoother = get_loss(add_args, cls_list)

        elif add_args.loss_fn =='focalloss':
            self.label_smoother = get_loss(add_args, cls_list)
        elif add_args.loss_fn =='labelsmoothingloss':
            self.label_smoother = get_loss(add_args, cls_list)
        elif add_args.loss_fn =='base':
            pass
        else:
            raise NotImplementedError

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):

        if self.control.should_log:
            # if is_torch_tpu_available():
            #     xm.mark_step()
            logs  = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None

        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            # metrics에 confusion matrix를 추가한 값들이 self.state.log_history에 추가되었음
            # 이 부분이 self._save_checkpoint에서 self.state를 저장하기 때문에 figure 값은 제거하려함.
            # history가 list로 관리됨 ->  train, eval 순서로 append 때문에 eval은 마지막 요소에서 pop
            # self.state.log_history[-1].pop('eval_cm_ratio')
            # self.state.log_history[-1].pop('eval_cm_samples')
            # for history in self.state.log_history:
            #     history
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss