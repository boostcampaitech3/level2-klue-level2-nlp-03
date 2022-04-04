from transformers import Trainer, TrainingArguments
# from custom.callback import customTrainerCallback
from loss import get_loss
from load_data import get_cls_list

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

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

class customTrainer3(Trainer):
    def __init__(self, add_args,cls_list=None, *args, **kwargs):
        super(customTrainer3, self).__init__(*args, **kwargs)

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
            self.state.log_history[-1].pop('eval_cm_ratio')
            self.state.log_history[-1].pop('eval_cm_samples')
            # for history in self.state.log_history:
            #     history
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].
        Will use no sampler if `self.train_dataset` does not implement `__len__`, a random sampler (adapted to
        distributed training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        breakpoint()
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        breakpoint()
        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def training_step(self, model, inputs):
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        breakpoint()
        inputs = self._prepare_inputs(inputs)
        breakpoint()
        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.do_grad_scaling else None
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        breakpoint()
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