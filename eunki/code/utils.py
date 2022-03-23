def logging_with_wandb(epoch, train_loss, train_f1_score, train_auprc, valid_loss, valid_f1_score, valid_auprc):
  """
    wandb에 학습 결과 기록
  """
  wandb.log({
    f"epoch": epoch,
    f"train_loss": train_loss,
    f"train_f1": train_f1_score,
    f"train_auprc": train_auprc,
    f"valid_loss": valid_loss,
    f"valid_f1": valid_f1_score,
    f"valid_auprc": valid_auprc,
    })

class Custom_Trainer(Trainer):
    def __init__(self, loss_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_name= loss_name
    
    def compute_loss(self, model, inputs, return_outputs= False):
        labels= inputs.pop('labels')
        outputs= model(**inputs)
        device= torch.device('cuda:0' if torch.cuda.is_available else 'cpu:0')
        
        if self.args.past_index >=0:
            self._past= outputs[self.args.past_index]

        if self.loss_name== 'CrossEntropyLoss':
            custom_loss= torch.nn.CrossEntropyLoss().to(device)
            loss= custom_loss(outputs['logits'], labels)
        
        elif self.loss_name== 'LabelSmoothLoss' and self.label_smoother is not None:
            loss= self.label_smoother(outputs, labels)
            loss= loss.to(device)
        
        return (loss, outputs) if return_outputs else loss