# from roberta import Roberta
from transformers import (
                    RobertaConfig,
                    RobertaTokenizer,
                    RobertaForSequenceClassification,
                    RobertaModel,
                    AutoTokenizer,
                    AutoConfig,
                    AutoModelForSequenceClassification)

def get_model(args):
    pass

if __name__=='__main__':
    name = 'roberta-large'
    conf = AutoConfig.from_pretrained(name)
    conf.num_labels = 30
    model = RobertaModel(conf)
    breakpoint()
    print('modeling')