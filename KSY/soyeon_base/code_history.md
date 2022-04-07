### Code Histroeis
> last updated: 04/07

> notebooks updated : done
> 
### TODO
- [x] multi-sentence, single sentence -> single sentence for RBERT
- [x] more elaborately entity-marking(e.g, punctuation, entity typing) -> entity marking by sujeong, entity embeddig by soyeon
- [x] model-tail design(e.g, stacking additional layers) - RBERT + modified LSTM; dropout, stacked FC
- [ ] entity grouping & each model for corresponding grouping or hierarchical -> limited time
- [ ] ner + gives penalty when entity pairs are never to happen - mentor suggested -> limited time


### 04/ 04-06
* entity embedding
* RBERT

### 04/03
* code integration
* model design undergoing

### 03/31~04/03
* code integration & experiments with ensemble version code

### 03/30
* minor updates. using custom model and labelsmoothing loss

### 03/ 29(afternoon)
* add notebooks for trainer/ custom model

### 03/ 29(morning)
* fix: roberta model running on both cases -> please refer `models/custom_roberta.py`
    * more dense: stacking with two fc layers. increasing hidden dim and decreasing hidden dim.
    * lstm : several options exists. 
      * For now, every token embedding feeds into lstm and use final hidden states(context vector)

### 03/ 28
* feat: label smoothing training
    * add `smoothing` factor in `arguments.py`
* feat: customizing roberta model, using pretrained class
    * please refer to `models/custom_roberta.py`
    - [x] lstm running policy


### 03/ 26(afternoon) + examples should be removed
* feat: entity markers in sentences -> refer to `load_data.py` & notebooks(uploaded privately)
  - [ ] issue : added special tokens don't get maksed.
  
```
'〈Something〉는 [E2]조지 해리슨[/E2]이 쓰고 [E1]비틀즈[/E1]가 1969년 앨범 《Abbey Road》에 담은 노래다.'
```

```shell
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
sub_obj = ["[E1]", "[/E1]","[E2]", "[/E2]"]
tokenizer.add_special_tokens({
                            "additional_special_tokens": sub_obj
                            })
print(*out[1].tokens)
print(*out[1].special_tokens_mask)

[CLS] 민주 ##평 ##화 ##당 [SEP] 대안 ##신 ##당 [SEP] 호남 ##이 기반 ##인 바른 ##미 ##래 ##당 · [E2] 대안 ##신 ##당 [/E2] · [E1] 민주 ##평 ##화 ##당 [/E1] 이 우여곡절 끝 ##에 합당 ##해 민생 ##당 ( 가칭 ) 으로 재 ##탄 ##생 ##한다 .
1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
```

* test : anyway, doing pilot test based on this version.

### 03/ 25
* feat : loss test- focal, ldam
    * train.py (+train_custom.py) : two losses require to receive `cls_distribution`. 
    * loss.py : add two loses
        * ldam, focal need weighting function.
        * focal works better when not weighting (need to check!)
    * custom/trainer.py : overriding trainer; revised `compute_loss` 
                          to exploit corresponding losses
* test : 
    * arguments.py : corresponding arguments.
  
### 03/ 23-4
* feat: train test split three ways
    * (split-basic) using sklearn train_test_split with stratified
    * (split-dup) uniq sentence stratified split. duplicated split by numbers of sentences by ratio
    * (split-eunki) stratified first, and if sentences in vals are also in trains, then swap so that no duplicated sentences in both vals and trains.
    
* feat : wandb cm matrix visualization with WandbCallback
    * issue: step point works weired... 
### 03/ 22
* feat: seed fix, mkdirs, arguments, scripts examples