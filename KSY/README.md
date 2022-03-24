### 코드 수정 사항
> last updated: 03/24

### 03/ 23-4
* feat: train test split three ways
    * (split-basic) using sklearn train_test_split with stratified
    * (split-dup) uniq sentence stratified split. duplicated split by numbers of sentences by ratio
    * (split-eunki) stratified first, and if sentences in vals are also in trains, then swap so that no duplicated sentences in both vals and trains.
    
* feat : wandb cm matrix visualization with WandbCallback
    * issue: step point works weired... 
### 03/ 22
* feat: seed fix, mkdirs, arguments, scripts examples