<img width="498" alt="image" src="https://user-images.githubusercontent.com/81913386/165032471-0d975c67-ebf2-4941-884f-64db62e2b68a.png">


## 프로젝트 개요

> 관계 추출(Relation Extraction) - 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 문제

이번 대회에서는 문장, 단어에 대한 정보를 통해 ,문장 속에서 단어 사이의 관계를 추론하는 모델을 학습시키고 모델이 단어들의 속성과 관계를 파악하며 개념을 학습하도록 만드는 것이 목표였다.
> 

---

## 프로젝트 팀 구성 및 역할

> 김은기 : 협업 리딩, EDA를 통한 데이터 탐색, baseline 코드에 기능 연결 및 실험 환경 세팅, 새로운 방식의 stratify k fold 구현, 실험 진행
> 

> 김상렬 : 데이터 분석 및 Back-Translation과 EDA, AEDA를 통한 데이터 증강, 실험 진행
> 

> 김소연: EDA& confusion matrix 시각화, huggingface 호환되도록 모델 구조 및 커스텀 트레이너(손실함수, wandbcallback) 수정, 실험 수행
> 

> 박세연 : BiLSTM 오류 제안, Lite 활용한 AMP 적용, 실험에 잘 맞는 loss 수정, 프로젝트 중간중간 코드 수합, 로드맵 제작, 실험 진행
> 

> 임수정 : Entity 관련 EDA 진행, 전처리 방법의 유효성 확인, Entity Tagging 구현 및 실험 진행, Entity Embedding을 위한 Entity 토큰 인덱스 정보 관련 코드 구현
> 

---

### 최종

- Public 3등, Private 5등

<img width="492" alt="image" src="https://user-images.githubusercontent.com/81913386/165032752-82c13af6-3a67-4402-ae34-625168f04667.png">

<img width="495" alt="image" src="https://user-images.githubusercontent.com/81913386/165032785-6072bbad-eec0-452c-a2c2-0e6f1726d0bb.png">


[기능과 성능 추이](https://www.notion.so/a76d50093c494ffd96c27acca388ba0c)

---

## 프로젝트 수행 절차 및 방법

- 사전 조사 단계
    - HuggingFace Documents를 읽어보며 이해
    - KLUE의 Github과 논문 서치를 통한 데이터 셋에 대한 이해와 SOTA 모델 탐색
- EDA 및 전처리 그리고 모델링 준비 단계
    - 주어진 데이터의 분포를 시각화 및 파악하고 중복 및 오태깅된 데이터 파악 및 전처리
    - 불균형한 데이터셋을 해결해주기 위한 방법 고안
    - 실험에 사용할 모델들 리스트업
- 모델링과 실험 초기 단계
    - Stratify K fold 구현
    - Soft Voting을 통한 Ensemble 구현
    - Pytorch를 통한 GPU 효율화
    - 기존 Loss를 비롯해 Label Smoothing, Focal loss 등을 구현
- 실험 중반 시기
    - 가장 좋은 성능을 보이는 모델 선택(RoBerta)
    - RoBerta를 상속받아 LSTM, Dense Layer 등
    - 가장 좋은 성능을 보이는 Label Smoothing Loss 선택 및 Smoothing 수치 탐색
- 실험 후반 시기
    - LSTM, ModifiedLSTM에 layer를 추가해 모델의 성능 업데이트
    - Entity Tagging 방법 다양화
    - 최종 제출물을 위해 모델(Modified LSTM , More_dense), 데이터(preprocess, augmentation, 원본) Entity, Tagging (typed_entity_marker_punc, entity_marker_punc) 등 다양한 모델들에 대해 Ensemble 진행

## 프로젝트 수행 결과

### 사전조사 단계

- HuggingFace Documents를 살펴보며 이해
    - [https://huggingface.co/klue](https://huggingface.co/klue)
- KLUE의 Github과 논문 서치를 통한 데이터 셋에 대한 이해와 SOTA 모델 탐색
    - [https://github.com/KLUE-benchmark](https://github.com/KLUE-benchmark)

### EDA 및 전처리 그리고 모델링 준비 단계

- 주어진 데이터의 분포를 시각화 및 파악하고 중복 및 오태깅된 데이터 파악 및 전처리
    - cleansing 작업을 통해  f1-socre를 0.3 정도 상승시킬 수 있었다.
    
    <img width="465" alt="image" src="https://user-images.githubusercontent.com/81913386/165032536-db45ace7-3e21-4df8-bb91-f68a97cb078f.png">
    
- Entity를 중심으로 세부 EDA 진행
    - 관계 예측에서 가장 중요한 데이터인 Entity들의 양상을 파악하고자 Entity 중심 EDA 진행
    - 한국어 Dataset이지만 영어, 중국어, 그리스어, 숫자, 특수문자 등 한국어가 아닌 문자로 구성된 Entity도 있다는 사실을 알게 되었고, 이를 통해 Entity를 보존하기 위해서는 Back translation이나 전처리 단계에서 Entity를 특정 토큰으로 치환하여 보호하는 방법을 생각하게 되었다.
        
        <img width="435" alt="image" src="https://user-images.githubusercontent.com/81913386/165032578-da3b3f47-7dd9-412b-8d5b-b461d58e71f1.png">

        
- Back translation
    - `Google translation api`를 이용해서 한국어에서 각각 영어, 일본어로 back-translation을 진행하엿다. back-translation을 진행하는 도중 entity들이 훼손되지 않게 Masking 작업을 진행했는데 일부 데이터의 entity masking이 깨진것을 확인했다. 깨진 데이터를 최대한 제거하고 train.csv와 concat 했다.
    
    <img width="466" alt="image" src="https://user-images.githubusercontent.com/81913386/165032617-e93af490-169b-47da-af8b-d582f46b3021.png">
    <img width="466" alt="image" src="https://user-images.githubusercontent.com/81913386/165032648-363ce3c6-1b3e-4f0d-bbc3-ec305445cf48.png">


    
    - 총 2개의 데이터셋을 만들었는데, 하나는 기존에 있는 train.csv 전체 데이터를 이용하여 영어와 일본어로 back-translation한 데이터를 추가하여 총 기존 train.csv 대비 3배의 데이터셋을 만들어내었다.
    - 나머지 하나는 데이터의 불균형이 크다는 점을 착안해 train.csv의 각 라벨의 평균 값보다 적은 개수의 라벨에 대한 데이터를 중심으로 데이터를 증강시켰다. 이경우 1.5배의 데이터셋을 만들어내었다.
    - 결론적으로 data augmentation을 이용한 모델의 성능 효과는 미미하거나 적었다. 3배로 늘린 데이터셋은 train/eval에서는 성능이 좋았지만 test에서는 성능이 많이 떨어졌다. overfitting이 많이 발생한것으로 보인다. 1.5배의 데이터셋으로 증강한 것에 대해서는 성능이 일부 증가한것으로 보이나 모델 측면에서 큰 발전을 이루지 못했다.
- EDA/AEDA(Easy Data Augmentation/An Easier Data Augmentation)
    - `koeda`([https://github.com/toriving/KoEDA](https://github.com/toriving/KoEDA))라이브러리를 이용하여 data augmentation을 하려고 시도했으나 entity를 masking하는 코드를 중간에 구현하는데 문제가 있어서 이번 대회에는 사용하지 못했다.
    - 순수 데이터 증강에는 매우 좋을듯 보여서 추후 대회에 사용하면 좋은 data augmentation 방법인것 같다.

### 모델링과 실험 초기 단계

- Stratify K fold와 Soft Voting
    - 중복된 문장이지만 tagging된 단어가 달라 label이 다른 경우를 고려해 중복된 문장은 train, valid 중 한 곳으로 가도록 설정.
    - 새로운 방식의 stratify k fold 방식과 동시에 각 fold마다 k개의 best model의 bin 파일이 저장될 수 있도록 환경 설정
    - 각 fold의 best model을 기반으로 soft voting할 수 있도록 환경 구성 → **f1 score  0.1↑, auprc 2↑**
- Pytorch를 통한 GPU 효율화
    - 이전 대회에서 large-scale의 pretrained model을 돌리는 데에 어려움이 있었고, 이번 대회에서도 비슷한 문제가 발생해 AMP(Auto Mixed Precision)을 적용
    - 정확도 손실 없이 memory에 batch size를 2배로 설정 가능
- CrossEntropyLoss, Label Smoothing, Focal, LDAM loss 등 구현 및 적용
    - Trainer 오버라이딩을 통해 손실함 수 적용
    - 데이터셋의 imbalance, noisy 함을 고려하여 generalization 성능을 높일 수 있는 손실함수 중 베이스라인 CE 대비 성능향상있는 label smoothing 선택(기본적인 single roberta-large 대비 **f1 score 2.09↑, auprc 3.05↑)**

### 실험 중반 시기

- 가장 좋은 성능을 보이는 모델 선택(RoBerta)
    - BERT-base, Koelectra, RoBERTa - base, large 등 모델들을 직접 실험해보며 성능 추이 비교
    - 가장 높은 성능을 보이는 RoBERTa - large 모델을 선택해 실험 진행
- RoBerta를 상속받아 LSTM, Dense Layer 등
- 가장 좋은 성능을 보이는 Label Smoothing Loss 선택과 Smoothing 하이퍼파라미터 튜닝
    - [0.1,0.2,0.3] 중 과한 smoothing 은 성능 악화 → 0.2로 선정
- RoBerta와 가장 성능을 잘 내는 loss 선정(adafactor) → **f1score 1 ↑, auprc 2.0 ↑**
    - 초반의 정확도가 급격하게 튀는 현상을 확인하고 overfitting을 의심, loss를 건드려보자는 아이디어
    - 다양한 loss를 실험해봤지만 AdamW의 성능이 가장 높았고, 대부분의 실험에서도 AdamW를 많이 사용하는 추세
    - 추가적인 정보를 찾아보던 중 Adam과 유사하며 memory-efficient하게 동작하는 loss를 발견해 적용 : Adafactor: Adaptive Learning Rates with Sublinear Memory Cost

### 실험 후반 시기

- Layer Add
    - 네번째로 RoBerta의 모델을 직접 상속받아 만든 새로운 모델들의 fc layer에 layer를 보다 깊게 쌓고 hidden vector size를 조절해보며 실험을 진행  → **f1 score 1~1.5 ↑**
- Entity Tagging 다양화
    - 논문(An Improved Baseline for Sentence-level Relation Extraction)에서 성능 향상의 효과를 보았던 4가지 Entity Tagging 방식 후보군을 모두 실험(아래 사진에서 Entity Mask를 제외한 4가지)
        
        <img width="432" alt="image" src="https://user-images.githubusercontent.com/81913386/165032718-98e9b8c4-f4de-4257-8364-46ef86c007f8.png">
        
    - 논문에서는 4번째 방식이 성능이 가장 좋았지만, 데이터가 다르기 때문에 성능이 다르게 적용할 수도 있다고 생각하여 4가지 방식 모두 성능 추이를 확인
    - 이전 실험들 중에서 좋은 성능을 보였던 Stratified kfold, BiLSTM layer를 사용한 환경으로 환경을 통일하고, Entity Tagging 방식만 변경하여 성능 비교 진행
    - 동일 조건에서 1번 방식([E1][/E1])은 **f1 score 기준 74.3408**, 2번 방식(##@@)은 75.1500, 3번 방식(<S:PERSON></S:PERSON>)은 **74.8734**, 4번 방식(@*person* Bill @)은 **75.5666**의 성능을 보임
        - 참고로 1, 2, 3번은 Batch size를 64로 진행했지만, 4번 방식의 경우 OOM이 발생하여 batch size를 60으로 진행
        - 2번과 4번 방식이 좋은 성능을 보인 이유는 새롭게 토큰을 추가하지 않고 이미 사전에 있는 토큰을 사용하였기 때문에 학습이 빠르게 되었기 때문이라고 예측


## 자체 평가 의견

### 최종성과

Public 3위, Private 5위라는 좋은 성과를 달성했다. 성능이 한 번에 오르는 것이 아니라 각 기능들을 추가하고 실험을 진행해나가며 단계적으로 상승했다는 점이 뜻깊다.

### 추가적으로 알게 된 사항

1) 모델들의 성능이 나오지 않는다고해서 그 실험을 완전히 폐기하지말고 추후 Ensemble에 추가시켜주는 방식이 좋을 수 있다는 것을 알게 되었다.

2) 실험과 그 결과에만 연연하지 말고 “이유"에 대한 탐색을 보다 깊게 하는 방식이 좋을 것 같다.

3) 마스터 클래스 시간에 알게 된 새로운 방법들로, 다음 대회 때 시도해봐도 좋을 것 같다. 

- 마스터님이 추천해주신 Data augmentation : paraphrasing 방식
- 마스터님이 추천해주신 Tokenizer : monolog tokenizer v3

### 다음 대회에 적용해보면 좋을 것

1) hyperparametere tuner(ex. ray-tune)와 같은 것을 빠르게 적용하지 못했던 점

2) 아이디어는 많았으나 담당자 분담이 잘 되지 않았던 것 같음. 아이디어가 깊게 들어가지 못함 → 가설탭을 좀더 구체화하고 노션에 잘 적용해보기

3) AutoML로 모델 architecture를 자동적으로 만들어보았다면?

4) 협업 툴의 편리함을 최대한 활용해보기

5) 실험을 단순히 기록하는 것보다 체계적으로 정리하기(그래프, 대시보드 등 다양한 시각화 방식 시도해보기)

6) 실험 균형을 맞추기 위해 빈 GPU 공유하기
