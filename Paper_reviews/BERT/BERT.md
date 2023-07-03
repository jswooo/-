# 논문 정리


## Abstract 
이 논문을 통해서 "BERT(Bidirectional Encoder Representations from Transformer)"를 제안한다. <br>
기존 GPT1의 경우, 트랜스포머의 디코더를 활용했기에 단방향 학습만이 가능했다. 그러나 BERT는 문맥의 양방향에 대한 학습을 가능하게 하며, 이러한 특징은 question answering이나 language inference에서
강점을 갖게 해준다.<br>
또한 모델 상단에 하나의 layer만을 추가하여 fine-tuning이 가능하며, 논문이 나온 당시에 여러 task에 대해서 SOTA를 차지했다. 
<br>
<br>

## 1. Introduction
NPL task들에 대하여 사전 학습 모델이 매우 효과적이며, 사전 학습 모델은 feature-based(e.g. ELMO)와 fine-tuning(e.g. GPT)로 나눌 수 있다. <br>
두 방법의 접근 방식은 다르나, 사전 학습 중에 단방향 언어 모델을 사용한다는 것은 동일했다. 이러한 특징은 문장의 양방향 문맥을 고려하지 못한다는 점에서, 'Question Answering'과 같은 양방향 문맥이 필요한 task에 대해 어려움을 겪었다. <br><br>
따라서 해당 논문에서는 사전 학습과정에서 마스킹된 일부 토큰을 예측하는 MLM(Masked Language Model)을 목적으로 사용하여 이러한 문제를 해결하고, deep bidirectional한 모델인 "BERT"를 제시한다.(단방향 LM 2개를 concat한 모델은 deep bidirectional이 아님) 또한 MLM과 동시에 "Next Sentence Prediction"도 목적 task로 사용하여, 문장 간의 관계도 학습한다. 
<br>
<br>

## 2. BERT 
<img width="684" alt="image" src="https://user-images.githubusercontent.com/102455634/233039340-b95b7b4f-acc6-48cc-9eba-40f39b8d535e.png">
다른 사전 학습 모델들과 동일하게 BERT도 pre-training과 fine-tuning 과정을 거친다. 앞서 introduction에서 말한 MLM과 NSP task에 대해서 unlabeled data로 pre-training을 진행하고, 이어서 각각 downstream task에 맞는 labeled data로 parameter들에 대해 fine-tuning을 진행한다.
<br><br>
BERT의 특징 중 하나는 동일한 모델 구조로 여러가지 downstream task들을 다룰 수 있다는 것이며, pre-training과 fine-tuning 과정을 간단하게 나타내면 위의 그림과 같다. 
<br>


#### [Model Architecture]
BERT는 'Attention is All You Need'논문에서 제시된 트랜스포머의 인코더 부분만을 여러 겹 쌓은 구조를 띠고 있다. <br>
BERT는 기존 GPT1과 동일한 size인 base model과 파라미터가 더 많은 large model이 존재하며, 각각의 정보는 다음과 같다. 
- BERT base model: (Encoder layer = 12, hidden layer = 768, Self-attention head = 12, total parameters = 110M)
- BERT large model: (Encoder layer = 24, hidden layer = 1024, Self-attention head = 16, total parameters = 340M)


#### [Input, Output 표현] 
<img width="539" alt="image" src="https://user-images.githubusercontent.com/102455634/233045289-e150761d-59ec-4cbe-abae-54686795a06e.png">
BERT의 입력은 문장 하나 또는 두 개로 구성된다. 따라서 이들을 구분하기 위해서 special token인 [CLS]와 [SEP]를 사용한다. 모든 입력의 가장 처음에는 CLS 토큰이 사용되며, 2개의 문장을 입력으로 사용하는 경우에는 문장 사이에 SEP 토큰을 사용하여 이들을 구분한다. <br><br>
주어진 입력 문장과 special token을 포함한 임베딩 값은 token, segment, position embedding의 합으로 구성된다. <br>
우선, 입력에 대해서 wordpiece 토큰화를 진행하여 임베딩 값을 얻는다. 그리고 토큰의 소속 문장을 나타내는 segement embedding과 시퀀스에서의 상대적 위치를 나타내는 position embedding을 더하여 최종 임베딩 값을 산출한다. 이때 각 임베딩 값의 차원은 동일하기에, 임베딩 간에 덧셈을 진행할 수 있다. <br><br>
CLS에 해당하는 토큰의 최종 hidden state 값은 분류 task의 결과값으로 사용된다. 따라서 pre-train 과정 중 NSP에 대한 출력 여부도 해당 토큰의 최종 hidden state 값을 통해서 확인한다. 
<br>
<br>

### 2.1 Pre-training BERT
BERT에서는 기존 left-to-right나 right-to-left 언어 모델을 사용하지 않고, 2개의 비지도 task를 통해 학습을 진행한다. 

#### #1. Masked Language Model(MLM)
BERT는 input 시퀀스의 토큰 중 15%를 마스킹하고, 마스킹된 토큰을 예측하는 MLM을 통해서 깊은 양방향 representation을 학습한다.<br>
단, MLM은 pre-training에서만 실행하고 fine-tuning에 있어서는 실행하지 않기에 mismatch가 발생할 수 있다. 따라서 pre-training 과정에서 마스킹 된 15%의 토큰은 각각 다음과 같이 변경해서 mismatch로 인한 문제점을 줄여준다. 
- [MASK] 토큰으로 대체(80%) 
- 랜덤 토큰으로 대체(10%)
- 바꾸지 않고 그대로 둠(10%) 


위와 같이 실행할 경우, 인코더가 어떤 단어를 예측해야 하는지 알 수 없고 어떤 단어가 랜덤으로 대체되었는지 알 수 없다. 따라서 모든 단어들에 대한 문맥적 representation을 확인해야 하기에, 성능면에서 이점을 얻을 수 있다. 또한, 랜덤으로 대체되는 것은 전체 토큰의 1.5%(마스킹 되는 토큰 15% 중 10%만 랜덤으로 대체)이다. 그렇기에 모델의 학습에 있어서도 크게 문제되지 않는다. 
<br> 
<br>
MLM의 효과는 BERT 논문에서 실행한 Ablation study를 통해서 알아볼 수 있다. <br>
<img width="400" alt="image" src="https://user-images.githubusercontent.com/102455634/233071660-90572448-abd4-4de0-b64c-9996712f5150.png">
<br>
토큰을 마스킹하기 때문에 left-to-right에 비해서 정확도 수렴은 늦지만, 같은 BERT base에 대해서 비교 실험을 진행해도 MLM 사용 모델의 정확도가 훨씬 높은 것을 확인할 수 있다. 
<br>
<br>
<img width="380" alt="image" src="https://user-images.githubusercontent.com/102455634/233074035-aafff2af-ca78-4a61-ba2b-335ec141238e.png">
<br>
앞서 언급한 마스킹 비율 또한 마스킹 비율에 대한 ablation study를 통해 확인 가능하다. 80%, 10%, 10%로 진행시 정확도가 가장 높은 것을 볼 수 있다. 
<br>
<br>

#### #2. Next Sentence Prediction(NSP)
Question Answering(QA)이나 자연어 추론(NLI) 같은 경우에는 문장 간의 이해가 중요하다. 따라서 BERT는 문장 간의 관계를 학습하기 위해서 NSP를 사용한다. <br>
<br>
pre-training에 대해서 2가지의 문장이 input으로 주어진다. 이 중 50%는 실제로 연결되는 문장이며(IsNext), 나머지 50%는 랜덤으로 선택된 문장(NotNext)이 주어진다. 다음과 같은 과정을 통해 문장 사이의 관계를 학습할 수 있으며, 실제로 QA와 NLI task에 대해서 효과적이다. <br>
<br>
<img width="380" alt="image" src="https://user-images.githubusercontent.com/102455634/233079908-5c5e630c-2917-49ee-9947-e642e6577325.png">
<br>
<br>
다음은 pre-training task의 효과를 확인하기 위한 ablation study이며, 결과를 통해 'Masked Language Model'과 'Next Sentence Prediction'이 효과적임을 확인할 수 있다. 
<br>
<br>
#### #3. Pre-training data 
Pre-training을 위해서 BooksCorpus(800M words)와 English Wikipedia(2500M words)를 사용했다. <br>
또한 긴 길이의 연속된 시퀀스를 input으로 사용하기 위해서, document level의 corpus를 활용했다. <br>
<br>
#### #4. Pre-training hyper parameter 조건
- token 길이 = 512 
- batch size = 256 
- Adam with learning rate = 1e-4 (β1 = 0.9, β2 = 0.999)
- L2 weight decay = 0.01 (10,000 step 이후 선형적 decay) 
- Dropout rate = 0.1 
- GeLU activation function 사용 
- 시간 절약을 위해 90%는 128 시퀀스 길이로 학습하고, 이후에 512로 학습 
<br>

### 2.2 Fine-tuning BERT
BERT는 트랜스포머의 self-attention을 활용해서 downstream task에 대한 fine-tuning도 쉽게 진행할 수 있다. <br>
task마다 각각 다른 input을 받아서 진행하는데, input은 다음과 같다.
- sentence pairs in paraphrasing 
- hypothesis premise pairs in entailment
- question passage in QA
- Degenerate-None pair in text classification or sequence tagging
<br>
또한 task에 따라서 output도 달라지게 되는데, 1. sequence tagging이나 QA 같은 token level task에 대하선 token representation 2. 분류 문제에 대해선 [CLS] representation이다.<br>
이러한 fine-tuning 과정은 사전 학습에 비해서 시간이 적게 드는데, TPU를 사용한다면 1시간에 끝나기도 한다.
<br>
<br>

## 3. 실험 결과 
<img width="596" alt="image" src="https://user-images.githubusercontent.com/102455634/234211826-63e66c1c-1c2d-47d1-b1ea-0ea4f9a27435.png">
BERT를 대상으로 11개의 task(GLUE tasks, SQuAD v1.1, SQuAD v2.0, SWAG)에 대해서 실험을 진행했다. <br>
위의 사진은 GLUE task에 대한 실험 결과이며, GLUE를 비롯한 총 11개의 task들에서 기존 방법들에 비해 'BERT'가 높은 성능을 보인다는 것을 확인할 수 있다. <br><br>
<img width="495" alt="image" src="https://user-images.githubusercontent.com/102455634/234215048-b9d711ad-cd8c-4355-9e37-3f7146c4f2b1.png">
BERT는 최종 layer 위에 하나의 layer를 추가하여, 각각의 다른 downstream task를 진행할 수 있다. 
<br>
<br>


## 4. BERT와 기존 model(GPT1, ELMo) 차이 
<img width="700" alt="image" src="https://user-images.githubusercontent.com/102455634/234219210-97167e84-3cca-4b23-a059-6f9a25e2cb01.png">
각 방법에 대해 간단하게 나타내면 다음과 같다. <br>
<br>

- BERT: 일정 비율을 마스킹하여, 양 방향을 동시에 학습 
- GPT1: 디코더만을 사용하여 정방향만을 학습 
- ELMo: biLSTM을 사용하여 정방향과 역방향을 각각 학습한 후, 해당 hidden state를 모두 선형결합

BERT는 양 방향 문맥을 고려할 수 있다는 점에서 GPT에 비해 'Question Answering'과 같은 task에서 강점을 갖는다. 또 이외의 다른 기본적인 NLP task에서도 양방향 문맥을 이해한다는 것이 단방향 문맥만을 이해하는 것보다 유리하다.<br>
<br>
ELMo가 LSTM을 통해 정방향과 역방향을 모두 학습한다는 점에서 양 방향 학습이 가능하다는 의문이 들 수도 있다. 그러나 이는 각각을 따로 학습하고, 그 후에 hidden state만을 단순 concat 한다는 점에서 'deep bidirectional'하지 않다. 
이와 달리 BERT는 masking을 통해 동시에 정방향과 역방향을 학습하므로, deep bidirectional하다. <br>
<br>
MLM 덕분에 완전한 양뱡항 학습이 가능하지만, 기존 left-to-right model에 비해서 수렴속도가 늦다는 단점이 있다. 하지만 이러한 cost보다 MLM을 통해서 얻는 학습 성능 향상이 더 높다. 
<br>
<br>


## 5. 기타 ablation studies 

### 5.1 Effect of Model Size 
<img width="369" alt="image" src="https://user-images.githubusercontent.com/102455634/234228053-5f7a2b8b-f5ef-4e61-9152-fd2ac40f0f5d.png">
train 과정에서 BERT 모델의 layer, hidden units, attention head를 늘릴수록 학습 정확도에 있어 향상을 가져올 수 있다는 ablation study를 진행했다. 기존 모델에 비해 매우 큰 모델에서도 크기를 늘릴수록 성능 향상을 확인할 수 있었다. <br>
이러한 사실은 기존의 논문들을 통해서도 알 수 있는 사실이다. 그러나 해당 BERT 논문은 이러한 점 이외에, pre train에 있어 모델의 사이즈가 크다면 작은 스케일의 downstream task에 있어 강점을 보인다는 점을 확인했다. 데이터가 매우 작은 downstream task에 대해서, 모델의 크기가 커질수록 강점을 갖는다. 
<br>
<br>

### 5.2 Feature-based approach with BERT 
<img width="330" alt="image" src="https://user-images.githubusercontent.com/102455634/234235257-ac216395-18bf-4121-956b-09ac5011d271.png">
BERT는 fine-tuning 만큼은 아니지만 feature-based에서도 높은 성능을 보인다. <br>
Feature-based 방법 중에서도 마지막 최종 4개의 hidden state를 concat하여 사용하는 것이 가장 성능이 높았다. 


