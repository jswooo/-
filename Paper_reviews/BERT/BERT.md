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


## 2.1 Pre-training BERT



