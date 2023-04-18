# 논문 정리


## 1. Abstract 
이 논문을 통해서 "BERT(Bidirectional Encoder Representations from Transformer)"를 제안한다. <br>
기존 GPT1의 경우, 트랜스포머의 디코더를 활용했기에 단방향 학습만이 가능했다. 그러나 BERT는 문맥의 양방향에 대한 학습을 가능하게 하며, 이러한 특징은 question answering이나 language inference에서
강점을 갖게 해준다.<br>
또한 모델 상단에 하나의 layer만을 추가하여 fine-tuning이 가능하며, 논문이 나온 당시에 여러 task에 대해서 SOTA를 차지했었다. 
<br>
<br>

## 2. Introduction
