## 논문 정리


### 1. Abstract 
순환 신경망 또는 컨볼루션 신경망 구조의 모델이 아닌, 오직 어텐션 메커니즘을 적용한 'Transformer' 구조를 제안한다.<br>
Transformer는 기존 모델들에 비해서 병렬화를 더 잘 적용할 수 있으며, 학습에 있어 적은 시간이 소요된다. 따라서 기계 번역에서 해당 시기에 SOTA를 달성했으며, 다른 task에 대해서도 일반적으로 잘 작동한다. 
<br>
<br>

### 2. Introduction
기존에는 sequence modeling이나 transduction 문제에 있어 RNN, LSTM, gated RNN 등이 SOTA를 차지해왔다. <br>
그러나 순환 모델들의 순차적 특성으로 인해 병렬화에 어려움이 존재하며, 메모리 제약에 의해 긴 길이의 시퀀스들에 처리에 문제가 있다. 이러한 문제들은 계산 효율성과 모델 성능을 향상시키더라도 존재한다.<br>
어텐션 메커니즘은 input과 output 시퀀스의 거리에 상관없이 각 단어의 관계를 종속성을 파악할 수 있으며, 대부분의 경우에서 순환 구조와 함께 사용되어왔다. <br>
따라서 해당 논문에서는 reccurence를 제외하고, 어텐션 메커니즘만 사용하는 'Transformer'를 제안한다. 
<br>
<br>

### 3. Background 
'Extended Neural GPU, ByteNet, ConvS2S와 같은 모델들은 순차 연산을 줄이기 위해 만들어졌으나, 입력 또는 출력 간의 종속성을 계산하기 위한 연산이 거리에 따라 증가된다는 문제가 발생한다. 그리고
이러한 문제는 종속성 학습을 매우 어렵게 만든다. <br>
이와 달리 Transformer는 연산 수를 상수로 줄여버리며, 이 과정에서 생기는 effective resolution 감소와 같은 문제는 Multi-Head Attention을 통해서 해결한다. <br>
해당 논문에서 제시하는 Transformer 구조는 RNN이나 합성곱을 사용하지 않고 오로지 self-attention 만을 이용해 input과 output representation을 계산하는 최초의 변환 모델이다. 
<br>
<br>

### 4. Model Architecture
<img width="400" alt="image" src="https://user-images.githubusercontent.com/102455634/227722170-e46583a6-4e76-415c-8d54-6e77f0fed4bd.png">
Transformer는 위와 같이, self-attention과 point-wise fully connected layer로 구성된 인코더와 디코더의 stack으로 구성된다. <br>
주어진 입력 시퀀스는 인코더, 디코더 순으로 거치며, 각 과정은 auto-regressive하게 진행된다. <br>
모델 구조에 대해 자세히 들여보면 다음과 같다. <br>
<br>

#### 4.1 Encoder and Decoder stack
- Encoder <br>
논문에서는 총 6개의 인코더 stack을 이용했다. 각각의 인코더는 multi-head self-attention과 position-wise fully connected feed-forward network라는 하부 레이어로 구성된다. 
또 이러한 하부 레이어 사이에는 residual 연결과 normalization을 적용한다. 이러한 연결들을 위해서 출력 및 임베딩 차원은 모두 512 지정했다. 
- Decoder <br>
디코더도 인코더와 마찬가지로 6개의 디코더 stack을 이용했다. 인코더에서 사용한 2개의 하부 레이어를 동일하게 사용하나, 추가로 인코더의 출력에 대해 multi-head attention을 적용하는 레이어를 
사용한다. 또한 디코더 self-attention layer에는 masking을 사용해서, 현재 시퀀스 출력에 대해 이후 스텝들이 참여하는 것을 방지한다. (현재 위치의 출력에 대해서는 이전 출력들의 영향만 받아야 하기 때문에..) <br>
이외에 residual이나 normalization은 인코더와 동일하게 모든 하부 레이어들에 대해 적용된다. <br>
<br>
#### 4.2 Attention <br>
<br>
어텐션은 기존 어텐션 방식과 유사하게 query, key, value를 사용하여 가중합 연산을 진행한다. <br>
<br>
<img width="600" alt="image" src="https://user-images.githubusercontent.com/102455634/227724804-3b0fd449-6506-42a2-9a80-e0fecfaf8fca.png">
<br>
- Scaled Dot-Product Attention <br>
ddd
