# 논문 정리


## 1. Abstract 
순환 신경망 또는 컨볼루션 신경망 구조의 모델이 아닌, 오직 어텐션 메커니즘을 적용한 'Transformer' 구조를 제안한다.<br>
Transformer는 기존 모델들에 비해서 병렬화를 더 잘 적용할 수 있으며, 학습에 있어 적은 시간이 소요된다. 따라서 기계 번역에서 해당 시기에 SOTA를 달성했으며, 다른 task에 대해서도 일반적으로 잘 작동한다. 
<br>
<br>

## 2. Introduction
기존에는 sequence modeling이나 transduction 문제에 있어 RNN, LSTM, gated RNN 등이 SOTA를 차지해왔다. <br>
그러나 순환 모델들의 순차적 특성으로 인해 병렬화에 어려움이 존재하며, 메모리 제약에 의해 긴 길이의 시퀀스들에 처리에 문제가 있다. 이러한 문제들은 계산 효율성과 모델 성능을 향상시키더라도 존재한다.<br>
어텐션 메커니즘은 input과 output 시퀀스의 거리에 상관없이 각 단어의 관계를 종속성을 파악할 수 있으며, 대부분의 경우에서 순환 구조와 함께 사용되어왔다. <br>
따라서 해당 논문에서는 reccurence를 제외하고, 어텐션 메커니즘만 사용하는 'Transformer'를 제안한다. 
<br>
<br>

## 3. Background 
'Extended Neural GPU, ByteNet, ConvS2S와 같은 모델들은 순차 연산을 줄이기 위해 만들어졌으나, 입력 또는 출력 간의 종속성을 계산하기 위한 연산이 거리에 따라 증가된다는 문제가 발생한다. 그리고
이러한 문제는 종속성 학습을 매우 어렵게 만든다. <br>
이와 달리 Transformer는 연산 수를 상수로 줄여버리며, 이 과정에서 생기는 effective resolution 감소와 같은 문제는 Multi-Head Attention을 통해서 해결한다. <br>
해당 논문에서 제시하는 Transformer 구조는 RNN이나 합성곱을 사용하지 않고 오로지 self-attention 만을 이용해 input과 output representation을 계산하는 최초의 변환 모델이다. 
<br>
<br>

## 4. Model Architecture
<img width="400" alt="image" src="https://user-images.githubusercontent.com/102455634/227722170-e46583a6-4e76-415c-8d54-6e77f0fed4bd.png">
Transformer는 위와 같이, self-attention과 point-wise fully connected layer로 구성된 인코더와 디코더의 stack으로 구성된다. <br>
주어진 입력 시퀀스는 인코더, 디코더 순으로 거치며, 각 과정은 auto-regressive하게 진행된다. <br>
모델 구조와 요소들에 대해 자세히 들여보면 다음과 같다. <br>
<br>

### 4.1 Encoder and Decoder stack
#### Encoder <br>
논문에서는 총 6개의 인코더 stack을 이용했다. 각각의 인코더는 multi-head self-attention과 position-wise fully connected feed-forward network라는 하부 레이어로 구성된다. 
또 이러한 하부 레이어 사이에는 residual 연결과 normalization을 적용한다. 이러한 연결들을 위해서 출력 및 임베딩 차원은 모두 512 지정했다. <br>
#### Decoder <br>
디코더도 인코더와 마찬가지로 6개의 디코더 stack을 이용했다. 인코더에서 사용한 2개의 하부 레이어를 동일하게 사용하나, 추가로 인코더의 출력에 대해 multi-head attention을 적용하는 레이어를 
사용한다. 또한 디코더 self-attention layer에는 masking을 사용해서, 현재 시퀀스 출력에 대해 이후 스텝들이 참여하는 것을 방지한다. (현재 위치의 출력에 대해서는 이전 출력들의 영향만 받아야 하기 때문에..) <br>
이외에 residual이나 normalization은 인코더와 동일하게 모든 하부 레이어들에 대해 적용된다. <br>
<br>

### 4.2 Attention <br>
<br>
어텐션은 기존 어텐션 방식과 유사하게 query, key, value를 사용하여 가중합 연산을 진행한다. <br>
<br>
<img width="600" alt="image" src="https://user-images.githubusercontent.com/102455634/227724804-3b0fd449-6506-42a2-9a80-e0fecfaf8fca.png">



#### Scaled Dot-Product Attention <br>
<img width="300" alt="image" src="https://user-images.githubusercontent.com/102455634/227725880-a7ac7af6-57a8-4c78-b245-dcaeac19ce79.png">
query, key, value에 대한 행렬을 각각 기호 Q, K, V로 나타낼 때, 어텐션을 위와 같은 수식을 통해 얻을 수 있다. <br>
dot-product attention에서는 query와 key의 차원이 커질수록 softmax 값이 작아져, 역전파 시 gradient가 매우 작아 학습이 잘 안되는 문제가 발생한다. 따라서 이러한 문제를 방지하고자 
해당 논문에서는 softmax 이전에 차원의 제곱근으로 나눠준다. <br>
<br>

#### Multi-Head Attention
<img width="400" alt="image" src="https://user-images.githubusercontent.com/102455634/227726466-8b1e3ec8-652f-41d9-86d2-a76e587b7233.png">
어텐션을 여러 개 사용하여 다양한 어텐션 표현을 얻을 경우, 기존 하나의 어텐션 만을 사용할 때보다 성능이 향상된다. 따라서 해당 논문에서는 8개의 어텐션 head를 활용하여 진행했다. 또한 각 어텐션의 차원은
concat하여 가중치 행렬을 내적한 후, 하나의 input으로서 feed forward layer를 통과한다.<br> 
<br>

#### Applications of attention in this paper 
이번 논문에서는 총 3가지 형태의 어텐션이 활용된다.
> Encoder-decoder attention <br>
> <br>
> 디코더에서 사용되는 어텐션으로서, query만 디코더에서 가져오고 key와 value는 모두 인코더의 출력에서 가져온다. 이를 통해서 디코더의 출력이 encoder input의 어느 부분으로부터 영향을 많이 받는지 파악할 수 있다. 


> Self-attention layers in encoder <br>
> <br>
> query, key, value를 모두 encoder 내부에서 가져오는 가장 일반적인 형태의 어텐션이다. 각 단어가 인코더의 출력의 어느 부분에서 영향을 받는지 확인할 수 있다. 


> Self-attention layers in decoder <br>
> <br>
> 기본적인 형태는 인코더의 self-attention과 유사하나, 현재의 출력이 과거 출력에만 영향을 받을 수 있도록 설정했다. masking을 통해서 이를 구현한다. 
<br>



### 4.3 Position-wise Feed-Forward Networks 
인코더와 디코더 내부 fc layer에서는 활성화 함수로 ReLU를 사용한다. <br>
<br>



### 4.4 Positional Encoding 
트랜스포머는 recurrence나 convolution을 사용하지 않기 때문에, 각 토큰의 순서 정보를 제공해줄 필요가 있다. 따라서 각 토큰의 임베딩 과정시, positional encoding 값을 더하여 임베딩 값을 구성한다. 기존 임베딩에 더해줘야 하기에, positional encoding 값은 입력 차원과 동일하게 512로 구성된다. 
<img width="300" alt="image" src="https://user-images.githubusercontent.com/102455634/227761878-784609f3-a370-4994-aa62-e1eb2123a380.png">
Positional encoding은 sin과 cos 함수를 통해서 구성했으며, 기존에 사용하는 positional embedding에 비해 성능 차이가 발생하지 않는다. 
<br>
<br>

### 4.5 Self-attetion 사용 이유 
기존 recurrence 또는 convolution layer를 사용하는 것에 비해, self-attention을 사용하는 것은 다음과 같은 3가지 이점을 갖는다. <br>

<img width="716" alt="image" src="https://user-images.githubusercontent.com/102455634/227762413-11f85300-a6bd-490c-ad66-baaebf043267.png">

> Total computational complexity <br>
> <br>
> self-attention은 layer의 계산 복잡도로 $O(n^2 * d)$를 갖는다. 보통 sentence representation에서는 representation 차원인 d가 sequence 길이인 n보다 크다. 따라서 일반적인 경우,
> self-attention의 계산 복잡도가 recurrent나 convolutional 구조에 비해서 낮다.  


> Parallelize 할 수 있는 연산량 <br>
> <br>
> 병렬화 할 수 있는 연산량은 순차적 연산이 적을수록 높아진다. 따라서 위의 표를 확인하면, self-attention이 기존 recurrent 구조에 비해 우위를 가지는 것을 확인할 수 있다. 


> Long range dependency의 길이 <br>
> <br>
> 각 토큰 간의 dependency를 학습하는 것은 네트워크 내의 순방향 및 역방향이 통과해야하는 거리에 의존한다. 거리가 짧을수록 long-range dependency 학습이 용이해진다. self-attention의 
> maximum path length의 복잡도는 O(1)로 가장 작고, 이에 따라 기존 방법들에 비해 이점을 갖는다. 
<br>


## 5. Training 
학습 데이터와 방식은 다음과 같이 구성해서 진행했다. 
- 학습 데이터 셋 : 영어-독일어(WMT 2014 English-German dataset), 영어-프랑스어(WMT 2014 English-French dataset)
- 하드웨어 : NVIDIA P100 GPU 8개 
- 학습시간 : 12시간(base model), 3.5일(big model)
- optimizer : Adam optimizer를 사용하며, learning rate를 변경 
- regularization : residual dropout, label smoothing 
<br>
<br>

## 6. Result

### 6.1 각 task의 결과 
<img width="641" alt="image" src="https://user-images.githubusercontent.com/102455634/227769292-692b049f-6bba-497b-82c9-f90acf9d5a06.png">
<br>
앞서 학습한 machine translation task 경우, big transformer model을 통해서 SOTA를 달성할 수 있었다. 그리고 영어-독일어 기계 번역에 대해서는 base transformer model 만으로도 기존
모델에 비해서 높은 성능을 확보할 수 있었다. 또한, 영어-프랑스에 대해서는 기존에 필요한 학습 리소스에 비해 1/4를 가지고 학습을 진행할 수 있었다. (여기에서 학습 cost는 'training 시간 * 사용 GPU 수 * GPU 성능'을 통해 산출) <br>
학습에 있어 base model은 가중치로 10분 간격의 마지막 5개 체크포인트 값을 평균화하여 사용했으며, big model은 마지막 20개 체크포인트 값을 평균화하여 사용했다. 그리고 beam size를 4로 지정하여 bim search를 진행했다. 
<br>
<br>
<img width="603" alt="image" src="https://user-images.githubusercontent.com/102455634/227769640-1fd6adf4-b25c-4b2f-aba2-607dbf735c48.png">
<br>
이외에도 입력에 비해 길고, 문법적으로 구조화된 출력을 요구하는 'english constituency parsing'과 같은 task에 대해서도 트랜스포머는 높은 성능을 보인다. 
기존 RNN 기반의 sequence to sequence model은 데이터가 적은 상황에서 좋은 성능을 보이기 힘들었으나, 트랜스포머는 task-specific tuning이 부족한 상황에서도 더 나은 결과를 나타냈다.
<br>
<br>

### 6.2 모델의 변수 중요도 
<img width="735" alt="image" src="https://user-images.githubusercontent.com/102455634/227770446-13b31fc3-45f2-4991-a39d-cdfc06465425.png">
<br>
각 변수의 중요도를 알아보기 위해서 base model에서 변수를 변화시키며 실험을 진행했을때, 다음과 같은 정보를 알 수 있다. 
- Attention head를 늘릴 경우에 성능이 향상되나, 너무 많이 늘리면 다시 성능 하락을 보인다. ((A)를 통해서 확인)
- Attention size key를 줄이면 모델 성능이 떨어진다. ((B)를 통해서 확인) 
- 모델의 크기가 커질수록 성능이 좋아진다. ((C)를 통해서 확인)
- Dropout은 과적합 방지에 효과적이다. ((D)를 통해서 확인) 
- 논문에서 제시한 positional encoding 방식과 기존 학습된 positional embedding 사이에 성능 차이는 거의 없다. ((E)와 base model 비교) 
<br>
<br>


## 7. Conclusion 
논문에서 제시한 Transformer는 시퀀스 변환 모델에 있어 recurrent layer를 사용하지 않고, attention만을 사용해서 구현한 최초의 모델이다. <br> 
번역 task에 있어, Transformer는 기존 recurrent나 convolution layer를 사용한 모델들보다 뛰어난 성능을 보인다. 
<br>
<br>

## 몰랐던 용어 정리.. 
- [Byte-pair encoding](https://wikidocs.net/22592): 기존에 있는 단어들을 통해서 모르든 단어를 해결하는 알고리즘. 
- [Label smoothing](https://blog.si-analytics.ai/21#footnote_21_2): 결과 값에 대해서 0에서 1 사이의 확률 값으로 soft label하는 방법이다. 정답이 아닌 선지들에 대해 일정량의 확률을 분배하고, 최대 확률이 다른 확률들에 비해서 지나치게 커지는 것을 방지한다. 또한 딥러닝 모델이 정답에 대해서 과하게 확신하는 현상을 줄여서 신뢰성을 올린다. 
- [Beam search](https://littlefoxdiary.tistory.com/4): 각각의 step에서 확률이 높은 k개의 토큰들을 유지하며 시퀀스 번역을 진행하는 방법이다. k를 높게 잡을 경우에는 속도에 문제가 생긴다. 
- [additive attention](https://wikidocs.net/73161): bahdanau attention이라고도 하며, hidden state를 통해 얻은 context vector와 input embedding을 concat하여 input으로 사용한다. 
