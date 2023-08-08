---
title: "🤖 [Transformer] Attention Is All You Need"
excerpt: "Transformer Official Paper Review with Pytorch Implementation"
permalink: "/nlp/transformer"
toc: true  # option for table of content
toc_sticky: true  # option for table of content
categories:
  - NLP
tags:
  - Natural Language Process
  - Transformer
  - Self-Attention
  - Seq2Seq
  - Encoder
  - Decoder
  
last_modified_at: 2023-08-03T11:00:00-05:00
---

### `🔭 Overview`

`Transformer`는 2017년 Google이 NIPS에서 발표한 자연어 처리용 신경망으로 기존 `RNN` 계열(LSTM, GRU) 신경망이 가진 문제를 해결하고 최대한 인간의 자연어 이해 방식을 수학적으로 모델링 하려는 의도로 설계 되었다. 이 모델은 초기 `Encoder-Decoder` 를 모두 갖춘 `seq2seq` 형태로 고안 되었으며, 다양한 번역 테스크에서 `SOTA`를 달성해 주목을 받았다. 이후에는 여러분도 잘 아시는 것처럼  `BERT`, `GPT`, `ViT`의 베이스 라인으로 채택 되며, 현대 딥러닝 역사에 한 획을 그은 모델로 평가 받고 있다.

현대 딥러닝의 전성기를 열어준 `Transformer`는 어떤 아이디어로 기존 `Recurrent` 계열이 가졌던 문제들을 해결했을까?? 이것을 제대로 이해하려면 먼저 기존 순환 신경망 모델들이 가졌던 문제부터 짚고 넘어갈 필요가 있다.

### **`🤔 Limitation of Recurrent Structure`**

- **1) 인간과 다른 메커니즘의 Vanishing Gradient 발생 (Activation Function with Backward)**
- **2) 점점 흐려지는 Inputs에 Attention (Activation Function with Forward)**
- **3) 디코더가 가장 마지막 단어만 열심히 보고 `denoising` 수행 (Seq2Seq with Bi-Directional RNN)**

#### `📈 1) 인간과 다른 메커니즘의 Vanishing Gradient 발생 (Activation Function with Backward)`

$$
h(t) = tanh(x_tW_x + h_{t-1}W_h + b))
$$

`RNN`의 활성 함수인 `Hyperbolic Tangent` 는 $y$값이 `[-1, 1]` 사이에서 정의되며 기울기의 최대값은 1이다. 따라서 이전 시점 정보는 시점이 지나면 지날수록 (더 많은 셀을 통과할수록) 그라디언트 값이 작아져 미래 시점의 학습에 매우 작은 영향력을 갖게 된다. 이것이 바로 그 유명한 `RNN`의 `Vanishing Gradient` 현상이다. 사실 현상의 발생 자체는 그렇게 큰 문제가 되지 않는다. `RNN`에서 발생하는 `Vanishing Gradient` 가 문제가 되는 이유는 바로 인간이 자연어를 이해하는 메커니즘과 다른 방식으로 현상이 발생하기 때문이다. 우리가 글을 읽는 과정을 잘 떠올려 보자. 어떤 단어의 의미를 알기 위해 가까운 주변 단어의 문맥을 활용할 때도 있지만, 저 멀리 떨어진 문단의 문맥을 활용할 때도 있다. 이처럼 단어 혹은 시퀀스를 구성하는 `원소 사이의 관계성`이나 `어떤 다른 의미론적인 이유`로 `불균형`하게 현재 시점의 학습에 영향력을 갖게 되는게 아니라, 단순 `입력 시점` 때문에 불균형이 발생하기 때문에 `RNN`의 `Vanishing Gradient`가 낮은 성능의 원인으로 지목되는 것이다. 

다시 말해, 실제 자연어의 문맥을 파악해 그라디언트에 반영하는게 아니라 단순히 시점에 따라서 그 영향력을 반영하게 된다는 것이다. 멀리 떨어진 시퀀스의 문맥이 필요한 경우를 `Recurrent` 구조는 정확히 학습할 수 없다. 

그렇다면 활성 함수를 `relu` 혹은 `gelu` 를 사용하면 위 문제를 해결할 수 있을까? `Vanishing Graident` 문제는 해결할 수도 있으나 `hidden_state` 값이 발산할 것이다. 그 이유는 두 활성 함수 모두 양수 구간에서 선형인데, 이전 정보를 누적해서 가중치와 곱하고 현재 입력값에 더하는 `RNN`의 구조를 생각해보면 넘어오는 이전 정보는 누적되면서 점점 커질 것이고 그러다 결국 발산하게 된다.  

결론적으로 `Vanishing Gradient` 현상 자체가 문제는 아니지만 모델이 자연어의 문맥을 파악해 그라디언트에 반영하는게 아니라 단순히 시점에 따라서 불균형하게 발생하기 때문에 낮은 성능의 원인으로 지목 받는 것이다. 이것을 `long-term dependency`라고 부르기도 한다.

#### `✏️ 2) 점점 흐려지는 Inputs에 Attention (Activation Function with Forward)`

<p markdown="1" align="center">
![tanh function](/assets/images/transformer/tanh.png){: .align-center}{: width="75%", height="50%"}{: .image-caption}
__*tanh function*__
</p>


`Hyperbolic Tangent` 은  $y$값이 `[-1, 1]` 사이에서 정의된다고 했다. 다시 말해 셀의 출력값이 항상 일정 범위값( `[-1,1]` )으로 제한(가중치, 편향 더하는 것은 일단 제외) 된다는 것이다. 따라서 한정된 좁은 범위에 출력값들이 맵핑되는데, 이는 결국 입력값의 정보는 대부분 소실된 채 일부 특징만 정제 되어 출력되고 다음 레이어로 `forward` 됨을 의미한다. 그래프를 한 번 살펴보자. 특히 `Inputs` 값이 2.5 이상인 경우부터는 출력값이 거의 1에 수렴해 그 차이를 직관적으로 파악하기 힘들다. 이러한 활성함수가 수십개, 수백개 쌓인다면 결국 원본 정보는 매우 흐려지고 뭉개져서 다른 인스턴스와 구별이 힘들어 질 것이다.

#### `🔬 3) 디코더가 가장 마지막 단어만 열심히 보고 denoising 수행 (Seq2Seq with Bi-Directional RNN)`

`“쓰다”` ($t_7$)라는 단어의 뜻을 이해하려면 `“돈을”`, `“모자를”`, `“맛이”`, `“글을”`($t_1$)과 같이 멀리 있는 앞 단어를 봐야 알 수 있는데, $h_7$ 에는 $t_1$이 흐려진 채로 들어가 있어서 $t_7$의 제대로 된 의미를 포착하지 못한다. 심지어 언어가 영어라면 뒤를 봐야 정확한 문맥을 알 수 있는데 `Vanilla RNN`은 단방향으로만 학습을 하게 되어 문장의 뒷부분 문맥은 반영조차(뒤에 위치한 목적어에 따라서 쓰다라는 단어의 뉘앙스는 달라짐) 할 수 없다. 그래서 `Bi-directional RNN` 써야하는데, 이것도 역시도 여전히 `“거리”`에 영향 받는다는 건 변하지 않기 때문에 근본적인 해결책이라 볼 수 없다. 즉 인코덜

한편, 디코더의 `Next Token Prediction` 성능은 무조건 인코더로부터 받는 `Context Vector`의 품질에 따라 좌지우지 된다. 그러나 Recurrent 구조의 인코더로부터 나온 Context Vector는 앞서 서술한 것처럼 좋은 품질(뒤쪽 단어가 상대적으로 선명함)이 아니다. 따라서 디코더의 번역(다음 단어 예측) 성능 역시 좋을리가 없다. 

결국 `Recurrent` 구조 자체에 명확한 한계가 존재하여 인간이 자연어를 사용하고 이해하는 맥락과 다른 방식으로 동작햐게 되었다. `LSTM`, `GRU`의 제안으로 어느 정도 문제를 완화 시켰으나, 앞에서 서술했듯이 태생이 `Recurrent Structure`을 가지기 때문에 근본적인 해결책이 되지는 못했다. 그렇다면 이제 `Transformer`가 어떻게 위에 서술한 3가지 문제를 해결하고 현재의 위상을 갖게 되었는지 알아보자.

### **`🌟 Modeling`**

<p markdown="1" align="center">
![Attention Is All You Need](/assets/images/transformer/transformer_overview.png){: .align-center}{: width="50%", height="50%"}{: .image-caption}
__*[Attention Is All You Need](https://arxiv.org/abs/1706.03762)*__
</p>

앞서 `Recurrent` 구조의 `Vanishing Gradient` 을 설명하면서 시점에 따라 정보를 소실하게 되는 현상은 인간의 자연어 이해 방식이 아니라는 점을 언급한 적 있다. 따라서 `Transformer`는 최대한 인간의 자연어 이해 방식을 수학적으로 모델링 하는 것에 초점을 맞췄다. 우리가 쓰여진 글을 이해하기 위해 하는 행동들을 떠올려 보자. **`“Apple”`<U>이란 단어가 사과를 말하는 것인지, 브랜드 애플을 지칭하는 것인지 파악하기 위해 같은 문장에 속한 주변 단어를 살피기도 하고 그래도 파악하기 힘들다면 앞뒤 문장, 나아가 문서 전체 레벨에서 맥락을 파악하기 위해 노력한다.</U>** `Transformer` 연구진은 바로 이 과정에 주목했으며 이것을 모델링하여 그 유명한 `Self-Attention`을 고안해낸다.

<p markdown="1" align="center">
![Word Embedding Space](/assets/images/transformer/word_embedding.png){: .align-center}{: width="50%", height="50%"}{: .image-caption}
__*[Word Embedding Space](https://www.researchgate.net/figure/Visualization-of-the-word-embedding-space_fig4_343595281/download)*__
</p>

다시 말해 `Self-Attention`은 토큰의 의미를 이해하기 위해 `전체 입력 시퀀스` 중에서 어떤 단어에 주목해야할지를 수학적으로 표현한 것이라 볼 수 있다. **<U>좀 더 구체적으로는 시퀀스에 속한 여러 토큰 벡터(행백터)를 임베딩 공간 어디에 배치할 것인가에 대해 훈련하는 행위다.</U>** 

<p markdown="1" align="center">
![Scaled Dot-Product Attention](/assets/images/transformer/scaled_dot_attention.png){: .align-center}{: width="75%", height="50%"}{: .image-caption}
__*[Scaled Dot-Product Attention](https://arxiv.org/abs/1706.03762)*__
</p>

그렇다면 이제부터 `Transformer` 가 어떤 아이데이션을 통해 순환 신경망 모델의 단점을 해결하고 딥러닝계의 `G.O.A.T` 자리를 차지했는지 알아보자. 모델은 크게 인코더와 디코더 부분으로 나뉘는데, 하는 역할과 미세한 구조상의 차이만 있을뿐 두 모듈 모두 `Self-Attention`이 제일 중요하다는 본질은 변하지 않는다. 따라서 `Self-Attention` 은 특별히 사용된 하위 블럭 단위를 생략없이 모두 살펴볼 것이다. 그리고 나서 인코더와 디코더에 사용된 다른 모듈과 모델의 전반적인 구조에 대해 대해서 공부해보자.

<p markdown="1" align="center">
![Class Diagram](/assets/images/transformer/class_diagram.png){: .align-center}{: width="35%", height="50%"}{: .image-caption}
__*Class Diagram*__
</p>

이후에는 모델을 실제 코드로 어떻게 구현해야 하는지도 함께 알아볼 것이니까 포스트를 끝까지 읽어주시길 바란다.

#### `📐 Self-Attention with linear projection`

왜 이름이 `self-attention`일까 먼저 고민해보자. 사실 `attention` 개념은 본 논문이 발표되기 이전부터 사용되던 개념이다. `attention`은 `seq2seq` 구조에서 처음 나왔는데, `seq2seq` 은 번역 성능을 높이는 것을 목적으로 고안된 구조라서, 목표인 디코더의 `hidden_states` 값을 쿼리로, 인코더의 `hidden_states`를 키, 벨류의 출처로 사용했다. 즉, 서로 다른 출처에서 나온 `hidden_states` 을 사용해 내적 연산을 수행했던 것이다. 이런 개념에 이제 `“self"` 라는 이름이 붙었다. 결국 같은 출처에서 나온 `hidden_states` 를 내적하겠다는 의미를 내포하고 있는 것이다. 내적은 두 벡터의 `“닮은 정도”` 를 수학적으로 계산한다. 따라서 `self-attention` 이란 간단하게, 같은 출처에서 만들어진 $Q$(쿼리), $K$(키), $V$(벨류)가 `서로 얼마나 닮았는지` 계산해보겠다는 것이다.

<p markdown="1" align="center">
![self-attention with linear projection](/assets/images/transformer/linear_projection.png){: .align-center}{: width="50%", height="50%"}{: .image-caption}
__*[self-attention with linear projection](https://jalammar.github.io/illustrated-transformer/)*__
</p>

그렇다면 이제 $Q$(쿼리), $K$(키), $V$(벨류)의 정체, 같은 출처에서 나왔다는 말의 의미 그리고 입력 행렬 $X$를 `linear projection` 하여 $Q$(쿼리), $K$(키), $V$(벨류) 행렬을 만드는 이유를 **구체적인 예시를 통해 이해해보자.** 추가로 $Q$(쿼리), $K$(키), $V$(벨류) 개념은 `Information Retrieval`에서 먼저 파생된 개념이라서 예시 역시 정보 검색과 관련된 것으로 준비했다. 

당신이 만약 `“에어컨 필터 청소하는 방법”`이 궁금해 구글에 검색하는 상황이라고 가정해보겠다. **목표는 가장 빠르고 정확하게 내가 원하는 필터 청소 방법에 대한 지식을 획득하는 것이다.** **`그렇다면 당신은 뭐라고 구글 검색창에 검색할 것인가??`** **이것이 바로** $Q$**(쿼리)에 해당한다.** 당신은 검색창에 `“에어컨 필터 청소하는 방법”`을 입력해 검색 결과를 반환 받았다. **반환 받은 결과물의 집합이 바로** $K$**(키)가 된다.** 당신은 총 100개의 블로그 게시물을 키 값으로 받았다. 그래서 당신이 사용하는 삼성 무풍 에어컨의 필터 청소법이 정확히 적힌 게시물을 찾기 위해 하나 하나 링크를 타고 들어가 보았다. 하지만 정확하게 원하는 정보가 없어서 계속 찾다보니 결국 4페이지 쯤에서 원하던 정보가 담긴 게시물을 찾을 수 있었다. **이렇게 내가 원하는 정보인지 아닌지 대조하는 과정이 바로** $Q$**(쿼리)와** $K$**(키) 행렬을** `내적`**하는 행위가 된다.** 곧바로 에어컨 청소를 하려고 보니, 방법을 까먹어서 매년 여름마다 검색을 해야할 것 같아 해당 게시물을 북마크에 저장해두었다. **여기서 북마크가 바로** $V$**(벨류) 행렬이 된다.**

이 모든 과정에 10분이 걸렸다. 겨우 필터 청소 방법을 찾는데 10분이라니 당신은 자존심이 상했다. `더 빨리 원하는 정보(손실 함수 최적화)`를 찾을 수 있는 방법이 없을까 고민해보다가 `당신이 사용하는 에어컨 브랜드명(삼성 Bespoke 에어컨)을 검색어에 추가하기로 했다`. 그랬더니 1페이지 최하단에서 아까 4페이지에서 찾은 정보를 곧바로 찾을 수 있었다. 그 덕분에 시간을 `10분`에서 `1분 30초`로 단축시킬 수 있었다. **이렇게 검색 시간을 단축(손실 줄이기)하기 위해 더 나은 검색 표현을 고민하고 수정하는 행위가 바로 입력** $X$에 $W_{Q}$**를 곱해 행렬** $Q$ **을 만드는 수식으로 표현된다.**

1년 뒤 여름, 당신은 브라우저를 바꾼 탓에 북마크가 초기화 되어 다시 한 번 검색을 해야 했다. 하지만 여전히 검색어는 기억하고 있어서, 1년전 최적의 결과를 얻었던 그대로 다시 검색을 했다. 분명 똑같이 검색을 했는데 같은 결과가 1페이지 최상단에서 반환되고 있었다. 당신은 이게 어떻게 된 일인지 궁금해 포스트를 천천히 보던 중, 제목에 1년전에는 없던 `삼성 Bespoke 에어컨` 이라는 키워드가 포함 되어 있었다. 게시물의 주인장이 `SEO 최적화`를 위해 추가했던 것이었다. 덕분에 당신은 소요 시간을 `1분 30초`에서 `20초`로 줄일 수 있었다. **이런 상황이 바로 입력** $X$에 $W_{K}$**를 곱해 행렬** $K$ **를 만드는 수식에 대응된다.**

우리는 위 예시를 통해 원하는 정보를 빠르고 정확하게 찾는 행위란, 답변자가 이해하기 좋은 질문과 질문자의 질문 의도에 부합하는 좋은 답변으로 완성된다는 것을 알 수 있었다. 뿐만 아니라, 좋은 질문과 좋은 답변이라는 것은 처음부터 완성되는게 아니라 **검색 시간을 단축하려는 끊임없는 노력**을 통해 성취된다는 것 역시 깨우쳤다. 두가지 인사이트가 바로 `linear projection`으로 행렬 $Q, K,V$을 정의한 이유다. **내가 원하는 정보인지 아닌지 대조하는 내적 연산은 수행하는데 가중치 행렬이 필요 없기 때문에 손실함수의 오차 역전을 활용한 수치 최적화를 수행할 수 없다.** 그래서 손실함수 미분에 의한 최적화가 가능하도록  `linear projection matrix`를 활용해 행렬 $Q, K,V$를 정의해준 것이다. **이렇게 하면 모델이 우리의 목적에 가장 적합한 질문과 답변을 알아서 표현 해줄 것이라 기대할 수 있게 된다.** 한편, 같은 출처에서 나왔다는 말은 방금 예시에서 행렬 $Q, K,V$를 만드는데 동일하게 입력 $X$를 사용 것과 같은 상황을 의미한다.

이제 다시 자연어 처리 맥락으로 돌아와보자. `Transformer` 는 좋은 번역기를 만들기 위해 고안된 `seq2seq` 구조의 모델이다. 즉, 빠르고 정확하게 대상 언어에서 타겟 언어로 번역하는 것에 목표를 두고 만들어졌다는 것이다. 번역을 잘하기 위해서는 어떻게 해야 할까?? **1) 대상 언어로 쓰인 시퀀스의 의미를 정확하게 파악해야 하고, 2) 파악한 의미와 가장 유사한 시퀀스를 타겟 언어로 만들어 내야 한다.** `그래서 1번의 역할은 Encoder가 그리고 2번은 Decoder가 맡게 된다`. 인코더는 결국 (번역하는데 적합한 형태로) 대상 언어의 의미를 정확히 이해하는 방향(숫자로 표현, 임베딩 추출)으로 학습을 수행하게 되며, 디코더는 인코더의 학습 결과와 가장 유사한 문장을 타겟 언어로 생성해내는 과정을 배우게 된다. 따라서 인코더는 대상 언어를 출처로, 디코더는 타겟 언어를 출처로 행렬 $Q, K,V$를 만든다. 정확히 `self` 라는 단어를 이름에 갖다 붙인 의도와 일맥상통하는 모습이다.