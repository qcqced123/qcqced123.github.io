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

그렇다면 이제부터 `Transformer` 가 어떤 아이데이션을 통해 기존 순환 신경망 모델의 단점을 해결하고 딥러닝계의 `G.O.A.T` 자리를 차지했는지 알아보자. 모델은 크게 인코더와 디코더 부분으로 나뉘는데, 하는 역할과 미세한 구조상의 차이만 있을뿐 두 모듈 모두 `Self-Attention`이 제일 중요하다는 본질은 변하지 않는다. 따라서 `Input Embedding`부터 차례대로 살펴보되,  `Self-Attention` 은 특별히 사용된 하위 블럭 단위를 빠짐 없이, 세세하게 살펴볼 것이다.

<p markdown="1" align="center">
![Class Diagram](/assets/images/transformer/class_diagram.png){: .align-center}{: width="35%", height="50%"}{: .image-caption}
__*Class Diagram*__
</p>

이렇게 하위 모듈에 대한 설명부터 쌓아 나가 마지막에는 실제 구현 코드와 함께 전체적인 구조 측면에서도 모델을 해석해볼 것이다. 끝까지 포스팅을 읽어주시길 바란다.

#### `🔬 Input Embedding`

$$
X_E \in R^{B * S_E * V_E} \\
X_D \in R^{B * S_D * V_D}
$$

`Transformer`는 인코더와 디코더로 이뤄진 `seq2seq` 구조를 가지고 있다. 즉, 대상 언어를 타겟 언어로 번역하는데 목적을 두고 있기 때문에 입력으로 대상 언어 시퀀스와 타겟 언어 시퀀스 모두 필요하다. $X_E$는 `인코더`의 입력 행렬을 나타내고, $X_D$는 `디코더`의 입력 행렬을 의미한다. 이 때, $B$는 `batch size`, $S$는 `max_seq`, $V$는 개별 모듈이 가진 `Vocab`의 사이즈를 가리킨다. 위 수식은 사실 논문에 입력에 대한 수식이 따로 서술 되어 있지 않아, 필자가 직접 만든 것이다. 앞으로도 해당 기호를 이용해 수식을 표현할 예정이니 참고 바란다.

$$
W_E \in R^{V_E * d} \\
W_D \in R^{V_D * d} \\
$$

이렇게 정의된 입력값을 개별 모듈의 임베딩 레이어에 통과 시킨 결과물이 바로 `Input Embedding`이 된다. $d$는 `Transformer` 모델의 은닉층의 크기를 의미한다. 따라서 `Position Embedding` 과 더해지기 전, 임베딩 레이어를 통과한 `Input Embedding`의 모양은 아래 수식과 같다.

$$
X_E \in R^{B*S_E*d} \\
X_D \in R^{B*S_D*d} \\
$$

그렇다면 실제 구현은 어떻게 할까?? `Transformer` 의 `Input Embedding`은 `nn.Embedding`으로 레이어를 정의해 사용한다. `nn.Linear`도 있는데 왜 굳이 `nn.Embedding`을 사용하는 것일까??

자연어 처리에서 입력 임베딩을 만들때는 모델의 토크나이저에 의해 사전 정의된 `vocab`의 사이즈가 입력 시퀀스에 속한 토큰 개수보다 훨씬 크기 때문에 데이터 룩업 테이블 방식의 `nn.Embedding` 을 사용하게 된다. 이게 무슨 말이냐면, 토크나이저에 의해 사전에 정의된 `vocab` 전체가 `nn.Embedding(vocab_size, dim_model)`로 투영 되어 가로는 `vocab` 사이즈, 세로는 모델의 차원 크기에 해당하는 룩업 테이블이 생성되고, 내가 입력한 토큰들은 전체 `vocab`의 일부분일테니 전체 임베딩 룩업 테이블에서 내가 임베딩하고 싶은 토큰들의 인덱스만 알아낸다는 것이다. 그래서 `nn.Embedding` 은 레이어에 정의된 차원과 실제 입력 데이터의 차원이 맞지 않아도 함수가 동작하게 된다. `nn.Linear` 와 입력 차원에 대한 조건 빼고는 동일한 동작을 수행하기 때문에 사전 정의된 `vocab` 사이즈와 입력 시퀀스의 토큰 개수가 같다면 `nn.Linear`를 사용해도 무방하다.

```python
# Input Embedding Example

class Transformer(nn.Module):
	def __init__(
        self,
        enc_vocab_size: int,
        dec_vocab_size: int,
        max_seq: int = 512,
        enc_N: int = 6,
        dec_N: int = 6,
        dim_model: int = 512, # latent vector space
        num_heads: int = 8,
        dim_ffn: int = 2048,
        dropout: float = 0.1
    ) -> None:
        super(Transformer, self).__init__()
        self.enc_input_embedding = nn.Embedding(enc_vocab_size, dim_model) # Encoder Input Embedding Layer
        self.dec_input_embedding = nn.Embedding(dec_vocab_size, dim_model) # Decoder Input Embedding Layer
	
	def forward(self, enc_inputs: Tensor, dec_inputs: Tensor, enc_pad_index: int, dec_pad_index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            enc_x, dec_x = self.enc_input_embedding(enc_inputs), self.dec_input_embedding(dec_inputs)
```

위의 예시 코드를 함께 살펴보자. `__init__` 의 `self.enc_input_embedding`, `self._dec_input_embedding`이 바로 $W_E, W_D$에 대응된다. 한편 `forward` 메서드에 정의된 `enc_x`, `dec_x` 는 임베딩 레이어를 거치고 나온 $X_E, X_D$에 해당된다.

한편, $X_E, X_D$은 각각 인코더, 디코더 모듈로 흘러 들어가 `Absolute Position Embedding`과 더해진(행렬 합) 뒤, 개별 모듈의 입력값으로 활용된다.

**`🔢 Absolute Position Embedding(Encoding)`**

`Absolute Position Embedding(Encoding)`은 입력 시퀀스에 위치 정보를 맵핑해주는 역할을 한다. 필자는 개인적으로 `Transformer`에서 가장 중요한 요소를 뽑으라고 하면 세 손가락 안에 들어가는 파트라고 생각한다. 다음 파트에서 자세히 기술하겠지만, `Self-Attention(내적)`은 입력 시퀀스를 병렬로 한꺼번에 처리할 수 있다는 장점을 갖고 있지만, 그 자체로는 토큰의 위치 정보를 인코딩할 수 없다. 우리가 따로 위치 정보를 알려주지 않는 이상 쿼리 행렬의 2번째 행벡터가 입력 시퀀스에서 몇 번째 위치한 토큰인지 모델은 알 길이 없다. 

그런데, 텍스트는 `Permutation Equivariant`한 `Bias` 가 있기 때문에 토큰의 위치 정보는 `NLP`에서 매우 중요한 요소로 꼽힌다. **직관적으로도 토큰의 순서는 시퀀스가 내포하는 의미에 지대한 영향을 끼친다는 것을 알 수 있다.** 예를 들어 `“철수는 영희를 좋아한다”`라는 문장과 `“영희는 철수를 좋아한다”`라는 문장의 의미가 같은가 생각해보자. 주어와 목적어 위치가 바뀌면서 정반대의 뜻이 되어버린다.

<p markdown="1" align="center">
![Positional Encoding Example](/assets/images/transformer/positional_encoding.png){: .align-center}{: width="75%", height="50%"}{: .image-caption}
__*[Positional Encoding Example](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/56cf1596-c770-410c-8053-5876c3c66fff/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-10-09_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5.13.48.png)*__
</p>

따라서 저자는 입력 입베딩에 위치 정보를 추가하고자 `Position Encoding` 을 제안한다. 사실 `Position Encoding` 은 여러 단점 때문에 후대 `Transformer`  파생 모델에서는 잘 사용되지 않는 추세다. 대신 모델이 학습을 통해 최적값을 찾아주는 `Position Embedding` 방식을 대부분 차용하고 있다. 필자 역시 `Position Embedding` 을 사용해 위치 임베딩을 구현했기 때문에 원리와 단점에 대해서만 간단히 소개하고 넘어가려 한다. 또한 저자 역시 논문에서 두 방식 중 어느 것을 써도 비슷한 성능을 보여준다고 언급하고 있다. 

$$
P_E \in R^{B*S_E*D} \\
 P_D \in R^{B*S_D*D} \\
P(pos, 2i) = sin(pos/\overset{}
  {10000_{}^{2i/dmodel}}) \\
P(pos, 2i+1) = cos(pos/\overset{}
  {10000_{}^{2i/dmodel}})
$$

**원리는 매우 간단하다. 사인함수와 코사인 함수의 주기성을 이용해 개별 인덱스의 행벡터 값을 표현하는 것이다.** 행벡터의 원소 중에서 짝수번째 인덱스에 위치한 원소는 (짝수번째 열벡터) $$sin(pos/\overset{}{10000_{}^{2i/dmodel}})$$ 의 함숫값을 이용해 채워넣고, 홀수번째 원소는 $$cos(pos/\overset{}{10000_{}^{2i/dmodel}})$$를 이용해 채워넣는다.

<p markdown="1" align="center">
![periodic function graph](/assets/images/transformer/sin_cos_graph.png){: .align-center}{: width="75%", height="50%"}{: .image-caption}
__*periodic function graph*__
</p>

초록색 그래프는 $$sin(pos/\overset{}{10000_{}^{2i/dmodel}})$$, 주황색 그래프는 $$cos(pos/\overset{}{10000_{}^{2i/dmodel}})$$를 시각화했다. 지면의 제한으로 `max_seq=512` 만큼의 변화량을 담지는 못했지만, x축이 커질수록 두 함수 모두 진동 주기가 조금씩 커지는 양상을 보여준다. 따라서 개별 인덱스(행벡터)를 중복되는 값 없이 표현하는 것이 가능하다고 저자는 주장한다.

<p markdown="1" align="center">
![Positional Encoding Result](/assets/images/transformer/positional_encoding_result.png){: .align-center}{: width="75%", height="50%"}{: .image-caption}
__*[Positional Encoding Result](https://wikidocs.net/162099)*__
</p>

위 그림은 토큰 `256`개로 구성된 시퀀스에 대해 `Positional Encoding`한 결과를 시각화한 자료다. 그래프의 $x$축은 `행벡터의 원소`이자 `Transformer`의 은닉 벡터 차원을 가리키고, $y$축은 `시퀀스의 인덱스`(행벡터)를 의미한다. 육안으로 정확하게 차이를 인식하기 쉽지는 않지만, 행벡터가 모두 유니크하게 표현된다는 사실(직접 실수값을 확인해보면 정말 미세한 차이지만 개별 토큰의 희소성이 보장)을 알 수 있다. 작은 차이를 시각화 자료로 파악하기는 쉽지 않기 때문에 진짜 그런가 궁금하신 분들은 직접 실수값을 구해보는 것을 추천드린다. 

**여기서 행벡터의 희소성이란 개별 행벡터 원소의 희소성을 말하는게 아니다.** 0번 토큰, 4번 토큰, 9번 토큰의 행벡터 1번째 원소의 값은 같을 수 있다. 하지만 진동 주기가 갈수록 커지는 주기함수를 사용하기 때문에 다른 원소(차원)값은 다를 것이라 기대할 수 있는데, **바로 이것을 행벡터의 희소성이라고 정의하는 것이다.** 만약 1번 토큰과 2번 토큰의 모든 행벡터 원소값이 같다면 그것은 희소성 원칙에 위배되는 상황이다.

<p markdown="1" align="center">
![Positional Encoding](/assets/images/transformer/encoding.png){: .align-center}{: width="75%", height="50%"}{: .image-caption}
</p>

<p markdown="1" align="center">
![Compare Performance between Encoding and Embedding](/assets/images/transformer/embedding.png){: .align-center}{: width="75%", height="50%"}{: .image-caption}
__*[Compare Performance between Encoding and Embedding](https://arxiv.org/abs/1706.03762)*__
</p>

비록 개별 행벡터의 희소성이 보장된다고 해도 `Position Encoding`은 `not trainable`해서 `static`하다는 단점이 있다. 모든 배치의 시퀀스가 동일한 위치 정보값을 갖게 된다는 것이다. `512`개의 토큰으로 구성된 시퀀스 A와 B가 있다고 가정해보자. 이 때 시퀀스 A는 문장 `5`개로 구성 되어 있고, B는 문장 `12`개로 만들어졌다. 두 시퀀스의 `11`번째 토큰의 문장 성분은 과연 같을까?? 아마도 대부분의 경우에 다를 것이다. 텍스트 데이터에서 순서 정보가 중요한 이유 중 하나는 바로 `syntactical` 한 정보를 포착하기 위함이다. `Position Encoding`은 `static` 하기 때문에 이러한 타입의 정보를 인코딩 하기 쉽지 않다. 그래서 좀 더 풍부한 표현을 담을 수 있는 `Position Embedding`을 사용하는 것이 최근 추세다. 

**`✏️ Position Embedding`**

그렇다면 이제 `Position Embedding`에 대해 알아보자. `Position Embedding` 은 `Input Embedding`을 정의한 방식과 거의 유사하다. 먼저 입력값과 `weight` 의 모양부터 확인해보자.

$$
P_E \in R^{B*S_E*d} \\
P_D \in R^{B*S_d*d} \\
W_{P_E} \in R^{S_E * d} \\
W_{P_D} \in R^{S_D * d} \\
$$

$P_E, P_D$는 개별 모듈의 위치 임베딩 레이어 입력을 가리키며, $W_{P_E}, W_{P_D}$가 개별 모듈의 위치 임베딩 레이어가 된다. 이제 이것을 코드로 어떻게 구현하는지 살펴보자.

```python
# Absolute Position Embedding Example

class Encoder(nn.Module):
    """
    In this class, encode input sequence and then we stack N EncoderLayer
    First, we define "positional embedding" and then add to input embedding for making "word embedding"
    Second, forward "word embedding" to N EncoderLayer and then get output embedding
    In official paper, they use positional encoding, which is base on sinusoidal function(fixed, not learnable)
    But we use "positional embedding" which is learnable from training
    Args:
        max_seq: maximum sequence length, default 512 from official paper
        N: number of EncoderLayer, default 6 for base model
    """
    def __init__(self, max_seq: 512, N: int = 6, dim_model: int = 512, num_heads: int = 8, dim_ffn: int = 2048, dropout: float = 0.1) -> None:
        super(Encoder, self).__init__()
        self.max_seq = max_seq
        self.scale = torch.sqrt(torch.Tensor(dim_model))  # scale factor for input embedding from official paper
        self.positional_embedding = nn.Embedding(max_seq, dim_model)  # add 1 for cls token

		... 중략 ...
		def forward(self, inputs: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        inputs: embedding from input sequence, shape => [BS, SEQ_LEN, DIM_MODEL]
        mask: mask for Encoder padded token for speeding up to calculate attention score
        """
        layer_output = []
        pos_x = torch.arange(self.max_seq).repeat(inputs.shape[0]).to(inputs)
        x = self.dropout(
            self.scale * inputs + self.positional_embedding(pos_x)  # layernorm 적용하고
        )
		... 중략 ... 
```

위 코드는 `Transformer`의 인코더 모듈을 구현한 것이다. 그래서 `forward` 메서드의 `pos_x` 가 바로 $P_E$가 되며, `__init__`의 `self.positional_embedding`이 바로 $W_{P_E}$에 대응된다. 이렇게 정의한 `Position Embedding`은 `Input Embedding`과 더해서 `Word Embedding` 을 만든다. `Word Embedding` 은 다시 개별 모듈의 `linear projection` 레이어에 대한 입력 $X$로 사용 된다. 

**한편,** `Input Embedding` **과** `Position Embedding`**을 더한다는 것에 주목해보자. 필자는 본 논문을 보며 가장 의문이 들었던 부분이다. 도대체 왜 완전히 서로 다른 출처에서 만들어진 행렬 두개를** `concat` **하지 않고 더해서 사용했을까??** `concat`**을 이용하면 `Input`과 `Position` 정보를 서로 다른 차원에 두고 학습하는게 가능했을텐데 말이다.**

**`🤔 Why Sum instead of Concatenate`**

행렬합을 사용하는 이유에 대해 저자가 특별히 언급하지는 않아서 때문에 정확한 의도를 알 수 없지만, **추측하건데 `blessing of dimensionality` 효과를 의도했지 않았나 싶다.** `blessing of dimensionality` 란, 고차원 공간에서 무작위로 서로 다른 벡터 두개를 선택하면 두 벡터는 거의 대부분 `approximate orthogonality`를 갖는 현상을 설명하는 용어다. 무조건 성립하는 성질은 아니고 확률론적인 접근이라는 것을 명심하자. 아무튼 직교하는 두 벡터는 내적값이 0에 수렴한다. 즉, 두 벡터는 서로에게 영향을 미치지 못한다는 것이다. 이것은 전체 모델의 `hidden states space` 에서 `Input Embedding` 과 `Position Embedding` 역시 개별 벡터가 `span` 하는 부분 공간 끼리는 서로 직교할 가능성이 매우 높다는 것을 의미한다. 따라서 서로 다른 출처를 통해 만들어진 두 행렬을 더해도 서로에게 영향을 미치지 못할 것이고 그로 인해 모델이 `Input`과 `Position` 정보를 따로 잘 학습할 수 있을 것이라 기대해볼 수 있다. 가정대로만 된다면, `concat` 을 사용해 모델의 `hidden states space` 를 늘려 `Computational Overhead` 를 유발하는 것보다 훨씬 효율적이라고 볼 수 있겠다.

한편 `blessing of dimensionality`에 대한 설명과 증명은 꽤나 많은 내용이 필요해 여기서는 자세히 다루지 않고, 다른 포스트에서 따로 다루겠다. 관련하여 좋은 내용을 담고 있는 글의 링크를 같이 첨부했으니 읽어보실 것을 권한다([링크1](https://softwaredoug.com/blog/2022/12/26/surpries-at-hi-dimensions-orthoginality.html), [링크2](https://www.reddit.com/r/MachineLearning/comments/cttefo/comment/exs7d08/)).


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

**결국** `Transformer` **의 성능을 좌지우지 하는 것은 누가 얼마나 더** `linear projection weight`**을 잘 최적화 하는가에 달렸다고 볼 수 있다.**

**한편 필자는 처음 이 논문을 읽었을 때** `linear projection` **자체의 필요성은 공감했으나, 굳이 3개의 행렬로 나눠서** `train` **시켜야 하는** `param` **숫자를 늘리는 것보다는** `weight share` **하는 형태로 만드는게 더 효율적일 것 같다는 추측을 했었다.** 

그러나 이번 리뷰를 위해 다시 논문을 읽던 중, 좋은 질문을 하기 위한 노력과 좋은 답변을 하기 위한 노력, 그리고 필요한 정보를 정확히 추출해내는 행위를 각각 서로 다른 3개의 벡터로 표현했을 때 **벡터들이 가지는 방향성이 서로 다를텐데** 그것을 하나의 벡터로 표현하려면 모델이 학습을 하기 힘들 것 같다는 생각이 들었다. 방금 위에서 든 예시만 봐도 그렇다. 서로 다른 3개의 행위 사이의 최적 지점을 찾으라는 것과 마찬가진데 그런 스팟이 있다고 해도 언어 모델이 잘 찾을 수 있을까?? 인간도 찾기 힘든 것을 모델이 잘 찾을리가 없다.

**`📐 Scaled Dot-Product Attention`**