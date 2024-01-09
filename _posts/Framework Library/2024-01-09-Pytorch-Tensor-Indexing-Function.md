---
title: "🔥 Pytorch Tensor Indexing 자주 사용하는 메서드 모음집"
excerpt: "파이토치에서 자주 사용하는 텐서 인덱싱 관련 메서드 모음"
permalink: "/framework-library/torch-indexing-function"
toc: true  # option for table of content
toc_sticky: true  # option for table of content
categories:
  - Framework & Library
tags:
  - Pytorch
  - Tensor
  - Linear Algebra
  
last_modified_at: 2024-01-09T12:00:00-05:00
---
파이토치에서 필자가 자주 사용하는 텐서 인덱싱 관련 메서드의 사용법 및 사용 예시를 한방에 정리한 포스트다. 메서드 하나당 하나의 포스트로 만들기에는 너무 길이가 짧다 생각해 한 페이지에 모두 넣게 되었다. 지속적으로 업데이트 될 예정이다. 또한 텐서 인덱싱 말고도 다른 주제로도 관련 메서드를 정리해 올릴 예정이니 많은 관심 부탁드린다.

### `🔎 torch.argmax`

입력 텐서에서 가장 큰 값을 갖고 있는 원소의 인덱스를 반환한다. 최대값을 찾을 차원을 지정해줄 수 있다. 아래 예시 코드를 확인해보자.

```python
# torch.argmax params
torch.argmax(tensor, dim=None, keepdim=False)

# torch.argmax example 1
test = torch.tensor([1,29,2,45,22,3])
torch.argmax(test)
torch.argmax(test2, keepdim=True)

<Result>
tensor(3)

# torch.argmax example 2
test = torch.tensor([[4, 2, 3],
                     [4, 5, 6]])

torch.argmax(test, dim=0, keepdim=True)
<Result>
tensor([[0, 1, 1]])

# torch.argmax example 3
test = torch.tensor([[4, 2, 3],
                     [4, 5, 6]])

torch.argmax(test, dim=1, keepdim=True)
tensor([[0],
        [2]])
```

`dim` 매개변수에 원하는 차원을 입력하면 해당 차원 뷰에서 가장 큰 원소를 찾아 인덱스 값을 반환해줄 것이다. 이 때 `keepdim=True` 로 설정한다면 입력 차원에서 가장 큰 원소의 인덱스를 반환하되 원본 텐서의 차원과 동일한 형태로 출력해준다. `example 2` 의 경우 `dim=0` 라서 행이 누적된 방향으로 텐서를 바라봐야 한다. 행이 누적된 방향으로 텐서를 보게 되면 `tensor([[0, 1, 1]])`이 된다.

### `📚 torch.stack`

```python
"""
torch.stack
Args:
	tensors(sequence of Tensors): 텐서가 담긴 파이썬 시퀀스 객체
	dim(int): 추가할 차원 방향을 세팅, 기본값은 0
"""
torch.stack(tensors, dim=0)
```

매개변수로 주어진 파이썬 시퀀스 객체(리스트, 튜플)를 사용자가 지정한 새로운 차원에 쌓는 기능을 한다. 매개변수 `tensors` 는 텐서가 담긴 파이썬의 시퀀스 객체를 입력으로 받는다. `dim` 은 사용자가 텐서 적재를 하고 싶은 새로운 차원을 지정해주면 된다. 기본값은 0차원으로 지정 되어있으며, 텐서의 맨 앞차원이 새롭게 생기게 된다. `torch.stack` 은 기계학습, 특히 딥러닝에서 정말 자주 사용되기 때문에 사용법 및 사용상황을 익혀두면 도움이 된다. 예시를 통해 해당 메서드를 어떤 상황에서 어떻게 사용하는지 알아보자.

```python
""" torch.stack example """

class Projector(nn.Module):
    """
    Making projection matrix(Q, K, V) for each attention head
    When you call this class, it returns projection matrix of each attention head
    For example, if you call this class with 8 heads, it returns 8 set of projection matrices (Q, K, V)
    Args:
        num_heads: number of heads in MHA, default 8
        dim_head: dimension of each attention head, default 64
    """
    def __init__(self, num_heads: int = 8, dim_head: int = 64) -> None:
        super(Projector, self).__init__()
        self.dim_model = num_heads * dim_head
        self.num_heads = num_heads
        self.dim_head = dim_head

    def __call__(self):
        fc_q = nn.Linear(self.dim_model, self.dim_head)
        fc_k = nn.Linear(self.dim_model, self.dim_head)
        fc_v = nn.Linear(self.dim_model, self.dim_head)
        return fc_q, fc_k, fc_v

num_heads = 8
dim_head = 64
projector = Projector(num_heads, dim_head)  # init instance
projector_list = [list(projector()) for _ in range(num_heads)]  # call instance
x = torch.rand(10, 512, 512) # x.shape: [Batch_Size, Sequence_Length, Dim_model]
Q, K, V = [], [], []

for i in range(num_heads):
    Q.append(projector_list[i][0](x)) # [10, 512, 64]
    K.append(projector_list[i][1](x)) # [10, 512, 64]
	  V.append(projector_list[i][2](x)) # [10, 512, 64]
 
Q = torch.stack(Q, dim=1) # Q.shape: [10, 8, 512, 64]
K = torch.stack(K, dim=1) # K.shape: [10, 8, 512, 64]
V = torch.stack(V, dim=1) # V.shape: [10, 8, 512, 64]
```

위 코드는 `Transformer` 의 `Multi-Head Attention` 구현체 일부를 발췌해온 것이다. `Multi-Head Attention` 은 개별 어텐션 해드별로 행렬 $Q, K, V$를 가져야 한다. 따라서 입력 임베딩을 개별 어텐션 헤드에 `Linear Combination` 해줘야 하는데 헤드 개수가 8개나 되기 때문에 개별적으로 `Projection Matrix` 를 선언해주는 것은 매우 비효율적이다. 따라서 객체  `Projector` 에 행렬 $Q, K, V$에 대한 `Projection Matrix` 를 정의해줬다. 이후 헤드 개수만큼 객체  `Projector` 를 호출해 리스트에 해드별 `Projection Matrix` 를 담아준다. 그 다음 `torch.stack`을 사용해 `Attention Head` 방향의 차원으로 리스트 내부 텐서들을 쌓아주면 된다.

### `🔢 torch.arange`

사용자가 지정한 시작점부터 끝점까지 일정한 간격으로 텐서를 나열한다. Python의 내장 메서드 `range`와 동일한 역할을 하는데, 대신 텐서 그 결과를 텐서 구조체로 반환한다고 생각하면 되겠다.

```python
# torch.arange usage
torch.arange(start=0, end, step=1)

>>> torch.arange(5)
tensor([ 0,  1,  2,  3,  4])

>>> torch.arange(1, 4)
tensor([ 1,  2,  3])

>>> torch.arange(1, 2.5, 0.5)
tensor([ 1.0000,  1.5000,  2.0000])
```

`step` 매개변수로 원소간 간격 조정을 할 수 있는데, 기본은 1로 지정 되어 있으니 참고하자. 필자의 경우에는 `nn.Embedding`의 입력 텐서를 만들 때 가장 많이 사용했다. `nn.Embedding` 의 경우 Input으로 `IntTensor`, `LongTensor`를 받게 되어 있으니 알아두자. 

### `🔁 torch.repeat`

입력값으로 주어진 텐서를 사용자가 지정한 반복 횟수만큼 특정 차원 방향으로 늘린다. 예를 들면 `[1,2,3] * 3`의 결과는 `[1, 2, 3, 1, 2, 3, 1, 2, 3]` 인데, 이것을 사용자가 지정한 반복 횟수만큼 특정 차원으로 수행하겠다는 것이다. 아래 사용 예제를 확인해보자.

```python
# torch.repeat example

>>> x = torch.tensor([1, 2, 3])
>>> x.repeat(4, 2)
tensor([[ 1,  2,  3,  1,  2,  3],
        [ 1,  2,  3,  1,  2,  3],
        [ 1,  2,  3,  1,  2,  3],
        [ 1,  2,  3,  1,  2,  3]])

>>> x.repeat(4, 2, 1)
tensor([[[1, 2, 3],
         [1, 2, 3]],

        [[1, 2, 3],
         [1, 2, 3]],

        [[1, 2, 3],
         [1, 2, 3]],

        [[1, 2, 3],
         [1, 2, 3]]])

>>> x.repeat(4, 2, 1).size
torch.Size([4, 2, 3])

>>> x.repeat(4, 2, 2)
tensor([[[1, 2, 3, 1, 2, 3],
         [1, 2, 3, 1, 2, 3]],

        [[1, 2, 3, 1, 2, 3],
         [1, 2, 3, 1, 2, 3]],

        [[1, 2, 3, 1, 2, 3],
         [1, 2, 3, 1, 2, 3]],

        [[1, 2, 3, 1, 2, 3],
         [1, 2, 3, 1, 2, 3]]])
```

 $t$를 어떤 텐서 구조체 $x$의 최대 차원이라고 했을 , $x_t$를 가장 왼쪽에 넣고 가장 낮은 차원인 0차원에 대한 반복 횟수를 오른쪽 끝에 대입해서 사용하면 된다. (`torch.repeat(`$x_t, x_{t-1}, ... x_2, x_1, x_0$`))`. 

```python
# torch.arange & torch.repeate usage example

>>> pos_x = torch.arange(self.num_patches + 1).repeat(inputs.shape[0]).to(inputs)
>>> pos_x.shape
torch.tensor([16, 1025])
```

필자의 경우, `position embedding`의 입력을 만들고 싶을 때 `torch.arange` 와 연계해 자주 사용 했던 것 같다. 위 코드를 참고하자.

### `🔬 torch.clamp`

입력 텐서의 원소값을 사용자가 지정한 최대•최소값 범위 이내로 제한하는 메서드다.

```python
# torch.clamp params

>>> torch.clamp(input, min=None, max=None, *, out=None) → Tensor

# torch.clamp usage example

>>> a = torch.randn(4)
>>> a
tensor([-1.7120,  0.1734, -0.0478, -0.0922])

>>> torch.clamp(a, min=-0.5, max=0.5)
tensor([-0.5000,  0.1734, -0.0478, -0.0922])
```

입력된 텐서의 원소를 지정 최대•최소 설정값과 하나 하나 대조해서 텐서 내부의 모든 원소가 지정 범위 안에 들도록 만들어준다. `torch.clamp` 역시 다양한 상황에서 사용되는데, 필자의 경우 모델 레이어 중간에 제곱근이나 지수, 분수 혹은 각도 관련 연산이 들어가 `Backward Pass`에서 `NaN`이 발생할 수 있는 경우에 안전장치로 많이 사용하고 있다. ([자세히 알고 싶다면 클릭](https://qcqced123.github.io/framework-library/backward-nan/))

### `👩‍👩‍👧‍👦 torch.gather`

텐서 객체 내부에서 원하는 인덱스에 위치한 원소만 추출하고 싶을 때 사용하면 매우 유용한 메서드다. 텐서 역시 `iterable` 객체라서 `loop` 를 사용해 접근하는 것이 직관적으로 보일 수 있으나, 통상적으로 텐서를 사용하는 상황이라면 객체의 차원이 어마무시 하기 때문에 루프로 접근해 관리하는 것은 매우 비효율적이다. 루프를 통해 접근하면 파이썬의 내장 리스트를 사용하는 것과 별반 다를게 없어지기 때문에, 텐서를 사용하는 메리트가 사라진다. 비교적 크지 않은 2~3차원의 텐서 정도라면 사용해도 크게 문제는 없을거라 생각하지만 그래도 코드의 일관성을 위해 `torch.gather` 사용을 권장한다. 이제 `torch.gather`의 사용법에 대해 알아보자.

```python
# torch.gather params

>>> torch.gather(input, dim, index, *, sparse_grad=False, out=None)
```

`dim`과 `index`에 주목해보자. 먼저 `dim`은 사용자가 인덱싱을 적용하고 싶은 차원을 지정해주는 역할을 한다. `index` 매개변수로 전달하는 텐서 안에는 원소의 `‘인덱스’`를 의미하는 숫자들이 마구잡이로 담겨있는데, 해당 인덱스가 대상 텐서의 어느 차원을 가리킬 것인지를 컴퓨터에게 알려준다고 생각하면 된다. `index` 는 앞에서 설명했듯이 원소의 `‘인덱스’`를 의미하는 숫자들이 담긴 텐서를 입력으로 하는 매개변수다. 이 때 주의할 점은 대상 텐서(`input`)와 인덱스 텐서의 차원 형태가 반드시 동일해야 한다는 것이다. 역시 말로만 들으면 이해하기 힘드니 사용 예시를 함꼐 살펴보자.

```python
# torch.gather usage example
>>> q, kr = torch.randn(10, 1024, 64), torch.randn(10, 1024, 64) # [batch, sequence, dim_head], [batch, 2*sequence, dim_head]
>>> tmp_c2p = torch.matmul(q, kr.transpose(-1, -2))
>>> tmp_c2p, tmp_c2p.shape
(tensor([[-2.6477, -4.7478, -5.3250,  ...,  1.6062, -1.9717,  3.8004],
         [ 0.0662,  1.5240,  0.1182,  ...,  0.1653,  2.8476,  1.6337],
         [-0.5010, -4.2267, -1.1179,  ...,  1.1447,  1.7845, -0.1493],
         ...,
         [-2.1073, -1.2149, -4.8630,  ...,  0.8238, -0.5833, -1.2066],
         [ 2.1747,  3.2924,  6.5808,  ..., -0.2926, -0.2511,  2.6996],
         [-2.8362,  2.8700, -0.9729,  ..., -4.9913, -0.3616, -0.1708]],
        grad_fn=<MmBackward0>)
torch.Size([1024, 1024]))

>>> max_seq, max_relative_position = 1024, 512
>>> q_index, k_index = torch.arange(max_seq), torch.arange(2*max_relative_position)
>>> q_index, k_index
(tensor([   0,    1,    2,  ..., 1021, 1022, 1023]),
 tensor([   0,    1,    2,  ..., 1021, 1022, 1023]))

>>> tmp_pos = q_index.view(-1, 1) - k_index.view(1, -1)
>>> rel_pos_matrix = tmp_pos + max_relative_position
>>> rel_pos_matrix
tensor([[ 512,  511,  510,  ..., -509, -510, -511],
        [ 513,  512,  511,  ..., -508, -509, -510],
        [ 514,  513,  512,  ..., -507, -508, -509],
        ...,
        [1533, 1532, 1531,  ...,  512,  511,  510],
        [1534, 1533, 1532,  ...,  513,  512,  511],
        [1535, 1534, 1533,  ...,  514,  513,  512]])

>>> rel_pos_matrix = torch.clamp(rel_pos_matrix, 0, 2*max_relative_position - 1).repeat(10, 1, 1)
>>> tmp_c2p = tmp_c2p.repeat(10, 1, 1)
>>> rel_pos_matrix, rel_pos_matrix.shape, tmp_c2p.shape 
(tensor([[[ 512,  511,  510,  ...,    0,    0,    0],
          [ 513,  512,  511,  ...,    0,    0,    0],
          [ 514,  513,  512,  ...,    0,    0,    0],
          ...,
          [1023, 1023, 1023,  ...,  512,  511,  510],
          [1023, 1023, 1023,  ...,  513,  512,  511],
          [1023, 1023, 1023,  ...,  514,  513,  512]],
 
         [[ 512,  511,  510,  ...,    0,    0,    0],
          [ 513,  512,  511,  ...,    0,    0,    0],
          [ 514,  513,  512,  ...,    0,    0,    0],
          ...,
          [1023, 1023, 1023,  ...,  512,  511,  510],
          [1023, 1023, 1023,  ...,  513,  512,  511],
          [1023, 1023, 1023,  ...,  514,  513,  512]],
torch.Size([10, 1024, 1024]),
torch.Size([10, 1024, 1024]))

>>> torch.gather(tmp_c2p, dim=-1, index=rel_pos_matrix)
tensor([[[-0.8579, -0.2178,  1.6323,  ..., -2.6477, -2.6477, -2.6477],
         [ 1.1601,  2.1752,  0.7187,  ...,  0.0662,  0.0662,  0.0662],
         [ 3.4379, -1.2573,  0.1375,  ..., -0.5010, -0.5010, -0.5010],
         ...,
         [-1.2066, -1.2066, -1.2066,  ...,  0.5943, -0.5169, -3.0820],
         [ 2.6996,  2.6996,  2.6996,  ...,  0.2014,  1.1458,  3.2626],
         [-0.1708, -0.1708, -0.1708,  ...,  1.9955,  4.1549,  2.6356]],
```

위 코드는 `DeBERTa` 의 `Disentangled Self-Attention`을 구현한 코드의 일부분이다. 자세한 원리는 `DeBERTa` 논문 리뷰 포스팅에서 확인하면 되고, 우리가 지금 주목할 부분은 바로 `tmp_c2p`, `rel_pos_matrix` 그리고 마지막 줄에 위치한 `torch.gather` 다. `[10, 1024, 1024]` 모양을 가진 대상 텐서 `tmp_c2p` 에서 내가 원하는 원소만 추출하려는 상황인데, 추출해야할 원소의 인덱스 값이 담긴 텐서를 `rel_pos_matrix` 로 정의했다. `rel_pos_matrix` 의 차원은 `[10, 1024, 1024]`로 `tmp_c2p`와 동일하다. 참고로 추출해야 하는 차원 방향은 가로 방향(두 번째 1024)이다.

이제 `torch.gather`의 동작을 살펴보자. 우리가 현재 추출하고 싶은 대상은 3차원 텐서의 가로 방향(두 번째 1024, 텐서의 행 벡터), 즉 `2 * max_sequence_length` 를 의미하는 차원 방향의 원소다. 따라서 `dim=-1`으로 설정해준다. 이제 메서드가 의도대로 적용되었는지 확인해보자. `rel_pos_matrix` 의 0번 배치, 0번째 시퀀스의 가장 마지막 차원의 값은 `0`으로 초기화 되어 있다. 다시 말해, 대상 텐서의 대상 차원에서 0번째 인덱스에 해당하는 값을 가져오라는 의미를 담고 있다. 그렇다면 `torch.gather` 실행 결과가 `tmp_c2p`의 0번 배치, 0번째 시퀀스의 0번째 차원 값과 일치하는지 확인해보자. 둘 다 `-2.6477`, `-2.6477` 으로 같은 값을 나타내고 있다. 따라서 우리 의도대로 잘 실행되었다는 사실을 알 수 있다. 

### `👩‍👩‍👧‍👦 torch.triu, torch.tril`

각각 입력 텐서를 `상삼각행렬`, `하삼각행렬`로 만든다. `triu`나 `tril`은 사실 뒤집으면 같은 결과를 반환하기 때문에 `tril`을 기준으로 설명을 하겠다. 메서드의 매개변수는 다음과 같다.

```python
# torch.triu, tril params
upper_tri_matrix = torch.triu(input_tensor, diagonal=0, *, out=None)
lower_tri_matrix = torch.tril(input_tensors, diagonal=0, *, out=None)
```

`diagonal` 에 주목해보자. 양수를 전달하면 주대각성분에서 해당하는 값만큼 떨어진 곳의 대각성분까지 그 값을 살려둔다. 한편 음수를 전달하면 주대각성분을 포함해 주어진 값만큼 떨어진 곳까지의 대각성분을 모두 0으로 만들어버린다. 기본은 0으로 설정되어 있으며, 이는 주대각성분부터 왼쪽 하단의 원소를 모두 살려두겠다는 의미가 된다. 

```python
# torch.tril usage example
>>> lm_mask = torch.tril(torch.ones(x.shape[0], x.shape[-1], x.shape[-1]))
>>> lm_mask
1 0 0 0 0
1 1 0 0 0
1 1 1 0 0
1 1 1 1 0
```

두 메서드는 선형대수학이 필요한 다양한 분야에서 사용되는데, 필자의 경우, `GPT`처럼 `Transformer`의 `Decoder` 를 사용하는 모델을 빌드할 때 가장 많이 사용했던 것 같다. `Decoder`를 사용하는 모델은 대부분 구조상 `Language Modeling`을 위해서 `Masked Multi-Head Self-Attention Block`을 사용하는데 이 때 미래 시점의 토큰 임베딩 값에 마스킹을 해주기 위해 `torch.tril` 을 사용하게 되니 참고하자. 

### `👩‍👩‍👧‍👦 torch.Tensor.masked_fill`

사용자가 지정한 값에 해당되는 원소를 모두 마스킹 처리해주는 메서드다. 먼저 매개변수를 확인해보자.

```python
# torch.Tensor.masked_fill params
input_tensors = torch.Tensor([[1,2,3], [4,5,6]])
input_tensors.masked_fill(mask: BoolTensor, value: float)
```

`masked_fill` 은 텐서 객체의 내부 `attribute` 로 정의되기 때문에 해당 메서드를 사용하고 싶다면 먼저 마스킹 대상 텐서를 만들어야 한다. 텐서를 정의했다면 텐서 객체의 `attributes` 접근을 통해 `masked_fill()` 을 호출한 뒤, 필요한 매개변수를 전달해주는 방식으로 사용하면 된다. 

`mask` 매개변수에는 마스킹 텐서를 전달해야 하는데, 이 때 내부 원소는 모두 `boolean`이어야 하고 텐서의 형태는 대상 텐서와 동일해야 한다(완전히 같을 필요는 없고, 브로드 캐스팅만 가능하면 상관 없음).

`value` 매개변수에는 마스킹 대상 원소들에 일괄적으로 적용해주고 싶은 값을 전달한다. 이게 말로만 들으면 이해하기 쉽지 않다. 아래 사용 예시를 함께 첨부했으니 참고 바란다.

```python
# torch.masked_fill usage

>>> lm_mask = torch.tril(torch.ones(x.shape[0], x.shape[-1], x.shape[-1]))
>>> lm_mask
1 0 0 0 0
1 1 0 0 0
1 1 1 0 0
1 1 1 1 0
>>> attention_matrix = torch.matmul(q, k.transpose(-1, -2)) / dot_scale
>>> attention_matrix
1.22 2.1 3.4 1.2 1.1
1.22 2.1 3.4 9.9 9.9
1.22 2.1 3.4 9.9 9.9
1.22 2.1 3.4 9.9 9.9

>>> attention_matrix = attention_matrix.masked_fill(lm_mask == 0, float('-inf'))
>>> attention_matrix
1.22 -inf -inf -inf -inf
1.22 2.1 -inf -inf -inf
1.22 2.1 3.4 -inf -inf
1.22 2.1 3.4 9.9 -inf
```

### `🗂️ torch.clone`

`inputs` 인자로 전달한 텐서를 복사하는 파이토치 내장 메서드다.  사용법은 아래와 같다.

```python
""" torch.clone """
torch.clone(
    input, 
    *,
    memory_format=torch.preserve_format
) → [Tensor]
```

딥러닝 파이프라인을 만들다 보면 많이 사용하게 되는 기본적인 메서드인데, 이렇게 따로 정리하게 된 이유가 있다. 입력된 텐서를 그대로 복사한다는 특성 때문에 사용시 주의해야 할 점이 있기 때문이다. 해당 메서드를 사용하기 전에 반드시 입력할 텐서가 현재 어느 디바이스(CPU, GPU) 위에 있는지, 그리고 해당 텐서가 계산 그래프를 가지고 있는지를 **반드시** 파악해야 한다.

필자는 ELECTRA 모델을 직접 구현하는 과정에서 `clone()` 메서드를 사용했는데, Generator 모델의 결과 로짓을 Discriminator의 입력으로 변환해주기 위함이었다. 그 과정에서 Generator가 반환한 로짓을 그대로 `clone`한 뒤, 입력을 만들어 주었고 그 결과 아래와 같은 에러를 마주했다.

```python
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [
torch.cuda.LongTensor [8, 511]] is at version 1; expected version 0 
instead. Hint: the backtrace further above shows the operation that failed to compute its gradient. 
The variable in question was changed in there or anywhere later. Good luck!
```

에러 로그를 자세히 읽어보면 텐서 버전의 변경으로 인해 그라디언트 계산이 불가하다는 내용이 담겨있다. 구글링해봐도 잘 안나와서 포기하려던 찰라에 우연히 `torch.clone()` 메서드의 정확한 사용법이 궁금해 공식 Docs를 읽게 되었고, 거기서 엄청난 사실을 발견했다. `clone()` 메서드가 입력된 텐서의 현재 디바이스 위치에 똑같이 복사될 것이란 예상은 했지만, 입력 텐서의 계산그래프까지 복사될 것이란 생각은 전혀 하지 못했기 때문이다. 그래서 위와 같은 에러를 마주하지 않으려면, `clone()`을 호출할 때 뒤에 반드시 `detach()`를 함께 호출해줘야 한다.

`clone()` 메서드는 입력된 텐서의 모든 것을 복사한다는 점을 반드시 기억하자.
