---
title: "🤔 RuntimeError: Function 'LogSoftmaxBackward0' returned nan values in its 0th output"
excerpt: "Pytorch Error: Backward NaN values"
permalink: "/cs-ai/linear-algebra/inner-product"
toc: true  # option for table of content
toc_sticky: true  # option for table of content
categories:
  - Linear Algebra
tags:
  - Pytorch
  - Inner Product
  - Projection Matrix
  - 내적
  - 정사영
last_modified_at: 2023-07-10T23:00:00-05:00
---

## `RuntimeError: Function 'LogSoftmaxBackward0' returned nan values in its 0th output`

커스텀으로 모델, 여러 풀링, 매트릭, 손실 함수들을 정의하면서부터 제일 많이 마주하게 되는 에러다. 진심으로 요즘 `CUDA OOM` 보다 훨씬 자주 보는 것 같다. 해당 에러는 `LogSoftmax` 레이어에 전달된 입력값 중에서 `nan`, `inf` 가 포함되어 연산을 진행할 수 없다는 것을 의미한다. 딥러닝 실험을 진행하면서 가장 해결하기 까다로운 녀석으로 원인을 특정하기 힘들기 때문이다. 원인을 잡기 어려운 이유는 바로 우리가 지금 하고 있는게 `‘딥러닝’` 이라서 그렇다. 위 에러는 대부분 연산자가 우리가 의도하지 않은 동작을 하는 케이스 때문인데, 하나 하나 디버깅하기에는 너무나도 연산자가 많다. 또한 딥러닝은 입출력으로 엄청나게 큰 사이즈의 행렬을 사용한다. 우리가 `nan`, `inf` 값 존재에 대해서 인지하기 쉽지 않다. 

한편, 위 에러는 필자의 경험상 대부분 커스텀으로 정의한 레이어에서 발생하는 경우가 많았으며 특히 `분수`, `각도`, `제곱근`, `지수` 개념을 사용하ㄷ는 연산자가 대부분 원인이었다. 예를 들어 코사인 유사도를 구하는 과정에서 연산 대상 벡터값에  `zero-value` 가 포함된 경우 분모가 0이 되기 때문에 연산 정의가 되지 않아 `nan` 을 반환해 위와 같은 에러가 발생하는 경우가 있다. 

```python
class CLIPGEMPooling(nn.Module):
    """
    Generalized Mean Pooling for Natural Language Processing
    This class version of GEMPooling for CLIP, Transfer from NLP Task Code
    Mean Pooling <= GEMPooling <= Max Pooling
    Because of doing exponent to each token embeddings, GEMPooling is like as weight to more activation token
    """
    def __init__(self, auto_cfg) -> None:
        super(CLIPGEMPooling, self).__init__()
        self.eps = 1e-6 # defend underflow 

    def forward(self, last_hidden_state, p: int = 4) -> Tensor:
        """
        last_hidden_state.size: [batch_size, patches_sequence, hidden_size]
        1) Pow last_hidden_state with p and then take a averaging 
        2) pow sum_embeddings with 1/p
        """
        sum_embeddings = torch.mean(torch.pow(last_hidden_state, p), 1) + self.eps
        gem_embeddings = torch.pow(sum_embeddings, 1 / p) + self.eps
        return gem_embeddings

class CLIPMultipleNegativeRankingLoss(nn.Module):
    """
    Multiple Negative Ranking Loss for CLIP Model
    main concept is same as original one, but append suitable for other type of model (Not Sentence-Transformers)
    if you set more batch size, you can get more negative pairs for each anchor & positive pair
    Args:
        scale: output of similarity function is multiplied by this value => I don't know why this is needed
        similarity_fct: standard of distance metrics, default cosine similarity
    """
    def __init__(self, reduction: str, scale: float = 20.0, similarity_fct=cos_sim) -> None:
        super().__init__()
        self.reduction = reduction
        self.eps = 1e-6
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.reduction = reduction
        self.cross_entropy_loss = CrossEntropyLoss(self.reduction)

    def forward(self, embeddings_a, embeddings_b):
        """
        Compute similarity between `a` and `b`.
        Labels have the index of the row number at each row, same index means that they are ground truth
        This indicates that `a_i` and `b_j` have high similarity
        when `i==j` and low similarity when `i!=j`.
        Example a[i] should match with b[i]
        """
        similarity_scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale + self.eps

        labels = torch.tensor(
            range(len(similarity_scores)),
            dtype=torch.long,
            device=similarity_scores.device,
        )
        return self.cross_entropy_loss(similarity_scores, labels)
```

필자의 경우, 두 개의 입력 행렬에 각각  `sqrt()` 를 적용하고 두 행렬의 개별 원소 사이의 코사인 유사도를 구해야 했던 적이 있다. `sqrt` 과정에서 너무 작은 값들이 입력으로 들어가 `underflow` 가 발생해 행렬에 `zero-value` 가 생겼고, 이를 모른채 코사인 유사도를 구하다가 한참을 위 에러와 싸웠던 적이 있다. 심지어 연산속도 향상을 위해서 **`torch.autocast`의** `grad_scaler(float32 to float16)` 까지 적용하고 있었다. 

이 글을 읽는 당신이 만약 `sqrt`  혹은 `pow`를 활용하는 경우, `underflow` 방지를 위해서 위 예시 코드처럼 꼭 적당한 입실론 값을 연산 전후에 필요에 따라 더해줄 것을 권장한다. 입실론 값의 설정은 현재 자신이 사용하고 있는 부동 소수점 정확도에 맞게 설정해주면 될 것 같다. `float32` 를 사용하는 경우에는 대부분 `1e-6` 을 많이 사용하는 것 같다. 필자도 정확히 어떤 값이 적당한지 아직 잘 모르겠다… 그리고  딥러닝 실험하면서 `overflow` 때문에 `inf` 이 발생했던 적은 없었다.

한편 `torch.autograd.set_detect_anomaly(*True*)` 를 훈련 루프 초반에 정의해주면, `nan`이 발생하는 즉시 실행이 멈추고 `nan`을 유발한 라인을 출력해준다. 꼭 활용해보자. 

### ****`RuntimeError: Attempting to deserialize object on CUDA device 2 but torch.cuda.device_count() is 1. Please use torch.load with map_location to map your storages to an existing device`****

```python
model.load_state_dict(
		torch.load(
				path,
				map_locat='cuda:0' # 데이터 및 모델을 사용하려는 GPU 번호를 입력값으로 전달
	) 
)
```

`pretrained model`, `weight`를 `load`하거나 혹은 훈련 루프를 `resume` 을 위해 `torch.load()` 를 사용할 때 마주할 수 있는 에러 로그다. 발생하는 이유는 현재 `GPU` 에 할당하려는 모델이 사전 훈련때 할당 되었던 `GPU` 번호와 현재 할당하려는 `GPU` 번호가 서로 상이하기 때문이다. 

### **`RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(hand≤)`**

파이토치 코드(`torch.nn.Embedding`)에서 정의한 입출력 차원과 실제 데이터의 차원이 다른 경우에 발생하는 에러다. 다양한 상황에서 마주할 수 있는 에러지만, 필자의 경우 `Huggingface`에서 불러온`pretrained tokenizer`에 `special token` 을 추가해 사용하는 경우가 많은데, 토큰을 추가했다는 사실을 잊고 `nn.Embedding` 에 정의한 입출력 차원을 변경하지 않아서 발생하는 경우가 많았다. 구글링해보니 해결하는 방법은 다양한 것 같은데, torch.nn.embedding에 정의된 입출력 차원을 실제 데이터 차원과 맞춰주면 간단하게 해결된다. 필자처럼 `special token` 을 추가해 사용하다 해당 에러가 발생하는 상황이라면, 아래 예시 코드를 토큰을 추가한 이후 시점에 선언해주면 해결할 수 있다.

```python
model.resize_token_embeddings(len(tokenizer))
```

### **`RuntimeError: CUDA error: device-side assert triggered`**

다양한 원인이 있다고 알려져 있는 에러지만, 필자의 경우 위 에러는 사전에 정의한 입출력 데이터의 차원과 실제 입출력 데이터 차원이 서로 상이하는 상황에 발생했다. 하지만 원인을 확실히 특정하고 싶다면 아래 예시 코드를 먼저 추가한 뒤, 다시 한 번 에러 로그를 확인해보길 권장한다.

```python
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

예시 코드처럼 환경변수를 추가하면 에러가 어느 부분에서 발생했는지 로그가 좀 더 구체적으로 나온다. 거의 대부분이 입출력 차원 문제일테니 귀찮으면 바로 차원을 수정하도록 하자. 

### `RuntimeError: stack expects each tensor to be equal size, but got [32] at entry 0 and [24] at entry 1`

커스텀 데이터 클래스와 데이터로더를 통해 반환되는 데이터 인스턴스의 텐서 크기가 일정하지 않아 발생하는 에러다. 특히 자연어 처리에서 자주 찾아 볼 수 있는데 데이터로더 객체 선언 시, 매개변수 옵션 중에 `collate_fn=collate` 를 추가해주면 해결 가능한 에러다. 이 때 매개변수 `collate_fn` 에 전달하는 값(메서드)은 사용자가 직접 정의해줘야 한다. 허깅페이스 라이브리러에 상황에 맞게 미리 제작된 `collate` 메서드를 지원해주고 있기 때문에 잘 이용하면 된다. 필자의 경우에는 커스텀으로 직접 정의한 메서드, 객체를 사용하고 있다. 

```python
# 데이터 로더 예시
loader_train = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
						collate_fn=collate,
            generator=self.generator,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )

# collate 메서드 예시: 
class MiniBatchCollate(object):
    """
    Collate class for torch.utils.data.DataLoader
    This class object to use variable data such as NLP text sequence
    If you use static padding with AutoTokenizer, you don't need this class 
    But if you use dynamic padding with AutoTokenizer, you must use this class object & call
    Args:
        batch: data instance from torch.utils.data.DataSet
    """
    def __init__(self, batch: torch.utils.data.DataLoader) -> None:
        self.batch = batch

    def __call__(self) -> tuple[dict[Tensor, Tensor, Tensor], Tensor, Tensor]:
        inputs, labels, position_list = self.batch
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-1
        )
        position_list = torch.nn.utils.rnn.pad_sequence(
            position_list,
            batch_first=True,
            padding_value=-1
        )
        return inputs, labels, position_list

def collate(inputs):
		"""
		slice input sequence by maximum length sequence in mini-batch, used for speed up training
    if you want slice other variable such as label feature, you can add param on them
    Args:
        inputs: list of dict, dict has keys of "input_ids", "attention_mask", "token_type_ids"    
		"""
		mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:, :mask_len]
    return inputs
```

일반적으로 `collate` 는 메서드로 구현해서 사용하지만, 위 코드처럼 객체로 구현하고 내부에 `__call__` 를 정의해 사용하는 방법도 있다. 필자 역시 단일 메서드 형태를 계속해서 사용하다가 최근 들어 에폭 한 번에 서로 다른 데이터 세트 및 모델을 훈련 시켜야 하는 상황을 마주한 이후 객체 형태로 다시 구현해 사용하고 있다. 

한편 예시 코드 가장 마지막 `collate` 메서드는 입력 시퀀스가 huggingface의 `AutoTokenizer.encode_plus` 를 이용해 사용자 지정 `max_len`까지 패딩을 마친 상태라는 가정하에 구현 되었다. 해당 메서드는 위에 발생한 에러를 해결하기 위함보다, 미니 배치에 속한 전체 데이터 중에서 최대 길이가 사용자 지정 `max_len`까지 미치지 못하는데 패딩이 된 경우에 사용하기 위해 만들었다. 불필요한 패딩을 `trucation` 하여 뉴럴 네트워크의 학습 속도를 높이기 위함이다. 해당 메서드는 포스팅의 제목에 달린 에러를 해결하는데 사용할 수는 없지만  `collate` 기능을 언급하는 김에 생각이나 같이 정리해봤다. 이 메서드는 `torch.utils.data.DataLoader` 의 인자가 아니라, 메인 학습 루프 내부에 사용한다. 다시 말해, 데이터로더가 배치 인스턴스를 반환한 다음 사용하면 된다는 것이다. 

반면 `MiniBatchCollate` 객체는 `torch.utils.data.DataLoader` 의 `collate_fn` 인자에 전달하면 된다. 필자의 경우는 `Dynamic Padding` 기법을 사용하기 때문에 미니 배치 내부의 인스턴스들이 서로 다른 시퀀스 길이를 갖는 경우가 발생한다. 데이터로더는 미니 배치에 속하는 데이터의 길이가 통일되지 않으면 배치 단위로 데이터를 묶을 수 없게 된다. 따라서 미니 배치 단위의 길이 통일을 위해 `torch.nn.utils.rnn.pad_sequence` 메서드를 사용한다. 이 메서드는 입력한 미니 배치 데이터 중에서 가장 긴 시퀀스를 기준으로 모든 데이터 길이를 통일한다. `batch_first=True` 를 주목하자. 이 인자를 `False` 로 설정할 경우, 배치 차원이 맨 앞이 아니라 중간에 정의된다. 일반적으로는 배치 차원을 맨 앞에 두는 워크플로우를 사용하기 때문에 꼭 해당 인자를 `True` 로 설정하고 사용하자. 

### `assert len(optimizer_state["found_inf_per_device"]) > 0, "No inf checks were recorded for this optimizer." AssertionError: No inf checks were recorded for this optimizer.`

텐서의 계산 그래프가 중간에 끊어져 옵티마이저가 그라디언트를 제대로 `Backward` 하지 못해 발생하는 에러다. 공부를 시작하고 정말 처음 마주하는 에러라서 정말 많이 당황했다. 래퍼런스 자료 역시 거의 없어서 해결하는데 애를 먹었던  쓰라린 사연이 있는 에러다. 이 글을 읽는 독자라면 대부분 텐서의 계산 그래프가 중간에 끊어진다는 것이 무슨 의미일지 이해하시지 못할거라 생각한다. 그게 정상이다. 필자 역시 알고 싶지 않았으나 욕심만 많고 멍청한 탓에… 알게 되었다. 아래 예시 코드를 먼저 살펴보자.

```python
# Before Append
def forward(self, inputs: dict, position_list: Tensor) -> Tensor:
        outputs = self.feature(inputs)
        feature = outputs.last_hidden_state
				pred = []
        for i in range(self.cfg.batch_size):
            """ Apply Pooling & Fully Connected Layer for each unique cell in batch (one notebook_id) """
            for idx in range(len(position_list[i])):
                src, end = position_list[i][idx]
                embedding = self.pooling(feature[i, src:end + 1, :].unsqueeze(dim=0))  # maybe don't need mask
                logit = self.fc(embedding)
                pred.append(logit)
						pred = torch.as_tensor(pred, device=self.cfg.device)
        return pred

# After Append
def forward(self, inputs: dict, position_list: Tensor) -> Tensor:
        outputs = self.feature(inputs)
        feature = outputs.last_hidden_state
        pred = torch.tensor([], device=self.cfg.device)
        for i in range(self.cfg.batch_size):
            """ Apply Pooling & Fully Connected Layer for each unique cell in batch (one notebook_id) """
            for idx in range(len(position_list[i])):
                src, end = position_list[i][idx]
                embedding = self.pooling(feature[i, src:end + 1, :].unsqueeze(dim=0))  # maybe don't need mask
                logit = self.fc(embedding)
                pred = torch.cat([pred, logit], dim=0)
        return pred
```

다음 코드는 필자가 공부를 위해 만든 모델 클래스의 `forward` 메서드이다. 전자는 이번 포스팅의 주제인 에러를 일으킨 주인공이고, 후자는 에러를 수정한 이후 정상적으로 작동하는 코드다. 독자 여러분들도 두 코드에 어떤 차이가 있는지 스스로 질문을 던지면서 읽어주시길 바란다. 

위의 코드들은 `DeBERTa-V3-Large` 의 마지막 인코더 레이어가 반환하는 `last_hidden_state` 를 미리 설정한 서브 구간별로 나누고 개별적으로 `pooling & fully connected layer` 에 통과시켜 로짓값으로 변환하기 위해 만들었다. 쉽게 말해 입력으로 토큰(단어) 384개 짜리 문장을 하나 넣었고, 모델은 384개의 개별 토큰에 대한 임베딩 값을 반환했는데 그것을 전부 이용하는 것이 아니라 예를 들어 `2번~4번` 토큰을 1번 구간, `6번~20번` 토큰을 2번 구간, `30번~50번` 토큰을 3번 구간 … `370번~380번` 토큰을 30번 구간으로 설정하고 구간 별로 따로 `pooling & fully connected layer` 에 통과시켜 로짓을 구하는 것이다. 일반적이라면 1개의 문장에서 1개의 최종 로짓값이 도출되는 것이라면, 위 코드는 30개의 로짓값이 도출된다. 

코드 이해를 위한 설명은 마쳤으니 본격적으로 본 에러와 어떤 연관이 있는지 살펴보자. `Before` 코드는 `pred` 라는 리스트에 개별 구간에 대한 로짓값을 `append` 하고 마지막에 `torch.as_tensor`를 활용해 텐서로 변환하고 있다. 한편 후자는 `pred` 를 깡통 텐서로 선언한 뒤, `torch.cat`으로 모든 구간에 대한 로짓값을 하나의 텐서 구조체에 담고 있다. 

얼핏보면 크게 다른점이 없어 보인다. 하지만 전자는 텐서 구조체를 새로 정의 하면서 `torch.Tensor[[logit1], [logit2], ….]` 형태를 갖고 후자는 `torch.Tensor[logit1, logit2, …]` 형태를 갖는다. 서로 다른 텐서 구조체를 그대로 모델 객체의 `forward` 메서드 및 `loss function`에 통과시키고 오차 역전을 하면 어떤 일이 생기는지 지금부터 알아보자.

전자의 경우는 도출된 손실함수의 미분값이 정의된 계산 그래프를 타고 역전될 수 없다. 이유는 전자의 `pred` 가 forward 메서드 내부에서 새로이 정의 되었기 때문이다. 후자 역시 마찬가지 아닌가 싶을 것이다. 후자의 `pred` 역시 `forward` 메서드 내부에서 정의된 것은 맞지만 `[torch.cat](http://torch.cat)`을 사용하면서 구간의 로짓값들 위에 새로이 차원을 덮어쓰는것이 아니게 된다. 이것이 매우 중요한 차이가 되는데, 후자와 같은 형태가 되는 경우, 손실값으로 부터 `Backward` 되는 미분값들이 곧바로 `forward` 과정에서 기록된 자신의 계산 그래프로 찾아 갈 수 있다. 한편 전자의 경우 새롭게 덮어 쓰여진 차원 때문에 미분값들이 자신의 계산 그래프로 찾아갈 수 없게 된다. 따라서 옵티마이저가 더 이상 `Backward` 를 수행할 수 없어 제목과 같은 에러를 반환하게 되는 것이다. 

처음 이 에러를 마주했을 때는  `found_inf_per_device`, `No inf checks` 라는 키워드에 꽂혀 (특히 `inf`)  <`RuntimeError: Function 'LogSoftmaxBackward0' returned nan values in its 0th output` > 이것과 유사한 종류의 에러라 생각하고 열심히 연산 과정에 문제가 없는지, 어디서 NaN이 발생하는지, 학습률을 너무 크게 설정했는지 등을 검토하며 하루를 날렸었던 기억이 있다.

### `RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.`

필자가 `torch.nn.MarginRankingLoss` 를 사용해 훈련 파이프라인을 정의할 때 마주했던 에러다. 이미 `optimizer.backward()`를 거친 텐서 객체를 또 다시 `backward pass` 하게 되면 발생하는데, 해당 텐서 객체에 `retain_graph=True` 매개변수를 설정해주면 해결할 수 있다. 이미 backward한 텐서를 다시 backward pass에 호출하면 이런 문제가 발생하게 되는 것일까? `pytorch` 는 `backward` 를 수행하면서 그라디언트 계산을 끝마친 노드는 메모리 사용의 효율성을 위해 할당 취소를 한다. 다시 말해, 그라디언트 계산이 완료되면 해당 노드를 시작으로 이전의 모든 계산 그래프 노드에 접근이 불가능해지는 것이다. 따라서 이미 `backward` 호출에 의해 그라디언트가 계산 되었던 노드(텐서 객체)를 또 다시 호출하면 저장된 계산 그래프가 이미 모두 할당 취소 되어 `backward` 호출을 더 이상 수행할 수 없게 되는 것이다. 따라서 같은 노드(텐서 객체)를 여러번 호출해야 하는 경우라면 계산 그래프가 할당 취소 되지 않도록 `retain_graph=True` 를 설정해줘야 하는 것이다. 이렇게 매개변수를 설정해주는 방법도 있고 임시용 변수를 만들어 저장하는 방법도 있다. 두 방식이 더 효율적일지는 실험을 해봐야 알 것 같다. 

`MarginRankingLoss` 는 Pairwise하게 순위를 비교해 손실값을 도출한다. 따라서 총 3가지 입력을 요구하는데, 순위 비교 대상인 2가지 데이터 인스턴스 그리고 라벨값이다. Pairwise하게 계산을 하기 때문에 같은 데이터 인스턴스가 `MarginRankingLoss` 객체에 의해 여러 번 호출 당하게 된다. 따라서 `retain_graph=True`  혹은 중간 변수에 저장해주지 않으면 위 에러를 마주하게 되는 것이다.