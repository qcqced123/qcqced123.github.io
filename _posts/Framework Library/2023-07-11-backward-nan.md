---
title: "🤔 RuntimeError: Function 'LogSoftmaxBackward0' returned nan values in its 0th output"
excerpt: "Pytorch Error: Backward NaN values"
permalink: "/cs-ai/framework-library/backward-nan"
toc: true  # option for table of content
toc_sticky: true  # option for table of content
categories:
  - Pytorch Error Handling
tags:
  - Pytorch
  - Logsoftmax
  - NaN
  - Error Handling
last_modified_at: 2023-07-10T23:00:00-05:00
---

### `🔥 Pytorch Backward 과정에서 NaN 발생하는 문제`

커스텀으로 모델, 여러 풀링, 매트릭, 손실 함수들을 정의하면서부터 제일 많이 마주하게 되는 에러다. 진심으로 요즘 `CUDA OOM` 보다 훨씬 자주 보는 것 같다. 해당 에러는 `LogSoftmax` 레이어에 전달된 입력값 중에서 `nan`, `inf` 가 포함되어 연산을 진행할 수 없다는 것을 의미한다. 딥러닝 실험을 진행하면서 가장 해결하기 까다로운 녀석으로 원인을 특정하기 힘들기 때문이다. 원인을 잡기 어려운 이유는 바로 우리가 지금 하고 있는게 `‘딥러닝’` 이라서 그렇다. 위 에러는 대부분 연산자가 우리가 의도하지 않은 동작을 하는 케이스 때문인데, 하나 하나 디버깅하기에는 너무나도 연산자가 많다. 또한 딥러닝은 입출력으로 엄청나게 큰 사이즈의 행렬을 사용한다. 우리가 `nan`, `inf` 값 존재에 대해서 인지하기 쉽지 않다. 

**<U>위 에러는 필자의 경험상 대부분 커스텀으로 정의한 레이어에서 발생하는 경우가 많았으며 특히</U>** `분수`, `각도`, `제곱근`, `지수` **<U>개념을 사용하는 연산자가 대부분 원인이었다.</U>** 예를 들어 코사인 유사도를 구하는 과정에서 연산 대상 벡터값에  `zero-value` 가 포함된 경우 분모가 0이 되기 때문에 연산 정의가 되지 않아 `nan` 을 반환해 위와 같은 에러가 발생하는 경우가 있다. 

```python
def check_nan(x: torch.Tensor) -> bool:
    """ Check if there is NaN in tensor """
    checker = False
    if True in torch.isnan(x):
        checker = True
    return checker

def zero_filtering(x: torch.Tensor) -> torch.Tensor:
    """
    Add eps value for zero embedding, because competition metric is cosine similarity
    Cosine Similarity will be returned NaN, when input value has zero, like as torch.clamp()
    """
    eps = 1e-4
    x[x <= eps] = eps
    return x

def nan_filtering(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Change eps value for NaN Embedding, because competition metric is cosine similarity
    Cosine Similarity will be returned NaN
    """
    return torch.nan_to_num(x, nan=eps)

class CLIPGEMPooling(nn.Module):
    """
    Generalized Mean Pooling for Natural Language Processing
    This class version of GEMPooling for CLIP, Transfer from NLP Task Code
    ViT don't use attention mask, because input image shape will be same

    Mean Pooling <= GEMPooling <= Max Pooling
    Because of doing exponent to each token embeddings, GEMPooling is like as weight to more activation token

    In original paper, they use p=3, but in this class, we use p=4 because torch doesn't support pow calculation
    for negative value tensor, only for non-negative value in odd number exponent
    """
    def __init__(self, auto_cfg: AutoConfig.from_pretrained) -> None:
        super(CLIPGEMPooling, self).__init__()

    @staticmethod
    def forward(last_hidden_state, p: int = 2) -> Tensor:
        """
        last_hidden_state.size: [batch_size, patches_sequence, hidden_size]
        1) Pow last_hidden_state with p and then take a averaging
        2) pow sum_embeddings with 1/p
        """
        p_embeddings = zero_filtering(torch.pow(last_hidden_state, p))
        # Check NaN value in Embedding after applying torch.pow
        if check_nan(p_embeddings):
            p_embeddings = nan_filtering(p_embeddings)
        sum_embeddings = torch.mean(p_embeddings, 1)

        gem_embeddings = zero_filtering(torch.pow(sum_embeddings, 1. / p))
        # Check NaN value in Embedding after applying torch.pow
        if check_nan(gem_embeddings):
            gem_embeddings = nan_filtering(gem_embeddings)
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
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.reduction = reduction
        self.cross_entropy_loss = CrossEntropyLoss(self.reduction)

    def forward(self, embeddings_a: Tensor, embeddings_b: Tensor) -> Tensor:
        similarity_scores = zero_filtering(self.similarity_fct(embeddings_a, embeddings_b)) * self.scale
        if check_nan(similarity_scores):
            """ Check NaN Value in similarity_scores """
            similarity_scores = nan_filtering(similarity_scores)

        labels = torch.tensor(
            range(len(similarity_scores)),
            dtype=torch.long,
            device=similarity_scores.device,
        )
        return self.cross_entropy_loss(similarity_scores, labels)
```

필자의 경우, 두 개의 입력 행렬에 각각  `sqrt()` 를 적용하고 두 행렬의 개별 원소 사이의 코사인 유사도를 구해야 했던 적이 있다. `sqrt` **<U>과정에서 너무 작은 값들이 입력으로 들어가</U>** `underflow` **<U>가 발생해 행렬에</U>** `zero-value` **<U>가 생겼고, 이를 모른채 코사인 유사도를 구하다가 한참을 위 에러와 싸웠던 적이 있다.</U>** 심지어 연산속도 향상을 위해서 **`torch.autocast`** 클래스의 `grad_scaler(float32 to float16)` 까지 적용하고 있었다. 

### `🖍️ 내가 해결한 방법`
이 글을 읽는 당신이 만약 `sqrt` 혹은 `pow`를 활용하는 경우, `underflow` 방지를 위해서 ~~위 예시 코드처럼 꼭 적당한 입실론 값을 연산 전후에 필요에 따라 더해줄 것을 권장한다.~~ 입실론 값의 설정은 현재 자신이 사용하고 있는 부동 소수점 정확도에 맞게 설정해주면 될 것 같다. `float32` 를 사용하는 경우에는 대부분 `1e-6` 을 많이 사용하는 것 같다. 필자도 정확히 어떤 값이 적당한지 아직 잘 모르겠다… 그리고 딥러닝 실험하면서 `overflow` 때문에 `inf` 이 발생했던 적은 없었다.

입실론 값을 문제가 되는 연산 전에 일괄적으로 더할 경우, 아무리 작은 값이라도 연산 종류에 따라서 결과가 크게 왜곡되는 경우가 발생한다. 따라서 연산을 먼저 적용한 뒤 결과에 `NaN`, `Inf`, `Zero`가 발생하는지 체크하고, 발생한 부분에 한해서 입실론 값을 더해주는 커스텀 `function`울 정의해 문제를 해결했다.  
(위의 코드 예제 `check_nan`, `zero_filtering`, `nan_filtering`)


한편 `torch.autograd.set_detect_anomaly(True)` 를 훈련 루프 초반에 정의해주면, `NaN`이 발생하는 즉시 실행이 멈추고 `NaN`을 유발한 라인을 출력해준다. 꼭 활용해보자.

