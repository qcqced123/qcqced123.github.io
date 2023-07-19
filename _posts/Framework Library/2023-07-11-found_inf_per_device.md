---
title: '🪢 assert len(optimizer_state["found_inf_per_device"]) > 0, "No inf checks were recorded for this optimizer." AssertionError: No inf checks were recorded for this optimizer.'
excerpt: "Pytorch Error: Optimizer can't backward loss"
permalink: "/framework-library/inf-per-device"
toc: true  # option for table of content
toc_sticky: true  # option for table of content
categories:
  - Framework & Library
tags:
  - Pytorch
  - CUDA
  - Error Handling
last_modified_at: 2023-07-10T23:00:00-05:00
---

### `🤔 Optimizer가 손실값을 제대로 Backward 할 수 없는 문제`

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
<p markdown="1" align="center">
![Model Overview](/assets/images/marginrankingloss.png){: .align-center}{: width="100%", height="50%"}{: .image-caption}
*Modeling Overview*
</p>

위의 코드들은 `DeBERTa-V3-Large` 의 마지막 인코더 레이어가 반환하는 `last_hidden_state` 를 미리 설정한 서브 구간별로 나누고 개별적으로 `pooling & fully connected layer` 에 통과시켜 로짓값으로 변환하기 위해 만들었다. 쉽게 말해 입력으로 토큰(단어) 384개 짜리 문장을 하나 넣었고, 모델은 384개의 개별 토큰에 대한 임베딩 값을 반환했는데 그것을 전부 이용하는 것이 아니라 예를 들어 `2번~4번` 토큰을 1번 구간, `6번~20번` 토큰을 2번 구간, `30번~50번` 토큰을 3번 구간 … `370번~380번` 토큰을 30번 구간으로 설정하고 구간 별로 따로 `pooling & fully connected layer` 에 통과시켜 로짓을 구하는 것이다. 일반적이라면 1개의 문장에서 1개의 최종 로짓값이 도출되는 것이라면, 위 코드는 30개의 로짓값이 도출된다. 

### `🖍️ 내가 해결한 방법`

코드 이해를 위한 설명은 마쳤으니 본격적으로 본 에러와 어떤 연관이 있는지 살펴보자. `Before` 코드는 `pred` 라는 리스트에 개별 구간에 대한 로짓값을 `append` 하고 마지막에 `torch.as_tensor`를 활용해 텐서로 변환하고 있다. 한편 후자는 `pred` 를 깡통 텐서로 선언한 뒤, `torch.cat`으로 모든 구간에 대한 로짓값을 하나의 텐서 구조체에 담고 있다. 

얼핏보면 크게 다른점이 없어 보인다. 하지만 전자는 텐서 구조체를 새로 정의 하면서 `torch.Tensor[[logit1], [logit2], ….]` 형태를 갖고 후자는 `torch.Tensor[logit1, logit2, …]` 형태를 갖는다. 서로 다른 텐서 구조체를 그대로 모델 객체의 `forward` 메서드 및 `loss function`에 통과시키고 오차 역전을 하면 어떤 일이 생기는지 지금부터 알아보자.

전자의 경우는 도출된 손실함수의 미분값이 정의된 계산 그래프를 타고 역전될 수 없다. 이유는 전자의 `pred` 가 forward 메서드 내부에서 새로이 정의 되었기 때문이다. 후자 역시 마찬가지 아닌가 싶을 것이다. 후자의 `pred` 역시 `forward` 메서드 내부에서 정의된 것은 맞지만 `torch.cat`을 사용하면서 구간의 로짓값들 위에 새로이 차원을 덮어쓰는것이 아니게 된다. 이것이 매우 중요한 차이가 되는데, 후자와 같은 형태가 되는 경우, 손실값으로 부터 `Backward` 되는 미분값들이 곧바로 `forward` 과정에서 기록된 자신의 계산 그래프로 찾아 갈 수 있다. 한편 전자의 경우 새롭게 덮어 쓰여진 차원 때문에 미분값들이 자신의 계산 그래프로 찾아갈 수 없게 된다. 따라서 옵티마이저가 더 이상 `Backward` 를 수행할 수 없어 제목과 같은 에러를 반환하게 되는 것이다. 

처음 이 에러를 마주했을 때는  `found_inf_per_device`, `No inf checks` 라는 키워드에 꽂혀 (특히 `inf`)  `<RuntimeError: Function 'LogSoftmaxBackward0' returned nan values in its 0th output>` 이것과 유사한 종류의 에러라 생각하고 열심히 연산 과정에 문제가 없는지, 어디서 NaN이 발생하는지, 학습률을 너무 크게 설정했는지 등을 검토하며 하루를 날렸었던 기억이 있다.