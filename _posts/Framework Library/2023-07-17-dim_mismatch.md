---
title: '🎲 RuntimeError: CUDA error: device-side assert triggered'
excerpt: "Pytorch Error: Mis-match between pre-defined dimension and input dimension"
permalink: "/cs-ai/framework-library/mismatch-dimension"
toc: true  # option for table of content
toc_sticky: true  # option for table of content
categories:
  - Pytorch Error Handling
tags:
  - Pytorch
  - Dimension Mismatch
  - CUDA
  - Error Handling
last_modified_at: 2023-07-17T17:00:00-05:00
---

### `😵 사전에 정의 입출력 차원 ≠ 실제 입출력 차원`

다양한 원인이 있다고 알려져 있는 에러지만, 필자의 경우 위 에러는 사전에 정의한 데이터의 입출력 차원과 실제 입출력 데이터 차원이 서로 상이할 때 발생했다. 하지만 원인을 확실히 특정하고 싶다면 아래 예시 코드를 먼저 추가한 뒤, 다시 한 번 에러 로그를 확인해보길 권장한다.

```python
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```
예시 코드처럼 환경변수를 추가하면 에러가 어느 부분에서 발생했는지 로그가 좀 더 구체적으로 나온다. 거의 대부분이 입출력 차원 문제일테니 귀찮으면 바로 차원을 수정하도록 하자. 