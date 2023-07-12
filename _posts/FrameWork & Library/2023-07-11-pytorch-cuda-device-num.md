---
title: "🖥️ RuntimeError: Attempting to deserialize object on CUDA device 2 but torch.cuda.device_count() is 1. Please use torch.load with map_location to map your storages to an existing device"
excerpt: "Pytorch Error: Wrong CUDA Device Number"
permalink: "/cs-ai/framework-library/cuda-num"
toc: true  # option for table of content
toc_sticky: true  # option for table of content
categories:
  - Pytorch Error Handling
tags:
  - Pytorch
  - CUDA
  - Error Handling
last_modified_at: 2023-07-10T23:00:00-05:00
---

### `🔢 Pytorch 잘못된 CUDA 장치 번호 사용 문제`

```python
model.load_state_dict(
    torch.load(path, map_location='cuda:0') 
)
```

`pretrained model`, `weight`를 `load`하거나 혹은 훈련 루프를 `resume` 을 위해 `torch.load()` 를 사용할 때 마주할 수 있는 에러 로그다. 발생하는 이유는 현재 `GPU` 에 할당하려는 모델이 사전 훈련때 할당 되었던 `GPU` 번호와 현재 할당하려는 `GPU` 번호가 서로 상이하기 때문이다. 따라서 `torch.load`의 `map_location`인자에 현재 자신이 사용하려는 `GPU` 번호를 입력해주자.