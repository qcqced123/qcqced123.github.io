---
title: "🚚 RuntimeError: stack expects each tensor to be equal size, but got [32] at entry 0 and [24] at entry 1"
excerpt: "Pytorch Error: Dataloader get non-equal size of tensor"
permalink: "/cs-ai/framework-library/dataloader-collatefn"
toc: true  # option for table of content
toc_sticky: true  # option for table of content
categories:
  - Pytorch Error Handling
tags:
  - Pytorch
  - DataLoader
  - collate_fn
  - Dynamic Padding
  - Padding
last_modified_at: 2023-07-11T23:00:00-05:00
---

### `📏 가변 길이의 텐서를 데이터로더에 전달하는 경우 `

커스텀 데이터 클래스와 데이터로더를 통해 반환되는 데이터 인스턴스의 텐서 크기가 일정하지 않아 발생하는 에러다. 특히 자연어 처리에서 자주 찾아 볼 수 있는데 데이터로더 객체 선언 시, 매개변수 옵션 중에 `collate_fn=collate` 를 추가해주면 해결 가능한 에러다. 이 때 매개변수 `collate_fn` 에 전달하는 값(메서드)은 사용자가 직접 정의해줘야 한다. 허깅페이스 라이브리러에 상황에 맞게 미리 제작된 `collate` 메서드를 지원해주고 있기 때문에 잘 이용하면 된다. 필자의 경우에는 커스텀으로 직접 정의한 메서드, 객체를 사용하고 있다. 

```python
# 데이터 로더 예시
loader_train = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            collate_fn=MiniBatchCollate,  # 여기에 사용하려는 collate function 혹은 객체를 전달하자!!
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
    This Function should be used after DataLoader return mini-batch instance
    Args:
        inputs: list of dict, dict has keys of "input_ids", "attention_mask", "token_type_ids"    
    """
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:, :mask_len]
    return inputs
```

일반적으로 `collate` 는 메서드로 구현해서 사용하지만, 위 코드처럼 객체로 구현하고 내부에 `__call__` 를 정의해 사용하는 방법도 있다. 필자 역시 단일 메서드 형태를 계속해서 사용하다가 최근 들어 에폭 한 번에 서로 다른 데이터 세트 및 모델을 훈련 시켜야 하는 상황을 마주한 이후 객체 형태로 다시 구현해 사용하고 있다. 

한편 예시 코드 가장 마지막 `collate` 메서드는 입력 시퀀스가 huggingface의 `AutoTokenizer.encode_plus` 를 이용해 사용자 지정 `max_len`까지 패딩을 마친 상태라는 가정하에 구현 되었다. 해당 메서드는 위에 발생한 에러를 해결하기 위함보다, 미니 배치에 속한 전체 데이터 중에서 최대 길이가 사용자 지정 `max_len`까지 미치지 못하는데 패딩이 된 경우에 사용하기 위해 만들었다. 불필요한 패딩을 `trucation` 하여 뉴럴 네트워크의 학습 속도를 높이기 위함이다. 해당 메서드는 포스팅의 제목에 달린 에러를 해결하는데 사용할 수는 없지만  `collate` 기능을 언급하는 김에 생각이나 같이 정리해봤다. 이 메서드는 `torch.utils.data.DataLoader` 의 인자가 아니라, 메인 학습 루프 내부에 사용한다. 다시 말해, 데이터로더가 배치 인스턴스를 반환한 다음 사용하면 된다는 것이다. 패딩방식과 `collate` 기능에 대한 자세한 설명은 다른 포스팅에서 다루도록 하겠다.

반면 `MiniBatchCollate` 객체는 `torch.utils.data.DataLoader` 의 `collate_fn` 인자에 전달하면 된다. 필자의 경우는 `Dynamic Padding` 기법을 사용하기 때문에 미니 배치 내부의 인스턴스들이 서로 다른 시퀀스 길이를 갖는 경우가 발생한다. 데이터로더는 미니 배치에 속하는 데이터의 길이가 통일되지 않으면 배치 단위로 데이터를 묶을 수 없게 된다. 따라서 미니 배치 단위의 길이 통일을 위해 `torch.nn.utils.rnn.pad_sequence` 메서드를 사용한다. 이 메서드는 입력한 미니 배치 데이터 중에서 가장 긴 시퀀스를 기준으로 모든 데이터 길이를 통일한다. `batch_first=True` 를 주목하자. 이 인자를 `False` 로 설정할 경우, 배치 차원이 맨 앞이 아니라 중간에 정의된다. 일반적으로는 배치 차원을 맨 앞에 두는 워크플로우를 사용하기 때문에 꼭 해당 인자를 `True` 로 설정하고 사용하자. 
