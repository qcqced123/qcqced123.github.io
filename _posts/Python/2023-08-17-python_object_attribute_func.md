---
title: "👨‍💻🐍 [Python] Object Attribute Function"
excerpt: "getattr, setattr, delattr, hasattr 사용방법"
permalink: "/python/attribute_function"
toc: true  # option for table of contents
toc_sticky: true  # option for table of content
categories:
  - Python
tags:
  - Python
  - Object
  - Attribute
  - ML
  - Deep Learning
  
last_modified_at: 2023-08-17T12:00:00-05:00
---

### `🧧 Attribute Function`

이번 포스팅은 `Python` 코드를 작성하면서 객체와 내부 메서드에 관련한 처리가 필요할 때 가장 많이 사용하게 되는 `getattr`, `setattr` , `delattr` , `hasttr` 함수들의 사용법에 대해 다뤄보려 한다. 특히 `getattr`, `setattr` 의 경우 머신러닝 혹은 딥러닝 관련 코드를 읽다가 심심치 않게 찾아볼 수 있다. 모델의 `hyper-parameter`를 튜닝하거나 기타 실험을 할 때 정의한 객체의 변수 혹은 메서드에 쉽고 간결하게 접근하기 위해 사용되고 있기 때문이다.

#### **`📌 getattr`**

```python
""" getattr(object, attribute_name, default) """

class CFG:
    """--------[Common]--------"""
    wandb, train, competition, seed, cfg_name = True, True, 'UPPPM', 42, 'CFG'
    device, gpu_id = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 0
    num_workers = 0
    """ Mixed Precision, Gradient Check Point """
    amp_scaler = True
    gradient_checkpoint = True # save parameter
    output_dir = './output/'
    """ Clipping Grad Norm, Gradient Accumulation """
    clipping_grad = True # clip_grad_norm
    n_gradient_accumulation_steps = 1 # Gradient Accumulation
    max_grad_norm = n_gradient_accumulation_steps * 1000
    """ Model """
    model_name = 'microsoft/deberta-v3-large'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
#    pooling = 'attention'
    max_len = 512
    """ CV, Epoch, Batch Size """
    n_folds = 4
    epochs = 180
    batch_size = 64
```

위의 객체는 실제 제가 캐글 대회를 준비하면서 사용했던 [`config.py`](http://config.py) 를 가져왔다. 

`getattr(object: object, attribute_name: str, default: Any)` 함수는 사용자가 지정한 객체에 매개변수로 전달한 `attribute`가 존재하는지 여부를 판단하고, 존재한다면 해당 `attribute`의 `value`를 반환한다. 한편 존재하지 않으면 `default`로 세팅한 값을 반환한다.

```python
getattr(CFG, 'epochs', "This Attribute doesn't find")
getattr(CFG, 'MPL', "This Attribute doesn't find")
--------------- Result --------------- 
180
This Attribute doesn't find
```

`if-else` 구문보다 훨씬 간결하게 객체의 메서드에 접근하는 것이 가능해졌으며, `default` 값을 매개변수로 전달 받기 때문에 클라이언트가 지정한 `attribute` 가 객체 내부에 없어도 `AttributeError` 를 발생시키지 않아 예외 처리를 별도로 지정할 필요가 사라져 코드 가독성 및 유지보수에 용이하다는 장점이 있다.

```python
class Exmple:
    def __init__(self):
        self.test1 = 0
        self.test2 = 0
    def A(self):
        print("A")  
    def B(self):
        print("B")  
    def C(self):
        print("C")

if __name__ == '__main__':
    exmple = Exmple()
    class_list = ['A','B','C']

    for c in class_list:
        getattr(exmple, c)()
```

한편 `getattr()` 뒤에 괄호를 하나 더 붙여서 사용하기도(머신러닝, 딥러닝 훈련 루프 코드에 종종 보임) 하는데,  해당 괄호는 지정 `attribute` 의 호출에 필요한 매개변수를 전달하기 위한 용도로 쓰인다. 이번 예시의 객체 내부 메서드들은 호출에 필요한 매개변수가 정의되어 있지 않기 때문에 괄호 안을 비워뒀다.

#### **`✂️ setattr`**

```python
""" setattr(object, attribute_name, value) """

class CFG:
    """--------[Common]--------"""
    wandb, train, competition, seed, cfg_name = True, True, 'UPPPM', 42, 'CFG'
    device, gpu_id = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 0
    num_workers = 0
    """ Mixed Precision, Gradient Check Point """
    amp_scaler = True
    gradient_checkpoint = True # save parameter
    output_dir = './output/'
    """ Clipping Grad Norm, Gradient Accumulation """
    clipping_grad = True # clip_grad_norm
    n_gradient_accumulation_steps = 1 # Gradient Accumulation
    max_grad_norm = n_gradient_accumulation_steps * 1000
    """ Model """
    model_name = 'microsoft/deberta-v3-large'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
#    pooling = 'attention'
    max_len = 512
    """ CV, Epoch, Batch Size """
    n_folds = 4
    epochs = 180
    batch_size = 64
```

`setattr(object: object, attribute_name: str, value: Any)` 는 지정 객체의 지정 메서드 혹은 변수에 접근하고 제어하는 용도로 사용하는 함수다. 지정 객체 단위로 접근 가능하기 때문에 모델을 튜닝할 때 정말 많이 사용하게 된다. `setattr()` 를 활용해 상황에 맞는 파라미터를 모델에 주입하고 해당 `config`를 `json` 혹은 `yaml` 형식으로 저장해두면 모델의 버전별 파라미터 값을 효율적으로 관리할 수 있으니 기억해두자.

```python
CFG.wandb
setattr(CFG, 'wandb', False)
CFG.wandb
setattr(CFG, 'wandb', True)
CFG.wandb

--------------- Result --------------- 
True
False
True
```

#### **`📌 hasattr`**

`hasattr(object, attribute_name)` 는 지정 객체에 매개변수로 전달한 `attribute` 가 존재하면 `True`, 없다면 `False` 를 반환한다. 사용법은 `getattr()` 와 매우 유사하기 때문에 생략한다.

#### **`✏️ delattr`**

`delattr(object, attribute_name)` 는 지정 객체에 매개변수로 전달한 `attribute`를 객체 내부에서 삭제하는 역할을 한다. 사용 예시는 아래와 같다.

```python
delattr(CFG, 'epochs')
hasattr(CFG, 'epochs')

--------------- Result --------------- 
False
```

한편, 모듈(ex: config,py, model.py, model_utils.py 등)도 객체로 간주되기 때문에 위에서 살펴본 4가지 function은 모듈 레벨에서도 동일하게 사용할 수 있다.