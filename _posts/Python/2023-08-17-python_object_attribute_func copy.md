---
title: "👨‍💻🐍 [Python] Object Attribute & Assertion Function"
excerpt: "getattr, setattr, delattr, hasattr, Assertion 사용방법"
permalink: "/python/attribute_function"
toc: true  # option for table of contents
toc_sticky: true  # option for table of content
categories:
  - Python
tags:
  - Python
  - Object
  - Attribute
  - Assertion
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

### `⚠️ Assertion`

```python
assert 조건, 메세지 
```

조건이 True이면 아무런 일이 일어나지 않는다. 하지만 조건이 False이면 AssertionError가 발생하고 지정한 메세지가 출력된다. 메세지를 지정하지 않았다면 `AssertionError`가 동일하게 발생하지만 구체적인 에러 명시란은 비워진 채로 로그가 출력된다. 

`assert`는 코드의 오류를 찾는 데 유용하다. 또한 코드의 의도를 명확하게 표현하는 데에도 유용하다. 예를 들어, 변수의 값이 특정 조건을 만족해야 한다는 것을 `assert`를 사용해 표현할 수 있다.

`assert`는 에러 로그를 반환하면서 개발자가 프로그램을 만드는 과정에 관여한다. 원하는 조건의 변수 값을 보증받을 때까지 `assert`로 테스트 할 수 있다. 이는 데이터 유효성 검사처럼 단순히 에러를 찾는것이 아니라 값을 보증하기 위해 사용된다. 예를 들어 함수의 입력 값이 어떤 조건의 참임을 보증하기 위해 사용할 수 있고 함수의 반환 값이 어떤 조건에 만족하도록 만들 수 있다. 혹은 변수 값이 변하는 과정에서 특정 부분은 반드시 어떤 영역에 속하는 것을 보증하기 위해 가정 설정문을 통해 확인 할 수도 있다. `assert`는 실수를 가정해 값을 보증하는 방식으로 코딩 하기 때문에 `'방어적 프로그래밍'`에 속한다. 방어적 프로그래밍에 대한 자세한 내용은 다음 포스트에서 살펴보도록 하자. 

```python
# Python assert 데이터 유효성 검사 예시
class DeBERTa(nn.Module):
    def __init__(self,):
    ...중략...

    def forward(self, inputs: Tensor, mask: Tensor):
        assert inputs.ndim == 3, f'Expected (batch, sequence, vocab_size) got {inputs.shape}'
    ...중략...
```

위의 코드는 필자가 논문을 보고 따라 구현한 `DeBERTa` 모델 최상위 객체의 코드 일부분이다. 최상위 객체는 모델의 입력 임베딩 층과 위치 임베딩 층을 정의해줘야 하기 때문에 반드시 입력값을 미리 정해진 차원 형식에 맞게 객체의 매개 변수로 넘겨줘야 한다. 지정 형식에서 벗어난 텐서는 입력으로 사용될 수 없게 만들기 위해 객체의 `forward` 메서드 시작부분에 `assert` 함수를 두어 데이터 유효성 검사를 하도록 구현했다. 지정된 차원 형태에 맞지 않는 데이터를 입력하게 되면 `AssertionError`와 함께 필자가 지정한 에러 메세지를 반환 받게 될 것이다. 

한편 `AssertionError`는 프로그래머가 의도에 맞지 않는 메서드 혹은 객체 사용을 막기 위해 선제적으로 대응한 것이라고 볼 수 있다. 이는 프로그래머가 만든 규칙에 해당할 뿐, 실제 파이썬이나 컴퓨터 내부 동작 문법에 틀렸다는 것을 의미하는 것은 아니다.