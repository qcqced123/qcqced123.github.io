---
title: "👨‍💻🐍 [Python] Function Argument"
excerpt: "함수 인자 전달 메커니즘에 대한 이해"
permalink: "/python/func_argu"
toc: true  # option for table of contents
toc_sticky: true  # option for table of content
categories:
  - Python
tags:
  - Python
  - Function
  - Argument
  - mutable
  - CS
  
last_modified_at: 2023-10-14T12:00:00-05:00
---

### **`👨‍👩‍👧‍👦 Function Argument`**

파이썬의 모든 메서드는 기본적으로 인자를 `call by value` 형태로 전달해야 한다. 하지만 `call by value` 라고 해서 함수의 동작과 원본 변수가 완전히 독립적인 것은 아니다. 이것은 인자로 어떤 데이터 타입을 전달하는가에 따라 달라진다. 만약 인자로 `mutable(dynamic)` 객체인 리스트 변수를 전달했다면, 함수의 동작에 따른 결과가 그대로 변수에 반영된다. `mutable` 객체는 함수 스코프 내부 지역 변수에 카피 되지 않기 때문에 이런 현상이 발생한다. 따라서 인자로 리스트와 같은 `mutable` 객체를 전달하는 것은 수많은 부작용을 불러오기 때문에 지양하는게 좋다. 아래 예시 코드를 보자.

```python
""" function argument experiment with mutable vs immutable """
def function(args):
    args += 'in runtime'
    print(args)

# immutable object string
>>> immutable = 'immutable'
>>> function(immutable)
immutable in runtime 

>>> print(mmutable)
immutable

# mutable object list
>>> mutable = list('mutable')
>>> function(mutable)
['m', 'u', 't', 'a', 'b', 'l', 'e', ' ', 'i', 'n', ' ', 'r', 'u', 'n', 't', 'i', 'm', 'e']
>>> print(mutable)
['m', 'u', 't', 'a', 'b', 'l', 'e', ' ', 'i', 'n', ' ', 'r', 'u', 'n', 't', 'i', 'm', 'e']
```

동일한 함수에 동일하게 인자로 사용했지만, `immutable` 객체인 문자열은 함수의 실행 결과와 전혀 무관한 모습이다. 하지만 `mutable` 객체인 리스트는 함수의 실행 결과가 스코프 바깥의 원본 변수에 그대로 반영되었음을 확인할 수 있다. 이것은 리스트가 기본적으로 컨테이너 내부에 실제 값이 아닌 주소 값을 갖고 있고, `Resize()` 연산을 통해 수정이 가능하기 때문이다. 한편, `immutable` 객체의 경우 원본 수정이 불가하기 때문에 관련 연산 혹은 명령을 만나면 `deepcopy` 가 발생한다. 

### **`📦 Packing/Unpacking`**