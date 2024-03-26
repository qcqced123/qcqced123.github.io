---
title: "👨⏰🐍 [Python] 시간복잡도 2"
excerpt: "시간 복잡도 줄이기"
permalink: "/python/time_complexity2"
toc: true  # option for table of contents
toc_sticky: true  # option for table of content
categories:
  - Python
tags:
  - Python
  - Time Complexity
  - CS
  
last_modified_at: 2024-03-26T12:00:00-05:00
---

### `Theme 1. 입력 모듈`

```python
""" Compare to Input module """
import sys

N = int(inputs())
K = int(sys.stdin.readline())
```

파이썬에서 사용자로부터 입력을 받는 모듈은 보통 `inputs()`, `sys.stdin.readline()` 을 사용한다. `inputs()` 는 입력 받는 데이터의 길이가 길고, 많아질수록 입력 효율이 떨어지는 단점이 있다. 그래서 이를 보완하기 위해 대부분의 코딩테스트 환경에서 입력을 받을 때는 `sys.stdin.readline()` 를 사용하는 것이 좋다. 어차피 테스트 케이스가 매우 길고 많기 때문에 일반적으로는 후자가 훨씬 효율적이다.

하지만 만약, 문제 조건에서 주어지는 입력 데이터의 개수와 그 형태가 적고 단순한 편이라면 전자의 사용도 고려해야 한다. 후자는 기본적으로 `sys` 모듈 내부에 내장된 메서드라서 `sys` 모듈을 불러와야 한다. 모듈을 `import` 하는 시간도 꽤나 걸리기 때문에, 데이터가 단순하다면 전자 사용도 고려해보자.

### `Theme 2. 선언/할당/재구성 횟수 줄이기`

파이썬은 자료구조 사용을 위해 명시적으로 메모리를 할당하고 해제하는 과정이 생략되어 있어서, 이 부분에 대한 인지를 하지 못하는 경우가 많다. 하지만 C/C++/JAVA 혹은 운영체제 공부를 해봤다면 알 것이다. 메모리 할당을 요구하는 일이 얼마나 시간을 많이 잡아 먹는 일인지 말이다. 프로그래머가 작성한 코드에서 메모리 할당을 요구한다면, 컴파일러/인터프리터는 운영체제 커널에 메모리 할당을 요구해야 한다. 그 다음 커널은 다시 가장 깊숙한 곳까지 내려가 메모리 할당을 하드웨어에 요청한 뒤, 작업 결과를 컴파일러에 전달해야 한다. 따라서 최대한 `할당`과 `재구성`하는 시간을 줄여야 효율적으로 코드를 작성할 수 있다.

```python
""" 1. 리스트 곱셈 """

arr1 = [0 for _ in range(100)]
arr2 = [0] * 100
```

arr1, arr2는 모두 길이가 100이고 모든 원소값이 0인 같은 배열(리스트)을 가지게 된다. 하지만, 실행 시간은 어느쪽이 더 빠를까?? 후자가 당연히 더 빠르다. 후자는 `재구성(resize)`연산 없이 한 번에 할당이 마무리 되기 때문이다. 반면 전자는 `선언`과 `할당` 및 `재구성` 시점이 서로 다르다. 그래서 일정 횟수마다 `재구성` 연산이 필요해진다. 앞서도 언급했듯, 커널에 메모리를 요구하는 횟수가 작아지는게 실행 시간에 유리하다. 따라서 명백히 후자의 실행 시간이 더 빠르다.

```python
""" 2. 튜플 사용 """

tdy, tdx = (-1, 1, 0, 0), (0, 0, -1, 1)  # tuple
ldy, ldx = [-1, 1, 0, 0], [0, 0, -1, 1]  # list
```

지금까지 설명한 내용과 같은 맥락에서, 배열 내부의 구성요소에 변경이 없는게 확실시 되는 경우, 배열을 튜플로 선언하는 것이 좋다. 위에 선언된 배열은 그래프 탐색 문제에서 상하좌우 방향 탐색을 구현할 때 자주 사용하는 방법이다. 상하좌우 방향을 표현하는게 목적이라서, 해당 배열은 미래에 변경될 일이 없다고 보장할 수 있다. 이런 경우에는 습관적으로 리스트를 사용하지말고, 튜플을 쓰자.

튜플이 리스트보다 나은 이유는 `immutable` 객체라는 점이다. `immutable` 객체는 변경되지 않는다는 것을 전제로 하기 때문에 `resize` 메서드가 객체에 없다. 이에 따라서 관련된 메타 정보도 저장하고 있을 필요가 사라진다. 그래서 리스트 보다 가볍고 빠른 계산이 가능하다. 또한 튜플은 20이하의 크기를 갖는 객체는 `래퍼런스 카운트`가 0이라도 `Python GC`가 메모리를 회수하지 않는다. 대신 메모리 상에 저장하고 있다가, 같은 크기의 튜플이 다시 한 번 선언될 때, 이것을 재활용한다. 

리스트와 튜플의 전체 생애주기를 비교해보면, 튜플의 커널&하드웨어 호출횟수가 리스트보다 훨씬 적어진다. 그래서 재구성이 필요없는 경우는 반드시 튜플로 선언하는게 이득이라고 볼 수 있다.

```python
""" 문자열 합치기 """

arr1, arr2 = 'I am a boy, ', 'you are a girl'

src = time.time()
result = arr1 + arr2
end = time.time()
print(f"Execution Time: {src-end}")
print(f"Result: {result}")

src = time.time()
result2 = ''.join([arr1, arr2])
end = time.time()
print(f"Execution Time: {src-end}")
print(f"Result: {result2}")

Execution Time: -3.600120544433594e-05
Result: I am a boy, you are a girl
Execution Time: -3.504753112792969e-05
Result: I am a boy, you are a girl
```

문자열은 파이썬에서 `immutable` 객체로 간주한다. 따라서 일반 연산자(+, *)를 사용해 문자열을 수정할 경우, 여타 `immutable` 객체처럼 새로운 메모리 할당이 발생한다. 이를 방지하기 위해 파이썬 내부적으로 최적화 되어 있는 `str.join()` 을 사용해 문자열을 조작하도록 하자.

### `Theme 3. 컴프리헨션/제너레이터`

```python
src = time.time()
arr1 = [i**2 for i in range(100000)]  # list comprehension
end = time.time()

print(f"Execution Time: {src-end}")
print(f"Memory Size: {sys.getsizeof(arr1)} byte")
# print(f"Result: {arr1}")

src = time.time()
arr2 = (i**2 for i in range(100000))  # init of generator
end = time.time()

print(f"Execution Time: {src-end}")
print(f"Memory Size: {sys.getsizeof(arr2)} byte")
print(f"Result: {arr2}")

Execution Time: -0.023138046264648438
Memory Size: 800984 byte
Execution Time: -5.2928924560546875e-05
Memory Size: 112 byte
Result: <generator object <genexpr> at 0x1058c3f20>
```

속도, 필요한 메모리 크기 모두 제너레이터가 압도적으로 효율적인 모습이다. 제너레이터의 경우, 실제로는 모든 연산값을 갖고 있는게 아니기 때문이다. 그래서 인덱싱, 슬라이싱등 파이썬 배열의 여러 연산을 활용할 수 없다. 따라서 상황에 따라서 컴프리헨션과 제너레이터를 사용하면 된다. 만약 인덱싱, 슬라이싱 연산처럼 일부에 접근하는 연산을 이용해야 한다면 컴프리헨션을, sum()처럼 전체 배열 단위 연산 혹은 단순히 순차적으로 원소 하나 하나에 접근해 무엇인가 해야 하는 상황(range() 등)이라면 제너레이터를 활용하자.