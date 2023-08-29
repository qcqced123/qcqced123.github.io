---
title: "✏️  Summary of Useful Library for Coding Test"
excerpt: "코딩 테스트 풀이에 자주 사용되는 파이썬 내장 라이브러리 사용법 정리"
permalink: "/algorithm/useful_library"
toc: true  # option for table of contents
toc_sticky: true  # option for table of content
categories:
  - Algorithm
tags:
  - Python
  - collections
  - Codeing Test
  - Algorithm
last_modified_at: 2023-08-17T12:00:00-05:00
---

### `📚 collections`

#### `🪢 deque`
`python`에서 `stack`이나 `queue` 자료형을 구현할 때 일반적으로 사용하는 내장 라이브러리 `collections`에 구현된 클래스다. 메서드가 아닌 객체라서 사용하려면 초기화가 필요하다. 사용 예시를 보자.

```python
# collections.deque usage example
deque([iterable[, maxlen]]) --> deque object

>>> from collections import deque, Counter
>>> queue = deque() # 1)
>>> queue
deque([])

>>> queue = deque(())  # 2)
>>> queue
deque([])

>>> queue = deque([]) # 3) 
>>> queue
deque([])

>>> queue = deque([1,2]) # 4)
>>> queue
deque([1, 2])

>>> queue = deque((1,2))  # 5)
>>> queue
deque([1, 2])

```

객체를 초기화할 때 값을 넣어줄 것이 아니라면 리스트를 굳이 전달하지 않아도 똑같이 초기화가 된다. 다만, 특정값을 넣어줄 것이라면 반드시 `iterable` 객체를 전달해야 한다. 이 때 `iterable` 객체는 어떤 형태를 넣어도 같은 결과를 반환하게 된다. 예를 들어 리스트를 넣고 `deque`를 초기화 하나 튜플을 넣고 초기화 하나 결과는 같다는 것이다. 예시 코드의 4번과 5번 예시를 보자. 각각 리스트와 튜플을 넣고 초기화를 했지만 초기화 결과는 동일한 것을 확인할 수 있다.

```python
# deque indexing & slicing

>>> queue[1]
2

>>> queue[:]
TypeError: sequence index must be integer, not 'slice'
```

`deque` 는 리스트처럼 인덱싱은 가능하지만, 슬라이싱 기능은 활용할 수 없으니 참고하자.

```python
# deqeue max_len example

>>> queue = deque([1,2,3,4,5],6)  # 객체라서 초기화가 필요하다.
>>> queue.append(6)
>>> queue.append(7)
>>> queue
deque([2, 3, 4, 5, 6, 7], maxlen=6)
```

한편, `max_len` 매개변수를 통해 선언한 객체의 최대 길이를 지정해 줄 수 있는데, 이 때 객체가 지정 길이를 넘어서면 큐의 동작처럼 가장 왼쪽에 위치한 원소를 먼저 빼고 새로운 원소를 가장 오른쪽에 추가한다.

만약 deque를 이용해 스택을 구현하고 싶다면 `deque.append`와 `deque.pop`을 사용하자. 정확히 선입후출 동작을 구현할 수 있다. 한편 큐를 구현하고 싶다면 `deque.append`와 `deque.popleft`를 함께 사용하자. 정확히 선입선출 구조를 만들어낼 수 있다. 한편 가장 첫번째 원소 앞에 새로운 원소를 추가하고 싶다면 `appendleft()` 역시 지원하고 있으니 참고해두면 좋을 것 같다. 

```python
# deque stack & queue 구현

""" stack """
>>> queue = deque()
>>> queue.append(1)
>>> queue.append(2)
>>> queue.append(3)
>>> queue.pop()
3
>>> queue.pop()
2
>>> queue.pop()
1

""" queue """
>>> queue.append(1)
>>> queue.append(2)
>>> queue.append(3)
>>> queue.popleft()
1
>>> queue.popleft()
2
>>> queue.popleft()
3
```

#### `🗂️ Counter`

`Iterable` 객체에 있는 `hashable item`의 개수를 세어 `Dict` 자료형으로 반환하는 역할을 한다. 아래 예시를 보자.

```python
# collection.Counter example

>>> test = [1,2,4,5,5,5,4,4,4,4,3,3,12,12,1]
>>> counter = Counter(test)
>>> counter
Counter({1: 2, 2: 1, 4: 5, 5: 3, 3: 2, 12: 2})

>>> test = 'aabcdacdbbbbaaaadddcccbcbcbcbcbcbbcbcbcbcbcdaadaabadbcdbcdacdbacdbacdbcadacbbcabcadb'
>>> counter = Counter(test)
>>> counter
Counter({'a': 19, 'b': 26, 'c': 24, 'd': 15})

>>> counter.most_common(3)
[('b', 26), ('c', 24), ('a', 19)]

>>> counter.values
dict_values([19, 26, 24, 15])

>>> sorted(counter)
['a', 'b', 'c', 'd']

>>> ''.join(sorted(counter.elements()))
'aaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbccccccccccccccccccccccccddddddddddddddd'
```

이처럼 `Iterable` 객체의 원소에 접근해 다양한 일을 수행할 수 있다. `most_common`은 가장 많이 사용된 원소를 순서대로 사용자가 지정한 등수만큼 보여준다. `values` 는 개별 `key`의 값을 반환한다. `join`, `sorted` 와 함께 하면 원소를 반복하는 것도 가능하다.

#### `👌 defaultdict`

사전의 `key`에 대한 `value`가 아직 없지만, 미리 `value`의 자료형을 지정해주고 싶을 때 사용하면 유용하다. 예를 들어 동물의 종류를 분류하는 사전을 만들고 싶다고 가정해보자. 0번 `key`가 조류라고 해보자. 그럼 조류에 해당되는 `value`는 참새, 비둘기 … 앵무새 등 정말 많을 것이다. 이것을 한번에 그룹핑해 조류라는 카테고리에 속한다고 표현해주기 위해 우리는 리스트를 `value`의 기본 자료형으로 지정해주는 것이다. 이 때 defaultdict을 이용하면 쉽게 구현이 가능하다.

```python
# collections.defaultdict example
from collections ipmort defaultdict

animal_dict = defaultdict(list)
```

`defaultdict` 에 어떤 자료형을 매개변수로 전달하는가에 따라서 초기화 되는 기본 자료형이 바뀐다. 우리는 `list` 를 기본 자료형으로 지정했지만 `int`, `set` 같은 것도 가능하니 참고해두자.

### **`🗂️ heapq`**

다익스트라 최단 경로 알고리즘을 포함한 다양한 알고리즘 문제에서 우선순위 큐 기능을 구현할 때 사용하는 라이브러리로 기본적으로 최소 힙(오름차순, 파이썬 내장 정렬 알고리즘의 특성으로 모두 기본값이 오름차순) 구성으로 되어 있다. `heapq.heappush()` 로 힙에 원소를 삽입하는 기능을 구현하며, `heapq.heappop()` 을 통해서 힙으로부터 원소를 빼낸다. 아래는 힙 정렬을 구현한 코드이다.

```python
# Min Heapsort
import heapq

def min_heapsort(iterable):
	h = []
	result = []
	# heap에 원소 삽입
	for value in iterable:
			heapq.heappush(h, value)
	# heap으로부터 원소 빼내기
	for _ in range(len(h)):
			result.append(heapq.heappop(h))
	
	return result

result = min_heapsort([])
```

```python
# Max Heapsort
import heapq

def max_heapsort(iterable):
	h = []
	result = []
	for value in iterable:
			heapq.heappush(h, -value)
	for _ in range(len(h)):
			result.append(-heapq.heappop(h))

	return result

result = max_heapsort([])
```

구현된 힙의 모든 원소가 정렬되는 것은 아니며 현재 최대,최소 원소에 대한 정렬만 보장하기 때문에 주의가 필요하다. 개별 시점에서 최대,최소값만 필요하다면 힙정렬 사용을 고려해보자.

### **`🗂️ sort & sorted`**

python `iterable` 객체를 빠르게 정렬할 때 사용하는 기능이다. 두 함수 모두 기능은 같지만 적용 대상 범위가 다르며, 함수 실행 결과가 반환 방식도 다르기 때문에 사용할 때 주의해야 한다. 

```python
# sort, sorted 차이
result = [[1,2], [2,1], [6,2]]
result.sort(key=lambda x: x[1], reverse=False) # 정렬 결과가 result에 반영
sorted(result, key=lambda x: x[1], reverse=False) # 정렬 결과를 result에 반영 x
```

`sort`는 정렬 결과를 매개 변수로 입력한 `iterable` 객체에 바로 적용되는 `in-place` 연산인 반면, `sorted`는 그렇지 않다. 한편 `sort` 는 리스트 자료형(`mutable object`)만 매개 변수로 입력 가능하지만, `sorted`는 모든 `iterable` 객체를 사용 가능하다. 

`reverse`, `reversed` 역시 위와 같은 규칙을 따르는데, `sort`, `reverse` 처럼 자료형 객체 내부에 내장된 메서드인 경우는 `in-place` 연산을 지원하고 `sorted`, `reversed` 처럼 `global` 한 내장 메서드인 경우는 `in-place` 를 지원하지 않는다. 이러한 경우에는 반드시 다른 변수에 대입해줘야 하니 주의하자.

만약 다중 조건을 적용해 `Iterable` 객체를 정렬하고 싶다면, 아래와 같이 튜플 형태로 `lambda function` 에 적용하고 싶은 우선순위대로 기준을 입력해주면 된다. 구체적인 예시는 아래와 같다.

```python
""" 
lecture_schedule = [시작시간, 끝시간] 
끝나는 시간을 기준으로 오름 차순 정렬하되, 끝나는 시간이 같으면, 시작 시간 오름 차순 정렬 적용 
"""
lecture_schedule = [[4,4], [1,4], [3,4]]
lecture_schedule.sort(key=lambda x: (x[1], x[0])) # key=lambda x:(우선순위1, 우선순위2, 우선순위3 ...)
lecture_schedule.sort(key=lambda x: (-x[1], x[0])) # - 붙인 정렬 조건은 현재 정렬 기준과 반대로
```

마지막으로 `-`를 붙인 조건은 현재 정렬 기준(오름차순, 내림차순)과 반대로 정렬을 하게 된다. 예시 코드의 3번째 라인처럼 `-` 을 붙이면 1번째는 내림 차순으로, 2번째는 오름 차순으로 정렬을 수행하게 된다. 정말 유용하니 꼭 기억해두자. 이걸 모르면, 해당 동작을 구현하기 위해 엄청난 코드 라인을 소비해야 한다.

```python
# lambda x: x[1]과 동일한 결과
def second(data):
	return data[1]

result = [[1,2], [2,1], [6,2]]
result.sort(key=second, reverse=False) # 정렬 결과가 result에 반영
sorted(result, key=second, reverse=False) # 정렬 결과를 result에 반영 x
```

`lambda` 대신 직접 함수를 정의해 사용하는 것도 가능하니 꼭 기억해두자.


