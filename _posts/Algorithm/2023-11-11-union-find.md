---
title: "🗂️ Graph Theory 4: Union-Find (Disjoint Set)"
excerpt: "Union-Find Algorithm with Disjoint Set"
permalink: "/algorithm/union-find"
toc: true  # option for table of contents
toc_sticky: true  # option for table of content
categories:
  - Algorithm
tags:
  - Python
  - Codeing Test
  - Algorithm
  - Union-Find
last_modified_at: 2023-11-11T12:00:00-05:00
---

### `🙅 Disjoint Set`



서로 공통된 원소를 가지고 있지 않은 여러 집합들을 지칭하는 용어다. 개별 원소가 정확히 하나의 집합에 속하며, 어떤 집합도 서로 공통 원소를 가지고 있지 않아야 한다. 서로소 집합 자료구조를 사용하면 서로 다른 원소들이 같은 집합군에 속해 있는가 판별하는 것과 같은 작업을 쉽게 할 수 있다. 그렇다면 이제부터 자료구조로서 서로소 집합을 효과적으로 표현하고 조작할 수 있는 `Makeset`, `Union`, `Find` 연산에 대해서 알아보자.

### `🗂️ Makeset`

트리 자료구조를 활용해 집합을 표현하는 방법 중 하나로, 주어진 요소만 중복없이 포함하는 집합을 생성하는 연산이다. 실제 코드상 구현으로는 배열, 리스트 자료구조를 활용한다. 배열의 인덱스를 개별 원소의 식별자로 간주하고 해당 위치의 값에는 부모 원소의 인덱스를 채워 넣는다. 만약 인덱스와 원소값이 동일하다면 해당 원소가 포함된 집합에서 현재 원소가 루트 노드임을 의미한다. 이렇게 특정 인덱스의 원소값을 타고 거슬러 올라가다면 만나게 되는 루트 노드의 값을 이용해 우리는 서로 다른 두 원소가 같은 집합에 속하는지 혹은 다른 집합에 속하는지 구별할 수 있게 된다. 

```python
""" Disjoint Makeset Example """

N = int(sys.stdin.readline()) # 노드 개수
M = int(sys.stdin.readline()) # 엣지 개수
parent = [0]*(N+1)

# 1) Makeset Array Init
for i in range(1, N+1):
    parent[i] = i
```

배열의 인덱스를 개별 노드의 식별자로 사용하기 위해, 전체 그래프 상의 노드 개수만큼 배열의 크기를 초기화 해주고 있다. 그리고 초기에는 아직 노드 사이의 연결 정보에 대해서 주어진게 전혀 없기 때문에 개별 노드 자신이 루트 노드가 되도록 초기화를 해주는게 일반적이다. 이렇게 초기화한 배열은 `Union`, `Find` 연산에 활용된다. 

### `🔬 Find`

어떤 원소가 속한 집합의 루트 노드 값을 반환하는 연산이다. `Find` 연산은 앞서 초기화한 `Makeset Array` 를 해당 집합(트리)의 루트 노드를 만날 때까지 재귀적으로 순회한다. 실제로는 단순 루트 노드를 반환하는 용도로 사용하지 않고, 특정 두 원소가 같은 집합(트리)에 속하는지 아니면 서로 다른 집합에 속하는지 판정하는데 사용된다. 서로 다른 두 원소를 `Find` 연산자에 넣어주면 각각의 루트 노드를 구할 수 있는데, 이 때 서로 같은 루트 노드값을 반환한다면 같은 집합이라고 간주하고 다르다면 서로 서로소 관계에 있다고 판단할 수 있다. 

```python
""" find method """

def find(arr: list, x: int) -> int:
    """ method for finding root node """
    if arr[x] != x:
        arr[x] = find(arr, arr[x])
    return arr[x]

# 1) Kruskal Algorithm
result = 0
for j in range(M):
    weight, start, final = graph[j]
    if find(parent, start) != find(parent, final):
        union(parent, start, final)
        result += weight
```

위 소스코드처럼 `Kruskal` Algorithm처럼 최소 스패닝 트리가 필요한 상황에 자주 사용된다. 또한 근본이 트리 자료구조에 대한 연산이라는 점을 활용해, 특정 그래프의 사이클 여부를 판정하는 알고리즘으로도 많이 사용되고 있다.

### `👩‍👩‍👧‍👦 Union`

두 개의 집합을 하나로 합치는 연산이다. 집합의 루트 노드를 다른 집합의 루트 노드 아래에 연결하는 방식으로 합친다. 합치는 방식에는 다양한 방법론이 존재하는데, 일반적으로 루트 노드의 번호가 더 작은 쪽에 더 큰 쪽의 집합(트리)를 붙여주는 `Union by Rank` 방식을 많이 사용한다. 

```python
""" union method """

def union(arr: list, x: int, y: int):
    """ method for union-find """
    x = find(arr, x)
    y = find(arr, y)
    if x < y:
        arr[y] = x
    else:
        arr[x] = y

# 1) Kruskal Algorithm
result = 0
for j in range(M):
    weight, start, final = graph[j]
    if find(parent, start) != find(parent, final):
        union(parent, start, final)
        result += weight

print(result)
```

역시 마찬가지로 최소 스패닝 트리가 필요한 상황에 자주 사용되고 있으며, 주로 Find를 통해 서로소 집합 관계에 놓인 집합들을 판정하고, 그들을 하나로 통합시키는데 자주 쓰인다.