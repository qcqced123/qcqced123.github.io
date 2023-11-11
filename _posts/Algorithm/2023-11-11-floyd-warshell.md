---
title: "🗂️ Graph Theory 3: Floyd-Warshall"
excerpt: "Floyd-Warshall Algorithm with DP Tabulation"
permalink: "/algorithm/floyd-warshell"
toc: true  # option for table of contents
toc_sticky: true  # option for table of content
categories:
  - Algorithm
tags:
  - Python
  - Codeing Test
  - Algorithm
  - Floyd-Warshall
last_modified_at: 2023-11-11T12:00:00-05:00
---

### `📚 Floyd-Warshall`

`Floyd-Warshall`은 모든 지점에서 다른 모든 지점까지의 최단 경로를 구하는 알고리즘이다. 지정된 출발점에서 나머지 다른 지점가지의 최단 경로를 구하는 다익스트라 알고리즘과는 차이가 있다. 따라서 솔루션을 도출하는 방식에도 살짝 차이가 생기는데, `Floyd-Warshall` 은 그리디하게 매번 최단 경로에 있는 노드를 구할 필요가 없다. 이유는 모든 지점에서 다른 모든 지점까지의 경로를 구해야 하기 때문에 그리디 대신 `DP Tabulation`으로 문제를 풀기 때문이다.  

`Floyd-Warshall` 은 주어진 $N$ 개의 노드에 대해서 매번 $N^2$ 번의 연산을 통해 최단 거리를 갱신한다. 따라서 최종적으로 $O(N^3)$ 의 시간 복잡도를 갖게 된다. 연산은 직선 경로와 경유 경로를 비교하는 형태로 이뤄진다. 둘 중에서 더 작은 값이 `DP Table`에 저장된다. 여기서 경유 경로란 전체 $N$ 개의 노드에 대한 `iteration` 중에서 $i$ 번째 노드를 경유하는 경로를 말한다. 만약 직선 경로가 [$d, k$]라면 경유 경로는 [$d, i$] + [$i, k$] 가 된다. 우리는 테이블에 직선 경로와 경유 경로중에서 최단 거리만 저장하고 있기 때문에, 이렇게 하나의 노드에 대해서만 경유하는 경우만 고려해도 괜찮다. 만약 3개의 중간 노드를 경유해야만 최단 거리가 되는 경로가 있다고 가정해보자. 최적의 솔루션인 전체 경로의 부분 경로 역시 중간에 최단 경로로 선택되어 이미 테이블 어딘가에 값으로 자리 잡고 있게 된다. 따라서 결국엔 부분 집합의 합으로 전체 최적 솔루션인 경로를 도출해낼 수 있게 된다. 이러한 경우의 수를 고려하기 위해 `DP  Table`을 사용하게 된 것이라 생각한다.

```python
""" Floyd-Warshall Implementation """

import sys
from typing import List
"""
[Floyd-Warshall]
1) DP Table init
2) triple-loop
    - dp[i][j] = min(dp[i][j], dp[i][k] + dp[k][j])
3) print result
"""
N = int(sys.stdin.readline())  # 노드 개수
M = int(sys.stdin.readline())  # 간선 개수
dp = [[float('inf')] * (N+1) for _ in range(N+1)]

# 1) DP Table init
for i in range(1, N+1):
    dp[i][i] = 0

for _ in range(M):
    src, end, cost = map(int, sys.stdin.readline().split())
    dp[src][end] = cost

# 2) triple-loop
for k in range(1, N+1):
    for i in range(1, N+1):
        for j in range(1, N+1):
            if i == j:
                continue
            dp[i][j] = min(dp[i][j], dp[i][k] + dp[k][j])

# 3) print result
for i in range(1, N+1):
    for j in range(1, N+1):
        if dp[i][j] == float('inf'):
            print('INF', end=' ')
        else:
            print(dp[i][j], end=' ')
    print()
```