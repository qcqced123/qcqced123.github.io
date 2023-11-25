---
title: "🔢 Eigen Decomposition"
excerpt: "💡 Concept of Eigen Decomposition"
permalink: "/linear-algebra/eigen-decomposition"
toc: true  # option for table of content
toc_sticky: true  # option for table of content
categories:
  - Linear Algebra
tags:
  - Linear Algebra
  - Eigen Decomposition
  - Eigen Vector
  - Eigen Value
  - SVD
  - PCA
last_modified_at: 2023-11-25T23:00:00-05:00
---

고유값, 고유벡터, 고유값 분해는 비단 선형대수학뿐만 아니라 해석기하학 나아가 데이터 사이언스 전반에서 가장 중요한 개념 중 하나라고 생각한다. 머신러닝에서 자주 사용하는 여러 행렬 분해(Matrix Factorization) 기법(ex: `SVD`)과 `PCA`의 이론적 토대가 되므로 반드시 완벽하게 숙지하고 넘어가야 하는 파트다. 이번 포스팅 역시 [**<U>혁펜하임님의 선형대수학 강의</U>**](https://www.youtube.com/watch?v=PP9VQXKvSCY&t=108s&ab_channel=%ED%98%81%ED%8E%9C%ED%95%98%EC%9E%84%7CAI%26%EB%94%A5%EB%9F%AC%EB%8B%9D%EA%B0%95%EC%9D%98)와 [**<U>공돌이의 수학정리님의 강의 및 포스트</U>**](https://www.youtube.com/watch?v=7dmV3p3Iy90&ab_channel=%EA%B3%B5%EB%8F%8C%EC%9D%B4%EC%9D%98%EC%88%98%ED%95%99%EC%A0%95%EB%A6%AC%EB%85%B8%ED%8A%B8) 그리고 [**<U>딥러닝을 위한 선형대수학 교재</U>**](https://product.kyobobook.co.kr/detail/S000001743773)을 참고하고 개인적인 해석을 더해 정리했다.

### `🌟 Concept of Eigen Value & Vector`

$$
Av = \lambda v
$$

등식을 만족시키는 벡터 $v$를 `고유 벡터(Eigen Vector)`, 람다 $\lambda$를 `고유값(Eigen Value)`이라고 정의한다. 좌변의 $A$는 `선형 변환(행렬)`을 의미한다. 이러한 정보를 활용해 위 등식의 의미를 살펴보자. 어떤 선형변환 $A$와 벡터 $v$를 곱했더니, 어떤 스칼라와 벡터를 곱한 결과와 같았다는 것인데, 벡터에 스칼라를 곱하면 그 크기만 변화할 뿐 방향은 이전과 동일하다. 따라서 선형변환 $A$를 가해도 그 크기만 스칼라 배(고유값 배)만큼 변할뿐 방향은 동일한 벡터를 찾고자 하는게 위 수식의 목적이며 이게 바로 고유벡터의 정의가 된다. 

그렇다면 수식을 풀어서 고유값과 고유벡터를 직접 구해보자. 먼저 좌변으로 모든 항을 넘긴 뒤, 고유 벡터 $v$로 좌변의 모든 항을 묶어준다.

$$
(A - \lambda I) v = 0
$$

고유 벡터 $v$로 묶어준다고만 했는데 왜 갑자기 람다 뒤에 항등행렬이 붙게 되었을까?? 람다는 고유값, 다시 말해 스칼라다. `행렬 - 스칼라`는 불가능하기 때문에 선형변환 $A$와 크기를 맞춰주기 위해 곱한 것이다. 다시 등식을 전체적인 관점에서 살펴보자. 지금 `행렬•벡터 = 0` 의 형태를 취하고 있다. 어디서 많이 본 듯한 꼴이 아닌가?? 바로 행렬의 영공간을 구할 때 사용하던 수식이다. 따라서 우리는 고유벡터 $v$가 좌측 괄호 안의 행렬 $A-\lambda I$의 영공간이 span하는 공간 어딘가에 위치했다는 것을 알 수 있다. 

한편, 우리가 이 등식을 풀어헤친 목적은 고유값 그리고 고유벡터를 구하기 위함이다. 등식을 만족시키려면 고유벡터가 0이기만 하면 되겠지만, $v=0$인 경우를 찾자고 우리가 이렇게 고생하는 것은 당연히 아닐 것이다. 따라서 $v=0$이 아니라 좌측 괄호 안의 항 $A-\lambda I=0$이 되어야 한다. 이 때 $det(A-\lambda I) =0$를 만족해야 한다. 그 이유는 만약 행렬식이 0이 아니라면 역행렬이 존재한다는 것이 되고, 전체 등식에서 좌측 항에 대한 역행렬을 양변에 곱해주면 다시 $v=0$이라는 결과를 얻게 된다. 따라서 반드시 $det(A-\lambda I) =0$을 충족해 역행렬이 없도록 만들어야 한다. 

따라서 결론적으로 우리는 두 가지 수식을 풀어내면 고유값과 고유벡터를 구할 수 있다.

$$
N(A-\lambda I) = V \\
det(A-\lambda I) = 0
$$

이 때, 영공간에 `span`하는 벡터는 무수히 많기 때문에 일반적으로 `Basis`를 고유값으로 간주한다. 

### `🔢 Eigen Decomposition`

$$
A = V\Lambda V^{-1} \\
\Lambda = V^{-1}AV
$$

위 수식과 같은 형태로 임의의 정사각행렬 $A$를 표현 가능하다면, 우리는 이러한 행렬 $A$를 `Diagonalizable Matrix`라고 부르며, `Diagonalizable Matrix`를 고유벡터와 고유값 행렬로 분해하는 것을 `고유값 분해(Eidgen Decomposition)`라고 한다. 

여기서 `Diagonalizable Matrix` 이란, 고유값 행렬을 이용해 대각행렬로 변환이 가능한 정사각행렬을 말한다. 두번째 수식이 바로 `Diagonalizable Matrix` 를 표현한 것이다. 어떤 행렬이 `Diagonalizable Matrix` 하다는 것은 다시 말해, 행렬에 `Independent`한 고유벡터가 N개 있다는 것과 동치다. 방금 서술한 사실을 유도해보자. 

3X3 크기의 행렬 $A$와 서로 독립인 고유 벡터 $v_1, v_2, v_3$과 이에 대응되는 고유값 $\lambda_1, \lambda_2, \lambda_3$이 있다고 가정해보자. 여러개의 고유 벡터와 고유값을 수식 하나로 표현하기 위해 벡터화를 이용하고자 한다. 

$$
A[v_1, v_2, v_3] = [v_1, v_2, v_3]•   \begin{bmatrix} 
   \lambda_1 & 0 & 0 \\
   0 & \lambda_2 & 0 \\
   0 & 0 & \lambda_3 \\
   \end{bmatrix} \\
$$

우리는 지금 어떤 행렬이 `Diagonalizable Matrix` 일 때 벌어지는 현상에 대해 증명하는게 목표라서 좌변에 행렬 $A$만 남기려고 한다. $[v_1, v_2, v_3]$ 은 서로 독립인 고유 벡터다. 그리고 사이즈는 3x3으로 정사각행렬에 해당된다. 열벡터가 서로 독립이면서 정사각행렬에 해당되기 때문에 $[v_1, v_2, v_3]$ 은 가역행렬의 조건을 모두 충족한다. 따라서 양변에 $[v_1, v_2, v_3]$ 의 역행렬을 곱해주자. 이제부터 편의상 $[v_1, v_2, v_3]$ 은 $V$,  고유값-대각행렬(우변 오른쪽 항) $\Lambda$로 표기하겠다.

$$
A = V\Lambda V^{-1} \\
$$

$V$가 `Independent`한 고유벡터가 N개를 갖고 있기 때문에 $V$를 일부분으로 갖고 있는 행렬 $A$는 당연히 `Independent`한 고유벡터가 N개 있다고 말할 수 있다. 

### `⭐️ Property of Eigen Decomposition`

고유값 분해가 가능한 `Diagonalizable Matrix` $A$의 속성에 대해 알아보자. 이런 속성들은 이후 `PCA`, `SVD`에서 사용되니 숙지하고 있는게 좋다.

- **1) $A^k = V \Lambda V^{-1}•V \Lambda V^{-1} ... = V \Lambda^k V^{-1}$**
- **2) $A^{-1} = (V \Lambda V^{-1})^{-1}$= $(V \Lambda^{-1} V^{-1})$**
    - $AA^{-1}=I$
- **3) $det(A) = det(V \Lambda V^{-1}) = det(V)det(\Lambda)det(V ^{-1}) = \prod_{i=1}^{N} {\lambda_i}$**
    - **행렬식은 곱으로 쪼개는게 성립**
- **4) $tr(A)=tr(V \Lambda V^{-1})=tr( \Lambda V^{-1}V)=tr(\Lambda)=\sum_i^{N}\lambda_i$**
    - **`trace` 는 원소의 순서를 바꾸는거 허용**
- **5) rank-difficient == $det(A)=0$ ⇒ 행렬 $A$에는 값이 0인 고유값이 적어도 하나 이상 존재**
    - **3)번 속성 이용**
- **6) Diagonalizable Matrix의 non-zero eidgen value 개수 == rank(A)**
    - $rank(A) = rank(V \Lambda V^{-1}) = rank(\Lambda)$
    - $V, V^{-1}$ **은 서로 독립인 열벡터를 쌓아 만든 행렬이라서 반드시 Full Rank**
    - **랭크의 성질에 의해 가장 작은 값이 행렬의 랭크가 된다**
    - **`non-zero eigen value 개수` ==** $rank(\Lambda)$

### `💡 Insight of Eigen Decomposition`

이렇게 고유값, 고유벡터, 고유값 분해에 대해서 전반적으로 살펴보았다. 하지만 아직도 왜 고유값 분해가 그리도 중요하다는 것인지 아직 감이 오지 않을 것이다. 고유값 분해의 중요성에 대해 알아보기 위해 먼저 다음과 같은 명제에 대해서 증명해보자. 

***“대칭행렬(Symmetric Matrix)은 대각화 가능한 행렬(Diagonalizable Matrix)이다”***

$$
V^T = V^{-1} = Q
$$

대칭행렬은 정사각행렬 중에서 원본과 전치행렬이 동일한($A=A^{T}$) 특수 행렬을 말한다. 따라서 어떤 행렬 $A$가 대칭행렬이라면, $V \Lambda V^{-1} = V^{-T} \Lambda V^{T}$가 된다. 각변의 가장 마지막 항에 주목해보자. 등식 조건에 의해 $V^{-1} = V^T$가 성립하기 때문에 행렬 $V$는 전치행렬과 역행렬이 같은 행렬이 된다. 다시 말해, $V$는 직교행렬 $Q$가 된다. 따라서 행렬 $A$에 대한 고유값 분해식을 아래처럼 직교행렬로 표현할 수 있다.

$$
A = Q \Lambda Q^{-1} \\
A = [q_1, q_2, q_3]•\begin{bmatrix} 
   \lambda_1 & 0 & 0 \\
   0 & \lambda_2 & 0 \\
   0 & 0 & \lambda_3 \\
   \end{bmatrix}•\begin{bmatrix} 
   q_1^T \\
   q_2^T \\
   q_3^T \\
   \end{bmatrix} \\
$$

이제 우변을 수식을 전개해서 그 의미를 알아보자. 전개하면 아래와 같다.

$$
A = \lambda_1q_1q_1^T + \lambda_2q_2q_2^T + \lambda_3q_3q_3^T
$$

우변의 항을 하나 하나 살펴보자. 세개의 항은 모두 개별 고유벡터에 대한 `고유값`, `고유벡터` 그리고 `고유벡터의 전치`에 대한 곱으로 구성되어 있다. 고유벡터와 그것의 전치벡터의 곱은 크기는 `nxn`이지만, 사실 같은 벡터를 두번 곱한 것과 같기에 랭크는 1이된다. 다시 말해, 3차원 공간에서 1차원 직선 공간으로 `span`하는 부분 공간이 3개가 만들어지며 3개의 부분 공간은 서로 독립이면서, 모두 근본이 직교 행렬의 열벡터라는 점 때문에 서로 직교한다. 따라서 우리는 대칭행렬 $A$를 크기는 `NxN`이면서 랭크는 `1`인 행렬 3개를 고유값을 이용해 `가중합 방식`으로 더한 것이라고 해석할 수 있다. 뒤집어 서술하면 대칭행렬 $A$를 크기는 `NxN`이면서 랭크는 `1`인 행렬 `3`개로 쪼개는 방식이 바로 `고유값 분해`이다.

고유값에 따라, 부분 공간의 크기를 조절할 수 있다는 점에서 차원 축소나 제거처럼 중요도가 높은 데이터•특징만 추출하는게 가능해진다. 이러한 고유값 분해를 정사각행렬(대칭행렬)이 아닌 일반적인 직사각행렬에도 적용할 수 있도록 개념을 확장한게 바로 `SVD(Singular Vector Decomposition)`이고, 중요도(고유값의 크기)에 따라서 중요한 특징•데이터만 남기는 방법론은 `PCA(Princlpal Component Analysis)`의 이론적 토대가 된다.

이렇게 대칭행렬은 대각화 가능한 행렬이라는 점을 통해 `고유값 분해`의 의미에 대해서 알아보았다. 이제 마지막으로 선형변환으로서 행렬 $A$가 갖는 의미를 살펴보자. 고유벡터가 아닌 임의의 벡터 $\vec x$를 선형변환 $A$에 통과시켜보자. 그럼 우리는 아래와 같은 식을 얻을 수 있다.

$$
A\vec x = \lambda_1q_1q_1^T•\vec x + \lambda_2q_2q_2^T•\vec x + \lambda_3q_3q_3^T•\vec x
$$

우변을 해석해보자. 아까 고유값 분해의 의미를 살펴보면서 $q_1q_1^T$는 전체 공간에서 1차원 직선 공간으로 span하는 부분 공간을 의미한다고 했었다. 따라서 우변에는 서로 다른 항이 3개 있기 때문에 부분 공간이 3개 있는 3차원 공간이 형성된다. 이제 부분 공간과 벡터 $\vec x$를 내적한 형태로 바라볼 수 있다. 내적은 정사영이다. 따라서 $q_nq_n^T•\vec x$은 벡터 $\vec x$를 부분 공간에 정사영 내려준 벡터가 된다. 그리고 고유값을 곱해 정사영 내린 벡터들의 크기를 조절해주는게 우변의 의미가 된다.