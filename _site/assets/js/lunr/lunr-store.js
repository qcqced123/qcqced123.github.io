var store = [{
        "title": "📏 Lp-Norm: Concept & Insight",
        "excerpt":"\\[||x||_p = (∑_{i=1}^n |x_i|^p)^{1/p}\\] Lp-Norm은 Lebesgue라는 프랑스 수학자에 의해 고안된 개념으로, 기계학습을 공부하는 사람이라면 지겹도록 듣는 L2-Norm, L1-Norm을 일반화 버전이라고 생각하면 된다. 다시 말해, 벡터의 크기를 나타내는 표현식을 일반화한 것이 바로 Lp-Norm 이며 수식은 위와 같다. p=1이라고 가정하고 수식을 전개해보자. $||x||_1 = (|x_1|^1 + |x_2|^1+ … + |x_n|^1)^{1/1}$이 된다. 우리가...","categories": ["Linear Algebra"],
        "tags": ["Linear Algebra","Norm","Pooling"],
        "url": "/linear-algebra/lp-norm",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "📐 Inner Product: Projection Matrix, Least Sqaure Method",
        "excerpt":"💡 Concept of Inner Product \\[a^Tb = ||a||•||b||cos\\theta\\] 내적은 Inner Product, Dot Product, Scalar Product로 불리며 두 벡터의 유사도, 즉 닮은 정도를 구하는데 사용되는 벡터•행렬 연산의 한 종류다. 두 벡터의 정사영과도 동일한 개념으로 사용된다. 위 수식의 우변에 주목해보자. $||a||cos\\theta$ 는 벡터 $a$를 벡터 $b$에 정사영 내린 크기로 해석할 수 있다. 한편...","categories": ["Linear Algebra"],
        "tags": ["Linear Algebra","Inner Product","Projection Matrix","내적","정사영"],
        "url": "/linear-algebra/inner-product",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "🤔 RuntimeError: Function 'LogSoftmaxBackward0' returned nan values in its 0th output",
        "excerpt":"🔥 Pytorch Backward 과정에서 NaN 발생하는 문제 커스텀으로 모델, 여러 풀링, 매트릭, 손실 함수들을 정의하면서부터 제일 많이 마주하게 되는 에러다. 진심으로 요즘 CUDA OOM 보다 훨씬 자주 보는 것 같다. 해당 에러는 LogSoftmax 레이어에 전달된 입력값 중에서 nan, inf 가 포함되어 연산을 진행할 수 없다는 것을 의미한다. 딥러닝 실험을 진행하면서...","categories": ["Framework & Library"],
        "tags": ["Pytorch","Logsoftmax","NaN","Error Handling"],
        "url": "/framework-library/backward-nan/",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "🖥️ RuntimeError: Attempting to deserialize object on CUDA device 2 but torch.cuda.device_count() is 1. Please use torch.load with map_location to map your storages to an existing device",
        "excerpt":"🔢 Pytorch 잘못된 CUDA 장치 번호 사용 문제 model.load_state_dict( torch.load(path, map_location='cuda:0') ) pretrained model, weight를 load하거나 혹은 훈련 루프를 resume 을 위해 torch.load() 를 사용할 때 마주할 수 있는 에러 로그다. 발생하는 이유는 현재 GPU 에 할당하려는 모델이 사전 훈련때 할당 되었던 GPU 번호와 현재 할당하려는 GPU 번호가 서로 상이하기...","categories": ["Framework & Library"],
        "tags": ["Pytorch","CUDA"],
        "url": "/framework-library/cuda-num/",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "🚚 RuntimeError: stack expects each tensor to be equal size, but got [32] at entry 0 and [24] at entry 1",
        "excerpt":"📏 가변 길이의 텐서를 데이터로더에 전달하는 경우 커스텀 데이터 클래스와 데이터로더를 통해 반환되는 데이터 인스턴스의 텐서 크기가 일정하지 않아 발생하는 에러다. 특히 자연어 처리에서 자주 찾아 볼 수 있는데 데이터로더 객체 선언 시, 매개변수 옵션 중에 collate_fn=collate 를 추가해주면 해결 가능한 에러다. 이 때 매개변수 collate_fn 에 전달하는 값(메서드)은 사용자가...","categories": ["Framework & Library"],
        "tags": ["Pytorch","DataLoader","collate_fn","Dynamic Padding","Padding"],
        "url": "/framework-library/dataloader-collatefn",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "🪢 assert len(optimizer_state[\"found_inf_per_device\"]) > 0, \"No inf checks were recorded for this optimizer.\" AssertionError: No inf checks were recorded for this optimizer.",
        "excerpt":"🤔 Optimizer가 손실값을 제대로 Backward 할 수 없는 문제 텐서의 계산 그래프가 중간에 끊어져 옵티마이저가 그라디언트를 제대로 Backward 하지 못해 발생하는 에러다. 공부를 시작하고 정말 처음 마주하는 에러라서 정말 많이 당황했다. 래퍼런스 자료 역시 거의 없어서 해결하는데 애를 먹었던 쓰라린 사연이 있는 에러다. 이 글을 읽는 독자라면 대부분 텐서의 계산...","categories": ["Framework & Library"],
        "tags": ["Pytorch","CUDA","Error Handling"],
        "url": "/framework-library/inf-per-device",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "🎲 RuntimeError: CUDA error: device-side assert triggered",
        "excerpt":"😵 사전에 정의 입출력 차원 ≠ 실제 입출력 차원 다양한 원인이 있다고 알려져 있는 에러지만, 필자의 경우 위 에러는 사전에 정의한 데이터의 입출력 차원과 실제 입출력 데이터 차원이 서로 상이할 때 발생했다. 하지만 원인을 확실히 특정하고 싶다면 아래 예시 코드를 먼저 추가한 뒤, 다시 한 번 에러 로그를 확인해보길 권장한다....","categories": ["Framework & Library"],
        "tags": ["Pytorch","Dimension Mismatch","CUDA","Error Handling"],
        "url": "/framework-library/mismatch-dimension",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "🎲 RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(hand≤)",
        "excerpt":"😵 nn.Embedding 차원 ≠ 실제 데이터 입력 차원 torch.nn.Embedding에서 정의한 입출력 차원과 실제 데이터의 차원이 다른 경우에 발생하는 에러다. 다양한 상황에서 마주할 수 있는 에러지만, 필자의 경우 Huggingface에서 불러온pretrained tokenizer에 special token 을 추가해 사용할 때, 토큰을 추가했다는 사실을 잊고 nn.Embedding 에 정의한 입출력 차원을 변경하지 않아서 발생하는 경우가 많았다....","categories": ["Framework & Library"],
        "tags": ["Pytorch","Dimension Mismatch","nn.Embedding","CUDA","Error Handling"],
        "url": "/framework-library/mismatch-embedding",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "🔢 Vector Space: Linear Independent, Span, Sub-space, Column Space, Rank, Basis, Null Space",
        "excerpt":"🔢 Linear Independent 기저에 대해 알기 위해서는 먼저 linear independent(선형 독립)의 의미를 알아야 한다. 선형독립이란, 왼쪽 그림처럼 서로 다른 벡터들이 관련성 없이 독립적으로 존재하는 상태를 말한다. 따라서 서로 다른 두 벡터가 선형 독립이라면 한 벡터의 선형결합(조합)으로 다른 벡터를 표현할 수 없다. 반대로 선형 종속 상태면 오른쪽 그림처럼 벡터를 다른 벡터의...","categories": ["Linear Algebra"],
        "tags": ["Linear Algebra","linear independent","span","sub-space","vector space","rank","column space","null space","basis"],
        "url": "/linear-algebra/vector-subspace",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "🌆 [ViT] An Image Is Worth 16x16 Words: Transformers For Image Recognition At Scale",
        "excerpt":"🔭 Overview 시작하기 앞서, 본 논문 리뷰를 수월하게 읽으려면 Transformer 에 대한 선이해가 필수적이다. 아직 Transformer 에 대해서 잘 모른다면 필자가 작성한 포스트를 읽고 오길 권장한다. 또한 본문 내용을 작성하면서 참고한 논문과 여러 포스트의 링크를 맨 밑 하단에 첨부했으니 참고 바란다. 시간이 없으신 분들은 중간의 코드 구현부를 생략하고 Insight 부터 읽기를...","categories": ["Computer Vision"],
        "tags": ["Computer Vision","Vision Transformer","ViT","Transformer","Self-Attention","Image Classification"],
        "url": "/cv/vit",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "🪢 [DeBERTa] DeBERTa: Decoding-Enhanced BERT with Disentangled-Attention",
        "excerpt":"🔭 Overview DeBERTa는 2020년 Microsoft가 ICLR에서 발표한 자연어 처리용 신경망 모델이다. Disentangled Self-Attention, Enhanced Mask Decoder라는 두가지 새로운 테크닉을 BERT, RoBERTa에 적용해 당시 SOTA를 달성했으며, 특히 영어처럼 문장에서 자리하는 위치에 따라 단어의 의미, 형태가 결정되는 굴절어 계열에 대한 성능이 좋아 꾸준히 사랑받고 있는 모델이다. 또한 인코딩 가능한 최대 시퀀스 길이가 4096으로...","categories": ["NLP"],
        "tags": ["Natural Language Process","DeBERTa","BERT","RoBERTa","Transformer","Self-Attention","Disentangled-Attention","Relative Position Embedding","EMD","Encoder"],
        "url": "/nlp/deberta",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "🤖 [Transformer] Attention Is All You Need",
        "excerpt":"🔭 Overview Transformer는 2017년 Google이 NIPS에서 발표한 자연어 처리용 신경망으로 기존 RNN 계열(LSTM, GRU) 신경망이 가진 문제를 해결하고 최대한 인간의 자연어 이해 방식을 수학적으로 모델링 하려는 의도로 설계 되었다. 이 모델은 초기 Encoder-Decoder 를 모두 갖춘 seq2seq 형태로 고안 되었으며, 다양한 번역 테스크에서 SOTA를 달성해 주목을 받았다. 이후에는 여러분도 잘 아시는...","categories": ["NLP"],
        "tags": ["Natural Language Process","Transformer","Self-Attention","Seq2Seq","Encoder","Decoder"],
        "url": "/nlp/transformer",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👨‍💻🐍 [Python] Object Attribute & Assertion Function",
        "excerpt":"🧧 Attribute Function 이번 포스팅은 Python 코드를 작성하면서 객체와 내부 메서드에 관련한 처리가 필요할 때 가장 많이 사용하게 되는 getattr, setattr , delattr , hasttr 함수들의 사용법에 대해 다뤄보려 한다. 특히 getattr, setattr 의 경우 머신러닝 혹은 딥러닝 관련 코드를 읽다가 심심치 않게 찾아볼 수 있다. 모델의 hyper-parameter를 튜닝하거나 기타...","categories": ["Python"],
        "tags": ["Python","Object","Attribute","Assertion","ML","Deep Learning"],
        "url": "/python/attribute_function",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👩‍💻🔠 [baekjoon] 1120번: 문자열",
        "excerpt":"🖍️ solution import sys \"\"\" [시간] 1) 22:10 ~ 22:32 [요약] 1) 두 문자열 X와 Y의 차이: X[i] ≠ Y[i]인 i의 개수 - X=”jimin”, Y=”minji”이면, 둘의 차이는 4 2) A ≤ B, 두 문자열의 길이가 똑같아 지도록 아래 연산 선택 - A의 앞에 아무 알파벳이나 추가한다. - A의 뒤에 아무 알파벳이나...","categories": ["Problem Solving"],
        "tags": ["Python","Codeing Test","Algorithm","Baekjoon","String Handle"],
        "url": "/ps/baekjoon-1120",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👩‍💻🔠 [baekjoon] 1427번: 소트인사이드",
        "excerpt":"🖍️ solution import sys from collections import Counter \"\"\" [시간] 1) 23:50 ~ 24:03 [요약] 1) 수의 각 자리수를 내림차순 - 2143: 4321 [전략] 1) 입력 받는 숫자를 split으로 잘라서 다시 sort 해야지 - split, Counter, sort 같이 사용하면 될 듯 \"\"\" n = list(sys.stdin.readline().rstrip()) count = Counter(n) tmp_result = sorted(count.elements(),...","categories": ["Problem Solving"],
        "tags": ["Python","Codeing Test","Algorithm","Baekjoon","String Handle"],
        "url": "/ps/baekjoon-1427",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👩‍💻🔠 [baekjoon] 1316번: 그룹 단어 체커",
        "excerpt":"🖍️ solution import sys \"\"\" [풀이 시간] 1) 16:30 ~ 17:50 [요약] 1) 그룹 문자: ccazzzzbb, kin - 아닌 경우: aabbbccb (b가 혼자 떨어져 있기 때문에 그룹 문자열이 아님) \"\"\" N = int(sys.stdin.readline()) result = N for i in range(N): word_set = {1} word = list(sys.stdin.readline().rstrip()) for j in range(len(word)): if...","categories": ["Problem Solving"],
        "tags": ["Python","Codeing Test","Algorithm","Baekjoon","String Handle"],
        "url": "/ps/baekjoon-1316",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👩‍💻🔠 [baekjoon] 9012번: 괄호",
        "excerpt":"🖍️ solution import sys \"\"\" [풀이 시간] 1) 15:45 ~ 16:15 \"\"\" for i in range(int(sys.stdin.readline())): left, right, checker = 0, 0, False ps = list(sys.stdin.readline().rstrip()) for j in ps: if j == '(': left += 1 else: right += 1 if right &gt; left: checker = True break if checker:...","categories": ["Problem Solving"],
        "tags": ["Python","Codeing Test","Algorithm","Baekjoon","String Handle"],
        "url": "/ps/baekjoon-9012",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👩‍💻🔠 [baekjoon] 1254번: 팰린드롬 만들기",
        "excerpt":"🖍️ solution import sys \"\"\" [풀이 시간] 1) 17:00 ~ 17:30 [요약] 1) 규완이가 적어놓고 간 문자열 S에 0개 이상의 문자를 문자열 뒤에 추가해서 팰린드롬을 만들려고 한다. - 가능한 짧은 문자열을 추가해 펠린드롬을 만들고 싶음 [전략] 1) 그냥 무식 단순 루프 돌리기 \"\"\" text = sys.stdin.readline().rstrip() result, slicer = 99999, 1...","categories": ["Problem Solving"],
        "tags": ["Python","Codeing Test","Algorithm","Baekjoon","String Handle"],
        "url": "/ps/baekjoon-1254",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👩‍💻🔠 [baekjoon] 14425번: 문자열 집합",
        "excerpt":"🖍️ solution import sys \"\"\" [풀이 시간] 1) 16:30 ~ 16:50 [요약] 1) N개의 문자열로 이루어진 집합 S가 주어진다. - 입력으로 주어지는 M개의 문자열 중에서 집합 S에 포함되어 있는 것이 총 몇 개인지 구하는 프로그램 작성 [전략] 1) 세트 교차 방식 (시간 효율성 GOOD) - 집합 S에 중복 문자열은 없지만, M개의...","categories": ["Problem Solving"],
        "tags": ["Python","Codeing Test","Algorithm","Baekjoon","String Handle"],
        "url": "/ps/baekjoon-14425",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👩‍💻🔠 [baekjoon] 1764번: 듣보잡",
        "excerpt":"🖍️ solution 1 import sys \"\"\" [풀이 시간] 1) 15:50 ~ 16:10 [요약] 1) 명단 A, 명단 B의 교집합 구하는 문제 [전략] 1) 두 명단을 세트 자료형에 넣고 교집합을 구해주기 \"\"\" N, M = map(int, sys.stdin.readline().split()) set_a, set_b = set(), set() # 듣도 못한 사람 명단 for _ in range(N): set_a.add(sys.stdin.readline().rstrip())...","categories": ["Problem Solving"],
        "tags": ["Python","Codeing Test","Algorithm","Baekjoon","String Handle"],
        "url": "/ps/baekjoon-1764",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👩‍💻🔠 [baekjoon] 5430번: AC",
        "excerpt":"🖍️ solution import sys from collections import deque \"\"\" [시간] 1) 18:25 ~ 18:55 [요약] 1) 새로운 언어 AC: AC는 정수 배열에 연산을 하기 위해 만든 언어 - R(뒤집기): 배열에 있는 수의 순서를 뒤집는 함수 =&gt; reversed - D(버리기): D는 첫 번째 수를 버리는 함수 =&gt; queue 2) 특정 동작을 의미하는...","categories": ["Problem Solving"],
        "tags": ["Python","Codeing Test","Algorithm","Baekjoon","String Handle"],
        "url": "/ps/baekjoon-5430",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👩‍💻🔠 [baekjoon] 12891번: DNA 비밀번호",
        "excerpt":"🖍️ solution import sys from collections import Counter, deque \"\"\" [시간] 1) 21:30 ~ 22:00 [요약] 1) DNA 문자열: A, C, G, T로만 구성된 문자열 =&gt; DNA 문자열의 일부를 뽑아 비밀번호로 사용 =&gt; 추출 기준은 서로 다른 문자의 개수가 특정 개수 이상 등장해야 함 =&gt; 만들 수 있는 비밀번호 종류, 추출된...","categories": ["Problem Solving"],
        "tags": ["Python","Codeing Test","Algorithm","Baekjoon","String Handle"],
        "url": "/ps/baekjoon-12891",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👩‍💻🔠 [baekjoon] 1969번: DNA",
        "excerpt":"🖍️ solution import sys from collections import Counter \"\"\" [시간] 1) 20:00 ~ 20:30 [요약] 1) DNA를 이루는 뉴클레오티드의 첫글자를 따서 표현, 종류는 4가지 - A, T, G, C 2) N개의 길이 M인 DNA가 주어지면 Hamming Distance의 합이 가장 작은 DNA S를 구하기 - Hamming Distance: 각 위치의 뉴클오티드 문자가 다른...","categories": ["Problem Solving"],
        "tags": ["Python","Codeing Test","Algorithm","Baekjoon","String Handle"],
        "url": "/ps/baekjoon-1969",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👩‍💻💵 [baekjoon] 11501번: 주식",
        "excerpt":"🖍️ solution import sys \"\"\" [시간] 1) 14:10 ~ 14:34 [요약] 1) 주식을 '하나' 사기/원하는 만큼 가지고 있는 주식을 팔기/아무것도 안하기 - 날 별로 주식의 가격을 알려주었을 때, 최대 이익이 얼마나 되는지 계산하는 프로그램 작성 [전략] 1) max() 이용해 문제 해결 - max - 현재 ≥ 0: 사기 - max -...","categories": ["Problem Solving"],
        "tags": ["Python","Codeing Test","Algorithm","Baekjoon","String Handle"],
        "url": "/ps/baekjoon-11501",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👩‍💻🔠 [baekjoon] 17609번: 회문",
        "excerpt":"🖍️ solution import sys \"\"\" [시간] 1) 14:20 ~ 14:45 [요약] 1) 유사회문: 한 문자를 삭제하여 회문으로 만들 수 있는 문자열 =&gt; 유사회문인지 아닌지 판단하는 프로그램 작성 2) 주어진 문자열의 길이는 10만, 문자열 개수는 최대 30개 =&gt; 제한 시간이 1초라서 O(n)의 알고리즘을 설계 필요, Counter 사용 불가 [전략] 1) 슬라이싱 이용해서...","categories": ["Problem Solving"],
        "tags": ["Python","Codeing Test","Algorithm","Baekjoon","String Handle"],
        "url": "/ps/baekjoon-17609",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👩‍💻💵 [baekjoon] 1789번: 수들의 합",
        "excerpt":"🖍️ solution import sys \"\"\" [시간] 1) 01:40 ~ 02:10 [요약] 1) S: 서로 다른 N개의 자연수들의 합 =&gt; 이 때, 자연수 N의 최대값 [전략] 1) 자연수 개수가 최대가 되도록 만들 어야 하기 때문에 최대한 작은 수들의 합으로 S를 구성 - 10: 1,2,3,4 =&gt; 4개 \"\"\" S = int(sys.stdin.readline()) # for...","categories": ["Problem Solving"],
        "tags": ["Python","Codeing Test","Algorithm","Baekjoon","String Handle"],
        "url": "/ps/baekjoon-1789",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👩‍💻🔠 [baekjoon] 5052번: 전화번호 목록",
        "excerpt":"🖍️ solution import sys \"\"\" [시간] 1) 15:20 ~ 16:00 [요약] 1) 주어진 전화번호 목록을 보고, 일관성이 여부 판단 - 하나의 번호가 다른 번호의 접두어 X - 주어진 모든 번호에 동일하게 연락할 수 있어야 일관성 있다고 판단 [전략] 1) 전화번호 앞자리를 최우선 기준으로 정렬 - 시간 제한 &amp; 입력의 길이: 이중...","categories": ["Problem Solving"],
        "tags": ["Python","Codeing Test","Algorithm","Baekjoon","String Handle"],
        "url": "/ps/baekjoon-5052",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👩‍💻🔭 [baekjoon] 1962번: 그림",
        "excerpt":"🖍️ solution import sys from collections import deque from typing import List \"\"\" [시간] 1) 16:50 ~ 17:20 [요약] 1) 큰 도화지에 그림이 그려져 있을 때, 그 그림의 개수와, 그 그림 중 넓이가 가장 넓은 것의 넓이를 출력 - 영역 구분 및 넓이가 가장 큰 영역의 넓이 구하는 프로그램 작성 -...","categories": ["Problem Solving"],
        "tags": ["Python","Codeing Test","Algorithm","Baekjoon","String Handle"],
        "url": "/ps/baekjoon-1962",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👨‍💻🐍 [Python] List & Tuple",
        "excerpt":"🗂️ Concept of Array in Python C, C++, Java 같은 언어를 배울 때 가장 먼저 배우는 자료구조는 바로 배열이다. 그러나 파이썬을 배울 때는 조금 양상이 다르다. 배열이라는 표현의 자료구조는 언급도 없고 리스트, 튜플, 딕셔너리와 같은 형태의 자료구조에 대해서만 배우게 된다. 그렇다면 파이썬에 배열은 없는 것일까?? 반은 맞고 반은 틀린 질문이라고 할...","categories": ["Python"],
        "tags": ["Python","array","list","tuple","list comprehension","CS"],
        "url": "/python/list_tuple",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👨‍💻🐍 [Python] Function Argument",
        "excerpt":"👨‍👩‍👧‍👦 Function Argument 파이썬의 모든 메서드는 기본적으로 인자를 call by value 형태로 전달해야 한다. 하지만 call by value 라고 해서 함수의 동작과 원본 변수가 완전히 독립적인 것은 아니다. 이것은 인자로 어떤 데이터 타입을 전달하는가에 따라 달라진다. 만약 인자로 mutable(dynamic) 객체인 리스트 변수를 전달했다면, 함수의 동작에 따른 결과가 그대로 변수에 반영된다. mutable...","categories": ["Python"],
        "tags": ["Python","Function","Argument","mutable","CS"],
        "url": "/python/func_argu",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👩‍💻🎄 [baekjoon] 15558번: 점프 게임",
        "excerpt":"🖍️ solution import sys from collections import deque from typing import List def bfs(y: int, x: int): time, flag = -1, False q = deque([[y, x]]) while q: for _ in range(len(q)): vy, vx = q.popleft() if vx+1 &gt;= N or vx+K &gt;= N: flag = True break if graph[vy][vx+1] and...","categories": ["Problem Solving"],
        "tags": ["Python","Codeing Test","Algorithm","Baekjoon","Graph","BFS"],
        "url": "/ps/baekjoon-15558",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👩‍💻🎄 [baekjoon] 16920번: 확장게임",
        "excerpt":"🖍️ solution import sys from collections import deque def solution(): N, M, P = map(int, sys.stdin.readline().split()) scores = [0] * (P + 1) dy = [0, 0, 1, -1] # direction of search dx = [1, -1, 0, 0] p_list = [0] + list(map(int, sys.stdin.readline().split())) # for matching index with player...","categories": ["Problem Solving"],
        "tags": ["Python","Codeing Test","Algorithm","Baekjoon","Graph","BFS"],
        "url": "/ps/baekjoon-16920",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "🗂️ Graph Theory 2: Dijkstra",
        "excerpt":"📚 Dijkstra 다익스트라 최단 경로 문제는 그래프 자료 구조에서 여러 개의 노드가 주어졌을 때, 특정한 노드(시작점)에서 특정한 노드(도착점)까지의 최단 경로를 구해주는 알고리즘을 설계해야 한다. 특히 다익스트라는 음의 간선이 없을 때 정상적으로 동작하며, 유향 &amp; 무향을 가리지 않고 적용할 수 있다. 다익스트라 알고리즘의 동작을 기술하면 아래와 같다. 1) 출발 노드 설정...","categories": ["Algorithm"],
        "tags": ["Python","Codeing Test","Algorithm","Dijkstra"],
        "url": "/algorithm/dijkstra",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "🗂️ Graph Theory 3: Floyd-Warshall",
        "excerpt":"📚 Floyd-Warshall Floyd-Warshall은 모든 지점에서 다른 모든 지점까지의 최단 경로를 구하는 알고리즘이다. 지정된 출발점에서 나머지 다른 지점가지의 최단 경로를 구하는 다익스트라 알고리즘과는 차이가 있다. 따라서 솔루션을 도출하는 방식에도 살짝 차이가 생기는데, Floyd-Warshall 은 그리디하게 매번 최단 경로에 있는 노드를 구할 필요가 없다. 이유는 모든 지점에서 다른 모든 지점까지의 경로를 구해야...","categories": ["Algorithm"],
        "tags": ["Python","Codeing Test","Algorithm","Floyd-Warshall"],
        "url": "/algorithm/floyd-warshell",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "🗂️ Graph Theory 5: MST with Kruskal & Prim",
        "excerpt":"🎡 Spanning Tree 그래프 내부에 포함된 모든 노드를 포함하는 트리를 의미한다. 모든 정점을 포함하긴 하지만 근본은 트리라서 사이클이 발생하면 안되며, 최소의 간선을 사용해 모든 노드를 연결해야 한다. 따라서 Spanning Tree 의 간선 개수는 노드 개수-1에 해당한다. 💵 Minimum Spanning Tree 그래프 상에서 발생할 수 있는 여러 Spanning Tree 중에서 간선들의 가중치 합이...","categories": ["Algorithm"],
        "tags": ["Python","Codeing Test","Algorithm","MST","Kruskal","Prim"],
        "url": "/algorithm/mst",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "✏️  Summary of Useful Library for Coding Test",
        "excerpt":"📚 collections 🪢 deque python에서 stack이나 queue 자료형을 구현할 때 일반적으로 사용하는 내장 라이브러리 collections에 구현된 클래스다. 메서드가 아닌 객체라서 사용하려면 초기화가 필요하다. 사용 예시를 보자. # collections.deque usage example deque([iterable[, maxlen]]) --&gt; deque object &gt;&gt;&gt; from collections import deque, Counter &gt;&gt;&gt; queue = deque() # 1) &gt;&gt;&gt; queue deque([]) &gt;&gt;&gt; queue...","categories": ["Algorithm"],
        "tags": ["Python","collections","Codeing Test","Algorithm"],
        "url": "/algorithm/useful_library",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "🗂️ Graph Theory 4: Union-Find (Disjoint Set)",
        "excerpt":"🙅 Disjoint Set 서로 공통된 원소를 가지고 있지 않은 여러 집합들을 지칭하는 용어다. 개별 원소가 정확히 하나의 집합에 속하며, 어떤 집합도 서로 공통 원소를 가지고 있지 않아야 한다. 서로소 집합 자료구조를 사용하면 서로 다른 원소들이 같은 집합군에 속해 있는가 판별하는 것과 같은 작업을 쉽게 할 수 있다. 그렇다면 이제부터 자료구조로서 서로소...","categories": ["Algorithm"],
        "tags": ["Python","Codeing Test","Algorithm","Union-Find"],
        "url": "/algorithm/union-find",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "🗂️ Convex Optimization Problem",
        "excerpt":"❓ Convex Optimization Problem \\[f(wx_1 + (1-w)x_2)≤ wf(x_1) + (1-w)f(x_2),\\ \\ w \\in [0,1] \\\\ f''(x) ≥ 0\\] Convex Problem 이란, 목적 함수 $f(x)$가 Convex Function 이면서 Feasible Set 역시 Convex Set 이 되는 문제 상황을 일컫는다. Convex Problem 는 수학적 최적화에서 매우 중요한 개념인데, 그 이유는 해당 조건을 만족하면 국소...","categories": ["Optimization Theory"],
        "tags": ["Optimization Theory","Convex Optimization"],
        "url": "/optimization-theory/convex",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "🍎 Newton-Raphson Method for Optimization",
        "excerpt":"🤔 Zero-Find Ver 비선형 방정식의 근사해를 찾거나 최적화 문제를 해결하는 방식으로, 같은 과정을 반복해 최적값에 수렴한다는 점에서 경사하강법이랑 근본이 같다. 반면, 경사하강에 비해 빠른 수렴 속도를 자랑하고 풀이 방식이 매우 간단하다는 장점이 있다. 하지만 여러 제약 조건과 더불어 해당 알고리즘이 잘 작동하는 상황이 비현실적인 부분이 많아 경사하강에 비해 자주 사용되지는 않고...","categories": ["Optimization Theory"],
        "tags": ["Optimization Theory","Newton-Raphson"],
        "url": "/optimization-theory/newton-raphson",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "🔢 Eigen Decomposition",
        "excerpt":"고유값, 고유벡터, 고유값 분해는 비단 선형대수학뿐만 아니라 해석기하학 나아가 데이터 사이언스 전반에서 가장 중요한 개념 중 하나라고 생각한다. 머신러닝에서 자주 사용하는 여러 행렬 분해(Matrix Factorization) 기법(ex: SVD)과 PCA의 이론적 토대가 되므로 반드시 완벽하게 숙지하고 넘어가야 하는 파트다. 이번 포스팅 역시 혁펜하임님의 선형대수학 강의와 공돌이의 수학정리님의 강의 및 포스트 그리고 딥러닝을...","categories": ["Linear Algebra"],
        "tags": ["Linear Algebra","Eigen Decomposition","Eigen Vector","Eigen Value","SVD","PCA"],
        "url": "/linear-algebra/eigen-decomposition",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "📈 Gradient: Directional Derivative",
        "excerpt":"🤔 Concept of Gradient 그라디언트는 다변수 함수의 기울기를 나타내는 벡터를 말한다. 그라디언트의 원소는 함수에 존재하는 모든 변수를 대상으로 편미분한 결과로 구성되는데, 예를 들어 변수가 $x_1, x_2$ 2개인 다변수 함수 $f(x_1, x_2)$가 있다고 가정해보자. 다변수 함수 $f$의 그라디언트는 아래 수식처럼 표현할 수 있다. \\[f'(x_1, x_2) = \\begin{vmatrix} \\frac{∂f}{∂x_1} \\\\ \\frac{∂f}{∂x_2} \\end{vmatrix}\\] 이러한...","categories": ["Optimization Theory"],
        "tags": ["Optimization Theory","Calculus","Partial Derivative","Total Derivative","loss function","Gradient","Gradient Descent","Machine Learning"],
        "url": "/optimization-theory/gradient",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "🗄️ SVD: Singular Value Decomposition",
        "excerpt":"특이값 분해는 고유값 분해를 일반적인 상황으로 확장시킨 개념으로 LSA(Latent Semantic Anaylsis), Collaborative Filtering과 같은 머신러닝 기법에 사용되기 때문에 자연어처리, 추천시스템에 관심이 있다면 반드시 이해하고 넘어가야 하는 중요한 방법론이다. 혁펜하임님의 선형대수학 강의와 공돌이의 수학정리님의 강의 및 포스트 그리고 딥러닝을 위한 선형대수학 교재을 참고하고 개인적인 해석을 더해 정리했다. 🌟 Concept of SVD \\[A...","categories": ["Linear Algebra"],
        "tags": ["Linear Algebra","Singular Value Decomposition","Singular Vector","Singular Value","SVD","PCA"],
        "url": "/linear-algebra/svd",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "📈 Chain Rule: 합성함수 미분법",
        "excerpt":"Chain Rule 이라고 불리기도 하는 합성함수 미분법은 미적분학에서 특히나 중요한 개념 중 하나다. 근래에는 신경망을 활용한 딥러닝이 주목받으면서 그 중요성이 더욱 부각되고 있다. 신경망 모델은 쉽게 생각하면 정말 많은 1차함수와 여러 활성함수를 합성한 것과 같기 때문이다. 따라서 오차 역전을 통해 가중치를 최적화 하는 과정을 정확히 이해하려면 합성함수 미분법에 대한 이해는...","categories": ["Optimization Theory"],
        "tags": ["Calculus"],
        "url": "/optimization-theory/chain-rule",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "🔢 Product & Quotient Rule: 곱의 미분, 몫의 미분",
        "excerpt":"곱의 미분, 몫의 미분은 함수가 곱의 꼴 형태 $f(x)g(x)$ 혹은 분수 꼴 형태 $\\frac{f(x)}{g(x)}$를 가지고 있을 때 도함수를 구하는 방법이다. 고등학교 미적분 시간(17~18학번 기준)에 배운적이 있지만, 합성함수 미분법과 더불어 단순 암기의 폐해로 까먹기 좋은 미분법들이다. 크로스 엔트로피, 소프트맥스 미분에 쓰이므로 합성함수 미분법과 마찬가지로 딥러닝, 머신러닝에서 매우 중요하다. ✖️ Product Rule 몫의...","categories": ["Optimization Theory"],
        "tags": ["Calculus","Product Rule","Quotient Rule"],
        "url": "/optimization-theory/product_quotient_rule",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "🔥 Pytorch Tensor Indexing 자주 사용하는 메서드 모음집",
        "excerpt":"파이토치에서 필자가 자주 사용하는 텐서 인덱싱 관련 메서드의 사용법 및 사용 예시를 한방에 정리한 포스트다. 메서드 하나당 하나의 포스트로 만들기에는 너무 길이가 짧다 생각해 한 페이지에 모두 넣게 되었다. 지속적으로 업데이트 될 예정이다. 또한 텐서 인덱싱 말고도 다른 주제로도 관련 메서드를 정리해 올릴 예정이니 많은 관심 부탁드린다. 🔎 torch.argmax 입력 텐서에서...","categories": ["Framework & Library"],
        "tags": ["Pytorch","Tensor","Linear Algebra"],
        "url": "/framework-library/torch-indexing-function",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👩‍💻🎄 [baekjoon] 1987번: 알파벳",
        "excerpt":"🖍️ solution import sys from typing import List def backtracking(y: int, x: int, count: int, visit: List, graph: List[List]): global result visit[ord(graph[y][x]) - 65] = True result.add(count) for i in range(4): ny, nx = dy[i] + y, dx[i] + x if -1 &lt; ny &lt; r and -1 &lt; nx &lt;...","categories": ["Problem Solving"],
        "tags": ["Python","Codeing Test","Algorithm","Baekjoon","Graph","DFS","BackTracking"],
        "url": "/ps/baekjoon-1987",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "🪢 [DeBERTa-V3] DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing",
        "excerpt":"🔭 Overview 2021년 Microsoft에서 공개한 DeBERTa-V3은 기존 DeBERTa의 모델 구조는 그대로 유지하되, ELECTRA의 Generator-Discriminator 구조를 차용하여 전작 대비 성능을 향상 시킨 모델이다. ELECTRA에서 BackBone 모델로 BERT 대신 DeBERTa을 사용했다고 생각하면 된다. 거기에 더해 ELECTRA의 Tug-of-War 현상을 방지하기 위해 새로운 임베딩 공유 기법인 GDES(Gradient Disentagnled Embedding Sharing)방법을 제시했다. 이번 포스팅에서는 구현 코드와...","categories": ["NLP"],
        "tags": ["Natural Language Process","DeBERTa-V3","DeBERTa","ELECTRA","Weight Sharing","GDES","Pytorch"],
        "url": "/nlp/deberta_v3",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "🧑‍🏫 [DistilBERT] DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter",
        "excerpt":"🔭 Overview DistilBERT 는 허깅 페이스 연구진이 2019년 발표한 BERT의 변형으로서, On-Device Ai 개발을 목표로 경량화에 초점을 맞춘 모델이다. GPT, BERT의 등장 이후, NLP 분야에서 비약적인 성능 향상이 이뤄졌음에도 불구하고, 터무니 없는 모델 사이즈와 컴퓨팅 리소스 요구로 인해 실생활 적용 같은 활용성은 여전히 해결해야할 문제로 남아 있었다. Google에서 발표한 초기 BERT-base-uncased...","categories": ["NLP"],
        "tags": ["Natural Language Process","DistilBERT","BERT","Self-Attention","Pytorch"],
        "url": "/nlp/distilbert",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👮 [ELECTRA] Pre-training Text Encoders as Discriminators Rather Than Generators",
        "excerpt":"🔭 Overview ELECTRA는 2020년 Google에서 처음 발표한 모델로, GAN(Generative Adversarial Networks) Style 아키텍처를 NLP에 적용한 것이 특징이다. 새로운 구조 차용에 맞춰서 RTD(Replace Token Dection) Task를 고안에 사전 학습으로 사용했다. 모든 아이디어는 기존 MLM(Masked Language Model)을 사전학습 방법론으로 사용하는 인코더 언어 모델(BERT 계열)의 단점으로부터 출발한다. [MLM 단점] 1) 사전학습과 파인튜닝 사이 불일치...","categories": ["NLP"],
        "tags": ["Natural Language Process","ELECTRA","BERT","GAN","Transformer","Self-Attention","Pytorch"],
        "url": "/nlp/electra",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "🎡 [Roformer] RoFormer: Enhanced Transformer with Rotary Position Embedding",
        "excerpt":"🔭 Overview Roformer는 2021년에 발표된 트랜스포머 모델의 변형으로, RoPE(Rotary Position Embedding)이라는 새로운 위치 정보 포착 방식을 제안했다. 근래 유명한 오픈소스 LLM 모델들(GPT-Neo, LLaMA)의 위치 정보 포착 방식으로 채택 되어 주목을 받고 있다. RoPE 기법에 대해 살펴보기 전에 일단, 관련 분야의 연구 동향 및 위치 정보의 개념에 대해 간단하게 살펴보고 넘어가려 한다....","categories": ["NLP"],
        "tags": ["Natural Language Process","Roformer","Transformation Matrix","Complex Space","Self-Attention","Linear-Attention","Pytorch"],
        "url": "/nlp/roformer",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "🗂️[SpanBERT] SpanBERT: Improving Pre-training by Representing and Predicting Spans",
        "excerpt":"🔭 Overview SpanBERT는 2020년 페이스북에서 발표한 BERT 계열 모델로, 새로운 방법론인 SBO(Span Boundary Objective)를 고안해 사전학습을 하여 기존 대비 높은 성능을 기록했다. 기존 MLM, CLM은 단일 토큰을 예측하는 방식을 사용하기 때문에 Word-Level Task에 아주 적합하지만 상대적으로 QA, Sentence-Similarity 같은 문장 단위 테스크에 그대로 활용하기에는 부족한 점이 있었다. 이러한 문제를 해결하기 위해...","categories": ["NLP"],
        "tags": ["Natural Language Process","SpanBERT","BERT","Self-Attention","Pytorch"],
        "url": "/nlp/spanbert",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "🌆 [Linear Attention] Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention",
        "excerpt":"🔭 Overview DistilBERT 는 허깅 페이스 연구진이 2019년 발표한 BERT의 변형으로서, On-Device Ai 개발을 목표로 경량화에 초점을 맞춘 모델이다. GPT, BERT의 등장 이후, NLP 분야에서 비약적인 성능 향상이 이뤄졌음에도 불구하고, 터무니 없는 모델 사이즈와 컴퓨팅 리소스 요구로 인해 실생활 적용 같은 활용성은 여전히 해결해야할 문제로 남아 있었다. Google에서 발표한 초기 BERT-base-uncased...","categories": ["NLP"],
        "tags": ["Natural Language Process","Linear-Attention","Transformer","BERT","Kernel Trick","Self-Attention","Pytorch"],
        "url": "/nlp/linear_attention",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👨⏰🐍 [Python] 시간복잡도 1",
        "excerpt":"Memeory 1) 232 ⇒ 4GB** 2) 216 ⇒ 64MB** Time 구체적인 성능은 플랫폼의 하드웨어에 따라서 달라지겠지만, 일반적으로 1초에 1억번 정도 계산할 수 있다고 가정하고 알고리즘의 시간 복잡도를 계산하면 된다. 즉, 어떤 문제의 시간 제한이 2초라면, 2억번 이하의 계산을 하는 알고리즘의 경우는 통과로 처리된다는 것이다. 시간 복잡도는 원래 데이터 개수에 따라서...","categories": ["Python"],
        "tags": ["Python","Function","Argument","mutable","CS"],
        "url": "/python/time_complexity1",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👨⏰🐍 [Python] 시간복잡도 2",
        "excerpt":"Theme 1. 입력 모듈 \"\"\" Compare to Input module \"\"\" import sys N = int(inputs()) K = int(sys.stdin.readline()) 파이썬에서 사용자로부터 입력을 받는 모듈은 보통 inputs(), sys.stdin.readline() 을 사용한다. inputs() 는 입력 받는 데이터의 길이가 길고, 많아질수록 입력 효율이 떨어지는 단점이 있다. 그래서 이를 보완하기 위해 대부분의 코딩테스트 환경에서 입력을 받을...","categories": ["Python"],
        "tags": ["Python","Time Complexity","CS"],
        "url": "/python/time_complexity2",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "🔪 [LoRA] Low-Rank Adaptation of Large Language Models",
        "excerpt":"🔭 Overview LoRA LoRA는 2021년 MS 연구진이 발표한 논문으로 원본(Full 파인튜닝)과 거의 유사한 성능(심지어 일부 벤치마크는 더 높음)으로 LLM 파인튜닝에 필요한 GPU 메모리를 획기적으로 줄이는데 성공해 주목을 받았다. 커뮤니티에서 LoRA is All You Need 라는 별명까지 얻으며 그 인기를 구가하고 있다. DistilBERT 리뷰에서도 살펴보았듯, BERT와 GPT의 등장 이후, 모든 NLP 도메인에서...","categories": ["NLP"],
        "tags": ["Natural Language Process","LoRA","Low-Rank Adaptation","Fine-Tune","Optimization","Pytorch","Huggingface","PEFT"],
        "url": "/nlp/lora",
        "teaser": "/assets/images/huggingface_emoji.png"
      }]
