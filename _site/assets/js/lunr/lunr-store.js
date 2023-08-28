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
        "title": "🔢 Vector Space: Column Space, Basis, Rank, Null Space",
        "excerpt":"🔢 Column Space \\[C(A) = Range(A)\\] 열벡터가 span하는 공간을 의미한다. span 이란, 벡터의 집합에 의해 생성된 모든 linear combination의 결과로 생성할 수 있는 부분 공간을 말한다. 따라서 column space 는 열벡터의 linear combination 결과로 생성할 수 있는 vector space의 부분 공간을 말한다. 🍖 Basis 기저에 대해 알기 위해서는 먼저 linear...","categories": ["Linear Algebra"],
        "tags": ["Linear Algebra","linear independent","vector space","rank","column space","null space","basis"],
        "url": "/linear-algebra/vector-subspace",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "🌆 [ViT] An Image Is Worth 16x16 Words: Transformers For Image Recognition At Scale",
        "excerpt":"🔭 Overview 시작하기 앞서, 본 논문 리뷰를 수월하게 읽으려면 Transformer 에 대한 선이해가 필수적이다. 아직 Transformer 에 대해서 잘 모른다면 필자가 작성한 포스트를 읽고 오길 권장한다. 또한 본문 내용을 작성하면서 참고한 논문과 여러 포스트의 링크를 맨 밑 하단에 첨부했으니 참고 바란다. 시간이 없으신 분들은 중간의 코드 구현부를 생략하고 Insight 부터 읽기를...","categories": ["Computer Vision"],
        "tags": ["Computer Vision","Vision Transformer","ViT","Transformer","Self-Attention","Image Classification"],
        "url": "/cv/vit",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "📈 Gradient: Directional Derivative",
        "excerpt":"🤔 Concept of Gradient 그라디언트는 다변수 함수의 기울기를 나타내는 벡터를 말한다. 그라디언트의 원소는 함수에 존재하는 모든 변수를 대상으로 편미분한 결과로 구성되는데, 예를 들어 변수가 $x_1, x_2$ 2개인 다변수 함수 $f(x_1, x_2)$가 있다고 가정해보자. 다변수 함수 $f$의 그라디언트는 아래 수식처럼 표현할 수 있다. \\[f'(x_1, x_2) = \\begin{vmatrix} \\frac{∂f}{∂x_1} \\\\ \\frac{∂f}{∂x_2} \\end{vmatrix}\\] 이러한...","categories": ["Calculus"],
        "tags": ["Calculus","Partial Derivative","Total Derivative","loss function","Gradient","Gradient Descent","Machine Learning"],
        "url": "/calculus/gradient",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "🔥 Pytorch Tensor Indexing 자주 사용하는 메서드 모음집",
        "excerpt":"파이토치에서 필자가 자주 사용하는 텐서 인덱싱 관련 메서드의 사용법 및 사용 예시를 한방에 정리한 포스트다. 메서드 하나당 하나의 포스트로 만들기에는 너무 길이가 짧다 생각해 한 페이지에 모두 넣게 되었다. 지속적으로 업데이트 될 예정이다. 또한 텐서 인덱싱 말고도 다른 주제로도 관련 메서드를 정리해 올릴 예정이니 많은 관심 부탁드린다. 🔎 torch.argmax 입력 텐서에서...","categories": ["Framework & Library"],
        "tags": ["Pytorch","Tensor","Linear Algebra"],
        "url": "/framework-library/torch-indexing-function",
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
        "title": "✏️  Summary of Useful Library for Coding Test",
        "excerpt":"📚 collections 🪢 deque python에서 stack이나 queue 자료형을 구현할 때 일반적으로 사용하는 내장 라이브러리 collections에 구현된 클래스다. 메서드가 아닌 객체라서 사용하려면 초기화가 필요하다. 사용 예시를 보자. # collections.deque usage example deque([iterable[, maxlen]]) --&gt; deque object &gt;&gt;&gt; from collections import deque, Counter &gt;&gt;&gt; queue = deque() # 1) &gt;&gt;&gt; queue deque([]) &gt;&gt;&gt; queue...","categories": ["Algorithm"],
        "tags": ["Python","collections","Codeing Test","Algorithm"],
        "url": "/algorithm/useful_library",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👨‍💻🐍 [Python] Object Attribute & Assertion Function",
        "excerpt":"🧧 Attribute Function 이번 포스팅은 Python 코드를 작성하면서 객체와 내부 메서드에 관련한 처리가 필요할 때 가장 많이 사용하게 되는 getattr, setattr , delattr , hasttr 함수들의 사용법에 대해 다뤄보려 한다. 특히 getattr, setattr 의 경우 머신러닝 혹은 딥러닝 관련 코드를 읽다가 심심치 않게 찾아볼 수 있다. 모델의 hyper-parameter를 튜닝하거나 기타...","categories": ["Python"],
        "tags": ["Python","Object","Attribute","Assertion","ML","Deep Learning"],
        "url": "/python/attribute_function",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👩‍💻🔠 [baekjoon] 1120번: 문자열",
        "excerpt":"🖍️ solution import sys \"\"\" [시간] 1) 22:10 ~ 22:32 [요약] 1) 두 문자열 X와 Y의 차이: X[i] ≠ Y[i]인 i의 개수 - X=”jimin”, Y=”minji”이면, 둘의 차이는 4 2) A ≤ B, 두 문자열의 길이가 똑같아 지도록 아래 연산 선택 - A의 앞에 아무 알파벳이나 추가한다. - A의 뒤에 아무 알파벳이나...","categories": ["Algorithm"],
        "tags": ["Python","Codeing Test","Algorithm","Baekjoon","String Handle"],
        "url": "/algorithm/baekjoon-1120",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👩‍💻🔠 [baekjoon] 1427번: 소트인사이드",
        "excerpt":"🖍️ solution import sys from collections import Counter \"\"\" [시간] 1) 23:50 ~ 24:03 [요약] 1) 수의 각 자리수를 내림차순 - 2143: 4321 [전략] 1) 입력 받는 숫자를 split으로 잘라서 다시 sort 해야지 - split, Counter, sort 같이 사용하면 될 듯 \"\"\" n = list(sys.stdin.readline().rstrip()) count = Counter(n) tmp_result = sorted(count.elements(),...","categories": ["Algorithm"],
        "tags": ["Python","Codeing Test","Algorithm","Baekjoon","String Handle"],
        "url": "/algorithm/baekjoon-1427",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👩‍💻🔠 [baekjoon] 1316번: 그룹 단어 체커",
        "excerpt":"🖍️ solution import sys \"\"\" [풀이 시간] 1) 16:30 ~ 17:50 [요약] 1) 그룹 문자: ccazzzzbb, kin - 아닌 경우: aabbbccb (b가 혼자 떨어져 있기 때문에 그룹 문자열이 아님) \"\"\" N = int(sys.stdin.readline()) result = N for i in range(N): word_set = {1} word = list(sys.stdin.readline().rstrip()) for j in range(len(word)): if...","categories": ["Algorithm"],
        "tags": ["Python","Codeing Test","Algorithm","Baekjoon","String Handle"],
        "url": "/algorithm/baekjoon-1316",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👩‍💻🔠 [baekjoon] 9012번: 괄호",
        "excerpt":"🖍️ solution import sys \"\"\" [풀이 시간] 1) 15:45 ~ 16:15 \"\"\" for i in range(int(sys.stdin.readline())): left, right, checker = 0, 0, False ps = list(sys.stdin.readline().rstrip()) for j in ps: if j == '(': left += 1 else: right += 1 if right &gt; left: checker = True break if checker:...","categories": ["Algorithm"],
        "tags": ["Python","Codeing Test","Algorithm","Baekjoon","String Handle"],
        "url": "/algorithm/baekjoon-9012",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👩‍💻🔠 [baekjoon] 1254번: 팰린드롬 만들기",
        "excerpt":"🖍️ solution import sys \"\"\" [풀이 시간] 1) 17:00 ~ 17:30 [요약] 1) 규완이가 적어놓고 간 문자열 S에 0개 이상의 문자를 문자열 뒤에 추가해서 팰린드롬을 만들려고 한다. - 가능한 짧은 문자열을 추가해 펠린드롬을 만들고 싶음 [전략] 1) 그냥 무식 단순 루프 돌리기 \"\"\" text = sys.stdin.readline().rstrip() result, slicer = 99999, 1...","categories": ["Algorithm"],
        "tags": ["Python","Codeing Test","Algorithm","Baekjoon","String Handle"],
        "url": "/algorithm/baekjoon-1254",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👩‍💻🔠 [baekjoon] 14425번: 문자열 집합",
        "excerpt":"🖍️ solution import sys \"\"\" [풀이 시간] 1) 16:30 ~ 16:50 [요약] 1) N개의 문자열로 이루어진 집합 S가 주어진다. - 입력으로 주어지는 M개의 문자열 중에서 집합 S에 포함되어 있는 것이 총 몇 개인지 구하는 프로그램 작성 [전략] 1) 세트 교차 방식 (시간 효율성 GOOD) - 집합 S에 중복 문자열은 없지만, M개의...","categories": ["Algorithm"],
        "tags": ["Python","Codeing Test","Algorithm","Baekjoon","String Handle"],
        "url": "/algorithm/baekjoon-14425",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👩‍💻🔠 [baekjoon] 1764번: 듣보잡",
        "excerpt":"🖍️ solution 1 import sys \"\"\" [풀이 시간] 1) 15:50 ~ 16:10 [요약] 1) 명단 A, 명단 B의 교집합 구하는 문제 [전략] 1) 두 명단을 세트 자료형에 넣고 교집합을 구해주기 \"\"\" N, M = map(int, sys.stdin.readline().split()) set_a, set_b = set(), set() # 듣도 못한 사람 명단 for _ in range(N): set_a.add(sys.stdin.readline().rstrip())...","categories": ["Algorithm"],
        "tags": ["Python","Codeing Test","Algorithm","Baekjoon","String Handle"],
        "url": "/algorithm/baekjoon-1764",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👩‍💻🔠 [baekjoon] 1969번: DNA",
        "excerpt":"🖍️ solution import sys from collections import Counter \"\"\" [시간] 1) 20:00 ~ 20:30 [요약] 1) DNA를 이루는 뉴클레오티드의 첫글자를 따서 표현, 종류는 4가지 - A, T, G, C 2) N개의 길이 M인 DNA가 주어지면 Hamming Distance의 합이 가장 작은 DNA S를 구하기 - Hamming Distance: 각 위치의 뉴클오티드 문자가 다른...","categories": ["Algorithm"],
        "tags": ["Python","Codeing Test","Algorithm","Baekjoon","String Handle"],
        "url": "/algorithm/baekjoon-1969",
        "teaser": "/assets/images/huggingface_emoji.png"
      },{
        "title": "👩‍💻🔠 [baekjoon] 5430번: AC",
        "excerpt":"🖍️ solution import sys from collections import deque \"\"\" [시간] 1) 18:25 ~ 18:55 [요약] 1) 새로운 언어 AC: AC는 정수 배열에 연산을 하기 위해 만든 언어 - R(뒤집기): 배열에 있는 수의 순서를 뒤집는 함수 =&gt; reversed - D(버리기): D는 첫 번째 수를 버리는 함수 =&gt; queue 2) 특정 동작을 의미하는...","categories": ["Algorithm"],
        "tags": ["Python","Codeing Test","Algorithm","Baekjoon","String Handle"],
        "url": "/algorithm/baekjoon-5430",
        "teaser": "/assets/images/huggingface_emoji.png"
      }]
