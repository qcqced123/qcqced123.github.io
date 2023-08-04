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
        "title": "🤖 [Transformer] Attention Is All You Need",
        "excerpt":" ","categories": ["Computer Vision"],
        "tags": ["Computer Vision","Vision Transformer","ViT","Transformer","Self-Attention","Image Classification"],
        "url": "/nlp/vit",
        "teaser": "/assets/images/huggingface_emoji.png"
      }]
