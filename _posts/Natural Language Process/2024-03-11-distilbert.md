---
title: "🌆 [DistilBERT] DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"
excerpt: "DistilBERT Official Paper Review with Pytorch Implementation"
permalink: "/nlp/distilbert"
toc: true  # option for table of contents
toc_sticky: true  # option for table of content
categories:
  - NLP
tags:
  - Natural Language Process
  - DistilBERT
  - BERT
  - Self-Attention
  - Pytorch

last_modified_at: 2024-03-11T12:00:00-05:00
---
### `🔭 Overview`

`DistilBERT` 는 허깅 페이스 연구진이 2019년 발표한 BERT의 변형으로서, On-Device Ai 개발을 목표로 경량화에 초점을 맞춘 모델이다. GPT, BERT의 등장 이후, NLP 분야에서 비약적인 성능 향상이 이뤄졌음에도 불구하고, 터무니 없는 모델 사이즈와 컴퓨팅 리소스 요구로 인해 실생활 적용 같은 활용성은 여전히 해결해야할 문제로 남아 있었다. Google에서 발표한 초기 `BERT-base-uncased` 만 해도 파라미터가 1억 1천만개 수준에 달한다. 

이를 다양한 비즈니스 요구 상황에 적용할 수 있으려면 최소한 8GB 이상의 가속기 전용 RAM 공간을 요구로 한다. 오늘날 개인용 PC 혹은 서버 컴퓨터의 경우, 8GB 이상의 VRAM이 달린 GPU가 일반적으로 탑재되기 때문에 크게 문제 될 것 없는 요구사항이지만, On-Device 환경에서는 이야기가 달라진다. 최신 하이엔드 스마트폰인 Galaxy S24 Ultra, iPhone 15 Pro의 경우 12GB, 8GB의 램 용량을 보유하고 있다. 그마저도 대부분의 온디바이스 환경은 SoC 구조를 채택하고 있기 때문에 전용 가속기가 온전히 저 모든 램 공간을 활용할 수 없다. 

따라서 온디바이스에 Ai를 적용하기 위해서는 획기적인 모델 경량화가 필요한 상황이고 그 출발점이 된 연구가 바로 `DistilBERT`다. 로컬 디바이스 환경에서도 언어 모델을 활용하기 위해 허깅 페이스 연구진은 지식 증류 기법을 활용해 인코더 기반 언어 모델의 파라미터를 획기적으로 줄이는데 성공한다.


정리하자면, `DistilBERT` 모델은 기존 BERT의 구조적 측면 개선이 아닌, 사전학습 방법 특히 경량화에 초점을 맞춘 시도라고 볼 수 있다. 따라서 어떤 모델이더라도, 인코더 언어 모델이라면 모두 `DistilBERT` 구조를 사용할 수 있으며, 기존 논문에서는 원본 BERT 구조를 사용했다. 이번 포스팅에서도 BERT 구조에 대한 설명 대신, `DistilBERT`의 사전 학습 방법론인 `Knowledge Distillation`에 대해서만 다루려고 한다.  

### `🌆 Knowledge Distillations`

$$
\min_{\theta}\sum_{x \in X} \alpha \mathcal{L}_{\text{KL}}(x, \theta) + \beta \mathcal{L}_{\text{MLM}}(x, \theta) + \gamma \mathcal{L}_{\text{Cos}}(x, \theta)
$$

`DistilBERT`는 Teacher-Student Architecture를 차용해 상대적으로 작은 파라미터 사이즈를 갖는 `Student` 모델에게 `Teacher`의 지식을 전수하는 것을 목표로 한다. 따라서 `Teacher` 모델은 이미 사전 학습을 마치고 수렴된 상태의 가중치를 갖고 있는 모델을 사용해야 한다. 더불어 Teacher 모델은 구조만 기존 BERT를 따르되, 사전 학습 방식은 RoBERTa의 방식과 동일(NSP 제거, Dynamic Masking 적용)하게 훈련되어야 한다.

한편, `Student` 모델은 `Teacher`의 60%정도 파라미터 사이즈를 갖도록 축소하여 사용한다. 이 때 축소는 모델의 `depth`(레이어 개수)에만 적용하는데, 연구진에 따르면 `width`(은닉층 크기)는 축소를 적용해도 연산 효율이 증가하지 않는다고 한다. 정리하면 `Teacher` 모델의 `레이어 개수*0.6`의 개수만큼 인코더를 쌓으면 된다는 것이다. 

그리고 최대한 `Teacher`의 지식을 전수해야 하기 때문에, 데이터는 `Teacher` 를 수렴시킨 것과 동일한 세트를 이용해야 한다. 이 때, Teacher 모델은 이미 MLE 방식으로 훈련이 된 상태라서 로짓이 단일 토큰 하나 쪽으로 쏠려 있을 가능성이 매우 높다. 이는 `Student` 모델의 일반화 능력에 악영향을 미칠 가능성이 높다. 따라서 Temperature 변수 $T$ 도입해 소프트 맥스(로짓)의 분포를 평탄화 한다. 이렇게 하면, `argmax()` 가 아닌 다른 토큰 표현에 대해서도 `Student` 모델이 지식을 습득할 수 있어서 풍부한 문맥을 학습하고 일반화 능력을 높이는데 도움이 된다. 이를 `암흑 지식(Dark Knowledge)` 을 활용한다고 표현한다. Temperature 변수 $T$ 도입한 소프트맥스 함수 수식은 아래와 같다.

$$
\text{softmax}(x_i) = \frac{e^{\frac{x_i}{\tau}}}{\sum_{j} e^{\frac{x_j}{\tau}}}
$$

수식상 변수 $T$의 값을 1이상으로 세팅해야 평탄화를 할 수 있다. 따라서 연구진은 $T =2$ 로 두고 사전 학습을 진행했다(논문에 공개안됨, GitHub에 있음). 이번 파트 맨 처음에 등장한 수식을 다시 보자. 결국 `DisilBERT`의 목적함수는 3가지 손실의 가중합으로 구성된다. 이제부터는 개별 손실에 대해서 자세히 살펴보자.

#### `🌆 Distillation Loss: KL-Divergence Loss`

$$
\text{KL-Divergence}(P || Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)}
$$

증류 손실로 사용되는 `KL-Divergence Loss`는 두 확률 분포 간의 차이를 측정하는 지표 중 하나다. 주로 확률 분포 P와 Q 사이의 차이를 나타내는데, 개별 요소의 확률값 차이가 클수록 합산값은 커져 손실이 커지게 된다. 반대로 두 분포의 개별 요소 확률값 차이가 작다면 당연히, 두 분포가 유사하다는 의미이므로 손실 역시 작아지게 된다. 일반적으로 `KL-Divergence Loss` 에서 확률분포 $P$ 가 이상적인 확률 분포를, $Q$ 가 모델이 예측한 확률분포를 의미한다. 따라서 `DistilBERT`의 경우 확률분포 $P$ 자리에는 `Teacher` 모델의 소프트맥스 분포가, $Q$ 에는 `Student` 모델의 소프트맥스 분포가 대입되면 된다. 이 때 두 확률분포 모두, 암흑 지식 획득을 위해 소프트맥스 평탄화를 적용한 결과를 사용한다. 논문에서, 선생 모델 예측에 평탄화를 적용한 것을 `소프트 라벨`, 학생 모델의 것에 적용한 결과는 `소프트 예측`이라고 부른다.

#### `🌆 Student Loss: MLM Loss`

$$
\mathcal{L}_{\text{MLM}} = - \sum_{i=1}^{N} \sum_{j=1}^{L} \mathbb{1}_{m_{ij}} \log \text{softmax}(x_{ij})
$$

학생 손실은 말그대로 기본적인 MLM 손실을 말한다. 정확한 손실값 계산을 위해서 학생의 소프트맥스 분포에 평탄화를 적용하지 않는다. 이를 논문에서는 `하드 예측`이라고 부른다. 라벨 역시 `Teacher`로부터 나온 것이 아닌 원래 MLM 수행에 사용되는 마스킹 라벨을 사용한다.

#### `🌆 Cosine Embedding Loss: Contrastive Loss by cosine similarity`

$$
\mathcal{L}_{\text{COS}}(x,y) = \begin{cases} 1 - \cos(x_1, x_2), & \text{if } y = 1 \\ \max(0, \cos(x_1, x_2) - \text{margin}), & \text{if } y = -1 \end{cases}

$$

`Teacher` 모델과 `Student` 모델의 마지막 인코더 모델이 출력하는 은닉값에 대한 `Contrastive Loss`를 의미한다. 이 때 `Distance Metric`은 코사인 유사도를 사용한다. 그래서 코사인 임베딩 손실이라고 논문에서 정의하는 것으로 추정된다. 위 수식을 최적화하는 것을 목적으로 한다. 이 때 라벨은 `[BS, Seq_len]`의 크기를 갖되, 모든 원소는 1이 되도록 만든다. 이유는 간단하다. `Student` 모델의 은닉값이 `Teacher` 모델의 것과 최대한 비슷해지도록 만드는게 우리 목적이기 때문이다.

### `👩‍💻 Implementation by Pytorch`
논문의 내용과 오피셜로 공개된 코드를 종합하여 파이토치로 `DistilBERT`를 구현해봤다. 논문에 포함된 아이디어를 이해하는데는 역시 어렵지 않았지만, 페이퍼에 hyper-param 테이블이 따로 제시되어 있지 않아 공개된 코드를 안 볼수가 없었다.

전체 모델 구조 대한 코드는 **[여기 링크](https://github.com/qcqced123/model_study)**를 통해 참고바란다.

#### `👩‍💻 Knowledge Distillation Pipeline`

```python
def train_val_fn(self, loader_train, model: nn.Module, criterion: Dict[str, nn.Module], optimizer,scheduler) -> Tuple[Any, Union[float, Any]]:
    """ Function for train loop with validation for each batch*N Steps
    DistillBERT has three loss:

        1) distillation loss, calculated by soft targets & soft predictions
            (nn.KLDIVLoss(reduction='batchmean'))

        2) student loss, calculated by hard targets & hard predictions
            (nn.CrossEntropyLoss(reduction='mean')), same as pure MLM Loss

        3) cosine similarity loss, calculated by student & teacher logit similarity
            (nn.CosineEmbeddingLoss(reduction='mean')), similar as contrastive loss

    Those 3 losses are summed jointly and then backward to student model
    """
    scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp_scaler)
    model.train()
    for step, batch in enumerate(tqdm(loader_train)):
        optimizer.zero_grad(set_to_none=True)
        inputs = batch['input_ids'].to(self.cfg.device, non_blocking=True)
        labels = batch['labels'].to(self.cfg.device, non_blocking=True)
        padding_mask = batch['padding_mask'].to(self.cfg.device, non_blocking=True)

        mask = padding_mask.unsqueeze(-1).expand(-1, -1, self.cfg.dim_model)  # for hidden states dim
        with torch.no_grad():
            t_hidden_state, soft_target = model.teacher_fw(
                inputs=inputs,
                padding_mask=padding_mask,
                mask=mask
            )  # teacher model's pred => hard logit

        with torch.cuda.amp.autocast(enabled=self.cfg.amp_scaler):
            s_hidden_state, s_logit, soft_pred, c_labels = model.student_fw(
                inputs=inputs,
                padding_mask=padding_mask,
                mask=mask
            )
            d_loss = criterion["KLDivLoss"](soft_pred.log(), soft_target)  # nn.KLDIVLoss
            s_loss = criterion["CrossEntropyLoss"](s_logit.view(-1, self.cfg.vocab_size), labels.view(-1))  # nn.CrossEntropyLoss
            c_loss = criterion["CosineEmbeddingLoss"](s_hidden_state, t_hidden_state, c_labels)  # nn.CosineEmbeddingLoss
            loss = d_loss*self.cfg.alpha_distillation + s_loss*self.cfg.alpha_student + c_loss*self.cfg.alpha_cosine  # linear combination loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
```

#### `👩‍💻 Knowledge Distillation Model`

```python
class DistillationKnowledge(nn.Module, AbstractTask):
    """ Custom Task Module for Knowledge Distillation by DistilBERT Style Architecture
    DistilBERT Style Architecture is Teacher-Student Framework for Knowledge Distillation,

    And then they have 3 objective functions:
        1) distillation loss, calculated by soft targets & soft predictions
            (nn.KLDIVLoss(reduction='batchmean'))
        2) student loss, calculated by hard targets & hard predictions
            (nn.CrossEntropyLoss(reduction='mean')), same as pure MLM Loss
        3) cosine similarity loss, calculated by student & teacher logit similarity
            (nn.CosineEmbeddingLoss(reduction='mean')), similar as contrastive loss

    References:
        https://arxiv.org/pdf/1910.01108.pdf
        https://github.com/huggingface/transformers/blob/main/examples/research_projects/distillation/distiller.py
    """
    def __init__(self, cfg: CFG) -> None:
        super(DistillationKnowledge, self).__init__()
        self.cfg = CFG
        self.model = DistilBERT(
            self.cfg,
            self.select_model
        )
        self._init_weights(self.model)
        if self.cfg.teacher_load_pretrained:  # for teacher model
            self.model.teacher.load_state_dict(
                torch.load(cfg.checkpoint_dir + cfg.teacher_state_dict),
                strict=False
            )
        if self.cfg.student_load_pretrained:  # for student model
            self.model.student.load_state_dict(
                torch.load(cfg.checkpoint_dir + cfg.student_state_dict),
                strict=True
            )
        if self.cfg.freeze:
            freeze(self.model.teacher)
            freeze(self.model.mlm_head)

        if self.cfg.gradient_checkpoint:
            self.model.gradient_checkpointing_enable()

    def teacher_fw(
        self,
        inputs: Tensor,
        padding_mask: Tensor,
        mask: Tensor,
        attention_mask: Tensor = None,
        is_valid: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """ teacher forward pass to make soft target, last_hidden_state for distillation loss """
        # 1) make soft target
        temperature = 1.0 if is_valid else self.cfg.temperature
        last_hidden_state, t_logit = self.model.teacher_fw(
            inputs,
            padding_mask,
            attention_mask
        )
        last_hidden_state = torch.masked_select(last_hidden_state, ~mask)  # for inverse select
        last_hidden_state = last_hidden_state.view(-1, self.cfg.dim_model)  # flatten last_hidden_state
        soft_target = F.softmax(
            t_logit.view(-1, self.cfg.vocab_size) / temperature**2,  # flatten softmax distribution
            dim=-1
        )  # [bs* seq, vocab_size]
        return last_hidden_state, soft_target

    def student_fw(
        self,
        inputs: Tensor,
        padding_mask: Tensor,
        mask: Tensor,
        attention_mask: Tensor = None,
        is_valid: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """ student forward pass to make soft prediction, hard prediction for student loss """
        temperature = 1.0 if is_valid else self.cfg.temperature
        last_hidden_state, s_logit = self.model.teacher_fw(
            inputs,
            padding_mask,
            attention_mask
        )
        last_hidden_state = torch.masked_select(last_hidden_state, ~mask)  # for inverse select
        last_hidden_state = last_hidden_state.view(-1, self.cfg.dim_model)  # flatten last_hidden_state
        c_labels = last_hidden_state.new(last_hidden_state.size(0)).fill_(1)
        soft_pred = F.softmax(
            s_logit.view(-1, self.cfg.vocab_size) / temperature**2,  # flatten softmax distribution
            dim=-1
        )
        return last_hidden_state, s_logit, soft_pred, c_labels
```

#### `👩‍💻 DistilBERT Model`

```python
class DistilBERT(nn.Module, AbstractModel):
    """ Main class for DistilBERT Style Model, Teacher-Student Framework
    for Knowledge Distillation aim to lighter Large Scale LLM model. This model have 3 objective functions:

        1) distillation loss, calculated by soft targets & soft predictions
            (nn.KLDIVLoss(reduction='batchmean'))

        2) student loss, calculated by hard targets & hard predictions
            (nn.CrossEntropyLoss(reduction='mean')), same as pure MLM Loss

        3) cosine similarity loss, calculated by student & teacher logit similarity
            (nn.CosineEmbeddingLoss(reduction='mean')), similar as contrastive loss

    soft targets & soft predictions are meaning that logit are passed through softmax function applied with temperature T
    temperature T aim to flatten softmax layer distribution for making "Dark Knowledge" from teacher model

    hard targets & hard predictions are meaning that logit are passed through softmax function without temperature T
    hard targets are same as just simple labels from MLM Collator returns for calculating cross entropy loss

    cosine similarity loss is calculated by cosine similarity between student & teacher
    in official repo, they mask padding tokens for calculating cosine similarity, target for this task is 1
    cosine similarity is calculated by nn.CosineSimilarity() function, values are range to [-1, 1]

    you can select any other backbone model architecture for Teacher & Student Model for knowledge distillation
    but, in original paper, BERT is used for Teacher Model & Student
    and you must select pretrained model for Teacher Model, because Teacher Model is used for knowledge distillation,
    which is containing pretrained mlm head

    Do not pass gradient backward to teacher model!!
    (teacher model must be frozen or register_buffer to model or use no_grad() context manager)

    Args:
        cfg: configuration.CFG
        model_func: make model instance in runtime from config.json

    References:
        https://arxiv.org/pdf/1910.01108.pdf
        https://github.com/huggingface/transformers/blob/main/examples/research_projects/distillation/distiller.py
    """
    def __init__(self, cfg: CFG, model_func: Callable) -> None:
        super(DistilBERT, self).__init__()
        self.cfg = cfg
        self.teacher = model_func(self.cfg.teacher_num_layers)  # must be loading pretrained model containing mlm head
        self.mlm_head = MLMHead(self.cfg)  # must be loading pretrained model's mlm head

        self.student = model_func(self.cfg.student_num_layers)
        self.s_mlm_head = MLMHead(self.cfg)

    def teacher_fw(
        self,
        inputs: Tensor,
        padding_mask: Tensor,
        attention_mask: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        """ forward pass for teacher model
        """
        last_hidden_state, _ = self.teacher(
            inputs,
            padding_mask,
            attention_mask
        )
        t_logit = self.mlm_head(last_hidden_state)  # hard logit => to make soft logit
        return last_hidden_state, t_logit

    def student_fw(
        self,
        inputs: Tensor,
        padding_mask: Tensor,
        attention_mask: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        """ forward pass for student model
        """
        last_hidden_state, _ = self.student(
            inputs,
            padding_mask,
            attention_mask
        )
        s_logit = self.s_mlm_head(last_hidden_state)  # hard logit => to make soft logit
        return last_hidden_state, s_logit
```
