---
title: '🎲 RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(hand≤)'
excerpt: "Pytorch Error: Mis-match between pre-defined dimension and input dimension"
permalink: "/cs-ai/framework-library/mismatch-embedding"
toc: true  # option for table of content
toc_sticky: true  # option for table of content
categories:
  - Pytorch Error Handling
tags:
  - Pytorch
  - Dimension Mismatch
  - nn.Embedding
  - CUDA
  - Error Handling
last_modified_at: 2023-07-17T12:00:00-05:00
---

### `😵 nn.Embedding 차원 ≠ 실제 데이터 입력 차원`
`torch.nn.Embedding`에서 정의한 입출력 차원과 실제 데이터의 차원이 다른 경우에 발생하는 에러다. 다양한 상황에서 마주할 수 있는 에러지만, 필자의 경우 `Huggingface`에서 불러온`pretrained tokenizer`에 `special token` 을 추가해 사용할 때, 토큰을 추가했다는 사실을 잊고 `nn.Embedding` 에 정의한 입출력 차원을 변경하지 않아서 발생하는 경우가 많았다. 

```python
from transformers import AutoTokenizer, AutoConfig, AutoModel

class CFG:
    model_name = 'microsoft/deberta-v3-large'
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, config=config)


def add_markdown_token(cfg: sCFG) -> None:
    """
    Add MarkDown token to pretrained tokenizer ('[MD]')
    Args:
        cfg: CFG, needed to load tokenizer from Huggingface AutoTokenizer
    """
    markdown_token = '[MD]'
    special_tokens_dict = {'additional_special_tokens': [f'{markdown_token}']}
    cfg.tokenizer.add_special_tokens(special_tokens_dict)
    markdown_token_id = cfg.tokenizer(f'{markdown_token}', add_special_tokens=False)['input_ids'][0]

    setattr(cfg.tokenizer, 'markdown_token', f'{markdown_token}')
    setattr(cfg.tokenizer, 'markdown_token_id', markdown_token_id)
    cfg.tokenizer.save_pretrained(f'{cfg.checkpoint_dir}/tokenizer/')


add_markdown_token(CFG)
CFG.model.resize_token_embeddings(len(tokenizer))
```
구글링해보니 해결하는 방법은 다양한 것 같은데, `torch.nn.Embedding`에 정의된 입출력 차원을 실제 데이터 차원과 맞춰주면 간단하게 해결된다. 필자처럼 `special token` 을 추가해 사용하다 해당 에러가 발생하는 상황이라면 새로운 토큰이 추가된 토크나이저의 길이를 다시 측정한 뒤 값을 `resize_token_embeddings` 메서드에 전달해 `nn.Embedding`을 업데이트 해주면 된다. 

