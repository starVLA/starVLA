# Qwen Special Token Addition Script

Quickly add new special tokens to Qwen/Qwen2.5-VL-3B-Instruct (or compatible models) and save them to a locally loadable directory.

## 运行

```bash


source_model_id=playground/Pretrained_models/Qwen3-VL-4B-Instruct-Fang
target_model_id=playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action
fast_token_list=starVLA/model/modules/vlm/tools/add_qwen_special_tokens/fast_tokens.txt

python starVLA/model/modules/vlm/tools/add_qwen_special_tokens/add_special_tokens_to_qwen.py \
  --model-id ${source_model_id} \
  --tokens-file ${fast_token_list} \
  --save-dir ${target_model_id} \
  --init-strategy normal
  
```

`tokens.txt` example:
```
<loc_x>
<loc_y>
<bbox_start>
<bbox_end>
```

 
## Arguments
 
- --model-id: HF Hub model ID or an existing local model directory
- --save-dir: Output directory
- --tokens-file
- --init-strategy: avg / normal / zero
- --as-special / --no-as-special: Add as special tokens or regular tokens
- --padding-side: left / right
- --device: cpu / cuda / mps / auto

 
## Results
 
The saved directory contains:
 
- config.json / model.safetensors / tokenizer.*
- added_token_id_map.json (records the mapping from newly added tokens to IDs)

 
 
## Load
 
```python
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
tok = AutoTokenizer.from_pretrained("./qwen_vl_with_spatial", trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("./qwen_vl_with_spatial", torch_dtype="auto", trust_remote_code=True)
print(tok.convert_tokens_to_ids("<loc_x>"))
```
