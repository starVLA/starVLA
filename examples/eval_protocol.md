# StarVLA 测试评测框架使用指南

## 1. 总览
StarVLA 在真机 / 仿真评测中的标准化推理接入流程（Pipeline）使用 WebSocket 做数据穿透，以最小改动将新模型集成到现有评测环境。


## 2. 架构示意

```
+--------------------+                   +---------------------------+
|  Eval Env (Sim/Real)|  ---->  request(example) -> |   Model Server (StarVLA)  |
|  - 采集多视角图像      |          WebSocket         |  - framework.predict_action |
|  - 语言指令           | <---- response(actions)   -|

```

## 3. 数据协议（Example 字典约定）


最小伪代码示例（评测侧 Client）：

```python

import WebsocketClientPolicy

client = WebsocketClientPolicy(
    host="127.0.0.0",
    port=10092
)

while True:
    images = capture_multiview()          # 返回 List[np.ndarray]
    lang = get_instruction()              # 可能来自任务脚本
    example = {
        "image": images,
        "lang": lang,
    }

    result = client.predict_action(example) #--> 直通到 framework.predict_action
    action = result["normalized_actions"][0]   # 取 batch 第一条
    apply_action(action)
```

```

注意：




## 7. 常见问题 FAQ

Q: 为什么examples 文件夹下面会事例存在 model2{bench}_client.py？
A: 因为 policy 到 sim apply action 中间还存在很多 bench / 用户特有的操作， 然后 delta action --> abs action。


Q: 为什么模型需要的是 PIL，但是传输的时候是 ndarray？

Q: 因为 WebSocket 图像不直接传 PIL，需要在 Client 侧转换（`PIL.Image -> np.ndarray`），Server 再还原（内部使用 `to_pil_preserve`）

