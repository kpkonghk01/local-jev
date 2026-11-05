# Local-Jev

在本機用小型模型（≤ 3B）提供 Jev-compatible 的決策服務：說 TypeSafe 的 `/v1/systemone` wire format，讓 jevbench 內建的 `typesafe` adapter 不需修改就能評測它。每個 server process 只跑一個 Decider，啟動時選定。

目前可用的 Decider：

- `d1`：`LiquidAI/d1-2B`（Decision model），透過它自己的 `system_one` head 一次 forward pass 直接回答，固定在 revision `652f920e544ea37f81e4b0ce89b81d4ed0da90b1`。

## 在 Mac 上原生執行（`uv run`）

需求：Apple Silicon Mac、[uv](https://docs.astral.sh/uv/)。Python 3.12 與所有相依套件都由 uv 依 `uv.lock` 安裝，不需另外準備。

```sh
uv sync                                  # 第一次：建立 .venv 並安裝鎖定的版本
uv run local-jev serve --decider d1      # 預設聽 127.0.0.1:8000
```

> **第一次啟動會下載模型。** `LiquidAI/d1-2B` 約 3.6 GB，會下載到 Hugging Face 快取（預設 `~/.cache/huggingface/hub`，可用 `HF_HOME` 改位置），之後的啟動直接讀快取。下載期間 server 已經在聽，但 health 會回報尚未就緒。下載完成後可以設 `HF_HUB_OFFLINE=1`，省掉啟動時對 Hub 的查詢。

選項：

| 選項 | 預設 | 說明 |
| --- | --- | --- |
| `--decider` | （必填） | 目前只有 `d1` |
| `--host` | `127.0.0.1` | 要讓其他機器連線時改成 `0.0.0.0` |
| `--port` | `8000` | |
| `--device` | `auto` | 自動依序選 CUDA → MPS → CPU；可指定 `cuda`、`mps`、`cpu` |
| `--dtype` | `auto` | CUDA/MPS 用 `bfloat16`、CPU 用 `float32`；可指定 `bfloat16`、`float16`、`float32` |

在 Mac 上 `auto` 會選 MPS（Metal）。容器裡沒有 MPS，只能用 CPU。

### 確認服務就緒

```sh
curl -s http://127.0.0.1:8000/health
# 載入中：HTTP 503 {"status":"loading"}
# 就緒：  HTTP 200 {"status":"ready"}
# 載入失敗：HTTP 503 {"status":"failed","error":"…"}
```

模型載入並暖機完成之前，`/v1/systemone` 一律回 503。請等 health 回報 ready 再送流量（jevbench 連續三次失敗就會停止整個 run）。

### 送一個 `choice` Question

```sh
curl -s http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state": "Where is my package? I ordered it last week and it still hasn'"'"'t arrived.",
       "model": "d1-2B",
       "questions": {"decision": {"type": "choice",
         "instructions": "Which intent does the user'"'"'s message express?",
         "criteria": {"track_order": "Wants to know where an order is or when it arrives",
                      "cancel_order": "Wants to cancel an order",
                      "billing_question": "Asks about a charge, invoice or payment"}}}}'
```

回應裡 `answers` 底下用同樣的名稱（`decision`）回答：`choice` 是最可能的 Label，`probabilities` 是這個 Question 的 Label set 上加總為 1 的 Distribution；`model` 原樣回傳請求裡的值。State 可以是純文字，也可以是 JSON 物件。

## 測試

```sh
uv run pytest                 # 快速測試：in-process 打 HTTP contract，用假的 d1 model，不載入權重
uv run pytest -m real_model   # 真模型測試：載入真正的 d1-2B（第一次會下載），按需執行
uv run ruff check
```
