# Local-Jev

在本機用小型模型（≤ 3B）提供 Jev-compatible 的決策服務：說 TypeSafe 的 `/v1/systemone` wire format，讓 jevbench 內建的 `typesafe` adapter 不需修改就能評測它。每個 server process 只跑一個 Decider，啟動時選定。

目前可用的 Decider：

- `lfm3`：LFM3 Option-ID Decider。`LiquidAI/LFM3-1.7B`（Chat model）透過 Option-ID scoring 回答：關掉 thinking，一次 forward pass 讀出每個 Option ID 的 next-token 機率，再在 Label set 上正規化。固定在 revision `0dc4268b561499c648c7ba7c9466bf3d8d01517a`。
- `d1`：`LiquidAI/d1-2B`（Decision model），透過它自己的 `system_one` head 一次 forward pass 直接回答，固定在 revision `652f920e544ea37f81e4b0ce89b81d4ed0da90b1`。

## 在 Mac 上原生執行（`uv run`）

需求：Apple Silicon Mac、[uv](https://docs.astral.sh/uv/)。Python 3.12 與所有相依套件都由 uv 依 `uv.lock` 安裝，不需另外準備。

```sh
uv sync                                  # 第一次：建立 .venv 並安裝鎖定的版本
uv run local-jev serve --decider lfm3    # 或 --decider d1；預設聽 127.0.0.1:8000
```

> **第一次啟動會下載模型。** `LiquidAI/LFM3-1.7B` 約 3.2 GB、`LiquidAI/d1-2B` 約 3.6 GB，會下載到 Hugging Face 快取（預設 `~/.cache/huggingface/hub`，可用 `HF_HOME` 改位置），之後的啟動直接讀快取。下載期間 server 已經在聽，但 health 會回報尚未就緒。下載完成後可以設 `HF_HUB_OFFLINE=1`，省掉啟動時對 Hub 的查詢。

選項：

| 選項 | 預設 | 說明 |
| --- | --- | --- |
| `--decider` | （必填） | `lfm3` 或 `d1`，見下方「選擇 Decider」 |
| `--host` | `127.0.0.1` | 要讓其他機器連線時改成 `0.0.0.0` |
| `--port` | `8000` | |
| `--device` | `auto` | 自動依序選 CUDA → MPS → CPU；可指定 `cuda`、`mps`、`cpu` |
| `--dtype` | `auto` | CUDA/MPS 用 `bfloat16`、CPU 用 `float32`；可指定 `bfloat16`、`float16`、`float32` |
| `--model` | Decider 自己的 model | Hugging Face model id；也可用環境變數 `LOCAL_JEV_MODEL` |
| `--revision` | Decider 自己固定的 revision | 給了 `--model` 就必須一起給；也可用 `LOCAL_JEV_REVISION` |
| `--chat-template-options` | `{"enable_thinking": false}` | 只適用 `lfm3`：傳給 chat template 的選項（JSON 物件），會合併在預設值之上；也可用 `LOCAL_JEV_CHAT_TEMPLATE_OPTIONS` |

在 Mac 上 `auto` 會選 MPS（Metal）。容器裡沒有 MPS，只能用 CPU。

### 選擇 Decider

每個 server process 只載入一個 Decider，啟動時用 `--decider` 選定；要比較兩者就各開一個 server（不同 port）分別跑 benchmark。兩個 Decider 對外的 wire format 完全一樣。

| | `lfm3` | `d1` |
| --- | --- | --- |
| Model | `LiquidAI/LFM3-1.7B`，Chat model | `LiquidAI/d1-2B`，Decision model |
| 怎麼得到 Distribution | Option-ID scoring：`choice` 的選項依 Criteria 送來的順序標成 A–Z，`noul` 讀 yes/no（`Yes`/`yes`、`No`/`no` 合併計算），`score` 讀等級數字 0–9；在 Label set 上正規化 | d1 自己的 decision head |
| 每個 Decision 的成本 | 一次 forward pass，不生成任何 token | 一次 forward pass |
| Reason（`explain`） | 支援，見下方「要求 Reason」 | 不支援：照常回答，不帶 `reason`，也不報錯 |
| MPS 上的延遲（M3 Pro） | 每個 Decision 約 115–240 ms（prompt 80–530 tokens） | 約 65–120 ms（prompt 約 65–75 tokens） |
| 記憶體（MPS bf16，driver） | 約 4.1 GB | 約 5.4 GB |

`lfm3` 在啟動時會先確認每個 Option ID（A–Z、yes/Yes/no/No、0–9）在被讀取的位置剛好是一個 token，不是的話 health 會回報 `failed`，不會等到第一個請求才出錯。

**換成另一個 ≤ 3B 的 Chat model 不需要改程式**，例如：

```sh
uv run local-jev serve --decider lfm3 \
  --model Qwen/Qwen3-1.7B --revision <commit sha> \
  --chat-template-options '{"enable_thinking": false}'
```

- 一定要給 `--revision`：model 用 `trust_remote_code` 載入，固定 revision 才能固定實際執行的程式碼。
- chat template 選項依 model 而定；Option ID 直接接在 chat template 的 generation prompt 之後讀取，所以 template 必須在那裡結束（例如 LFM3 的 `<think></think>`）。
- `--model`/`--revision` 也適用 `d1`（換另一個 d1 checkpoint）。

**已知風險：`lfm3` 有很強的 recency bias。** 研究時用同一個三選一的問題試過全部 6 種選項順序，LFM3-1.7B 每次都選兩個合理選項中「排在後面」的那個，機率 0.85–1.00。這會拉高 Brier 分數，temperature calibration 也修不了 argmax。v1 不處理它：把多種選項順序平均起來可以抵銷順序的影響，但每個 Decision 就不再只要一次 forward pass。

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

一個請求可以帶好幾個具名的 Question（都針對同一個 State），回應的 `answers` 底下用同樣的名稱各回一個答案。每個 Question 各自呼叫一次 Decider，所以答案不會因為同一個請求裡還有哪些其他 Question 而改變（d1 若把多個 question 塞進同一次呼叫，答案會偏移約 0.03）。

回應裡還有：

```json
{"usage":   {"input_tokens": 73, "output_tokens": 0},
 "timing":  {"total_ms": 98.512, "decider_ms": 97.904},
 "runtime": {"decider": "d1", "model": "LiquidAI/d1-2B",
             "revision": "652f920e544ea37f81e4b0ce89b81d4ed0da90b1", "device": "mps",
             "total_ms": 98.512, "decider_ms": 97.904}}
```

- `usage`：所有 Question 加總的 token 數。沒有要求 Reason 時，兩個 Decider 都只讀 prompt 不生成，所以 `output_tokens` 是 0；`lfm3` 寫 Reason 時，`output_tokens` 是 Reason 的 token 數，`input_tokens` 也加上寫 Reason 時另外讀的 prompt。
- `timing`：server 自己量的時間（毫秒）。`total_ms` 從解析完請求到產生回應為止（包含等前一個請求的時間），`decider_ms` 是其中花在 Decider 上的部分。拿來跟 client 端的 wall-clock 對照，就能分出網路與 client 的額外開銷。
- `runtime`：同樣的時間，再加上 Decider 名稱、model id、revision 與 device（`cuda`、`mps` 或 `cpu`）。jevbench 會把回應頂層的 `runtime` 原樣抄進 `results.jsonl` 的每一筆紀錄，所以每筆結果都帶著它是在什麼硬體上跑的。

請求一次處理一個：同時送來的請求會排隊，不會交錯使用模型。

### 三種 Question type

| type | `criteria` | 答案 |
| --- | --- | --- |
| `choice` | `{label: 說明}`，1 到 26 個選項（兩個 Decider 都一樣） | `{"type": "choice", "choice": <最可能的 Label>, "probabilities": {label: …}}` |
| `noul` | 選用，`{"true": "yes 代表什麼", "false": "no 代表什麼"}` | `{"type": "noul", "noul": <P(yes)>}` |
| `score` | 2 到 10 個等級說明的 list，由低到高 | `{"type": "score", "score": <最可能的等級>, "probabilities": {"0": …, "1": …}}` |

`score` 的 `probabilities` 以等級 index 字串（`"0"` 到 `"N-1"`）為 key。d1 自己的 `score` 欄位是中位數等級，不一定是最可能的等級，所以不採用。

### 要求 Reason（`explain`）

想知道某個 Decision 為什麼這樣判斷時（例如在看自己的資料），在請求裡加上 `"explain": true`，`lfm3` 會在每個答案裡多一個 `reason` 欄位：

```json
{"answers": {"decision": {"type": "choice", "choice": "track_order",
                          "probabilities": {"track_order": 0.99999, "…": 0.0},
                          "reason": "這位使用者想知道包裹的當前位置與預計送達時間，因此屬於 **track_order** 意圖。"}}}
```

- `explain` 是我們對 wire format 的擴充，預設 `false`，只接受 JSON 的 `true`/`false`（`"yes"`、`1` 之類會回 422）。jevbench 的 `typesafe` adapter 不會送它，benchmark 也一律不開，所以 benchmark 的延遲不受影響。
- Reason 在 Distribution 算完之後才生成：把同一個 State 與 Question、加上已經選定的最可能 Label 交給 model，請它用一句話說明理由。所以有沒有要求 Reason，`probabilities`、`choice`、`noul`、`score` 都完全一樣。
- Reason 用 State 的語言寫：State 以中文字為主時用繁體中文（台灣用語），否則用英文。Question 的 instructions 是英文也一樣。
- 長度上限 64 個 token（中文約一字一個 token），寫到 model 的回合結束或上限就停；碰到上限時句子可能被截斷。實測一句話約 27–35 個 token，在 M3 Pro 的 MPS 上每個 Reason 約 1.5 秒，容器裡的 CPU 會慢很多。
- Reason 是 model 生成的純文字，server 建回應時做 JSON escape，引號、換行之類的字元不會弄壞回應。
- **`d1` 不支援 Reason**：d1 是 Decision model，不會生成文字（見 ADR-0003）。對 `d1` 送 `"explain": true` 會照常回答，只是沒有 `reason` 欄位，也不會報錯。

### 錯誤的請求回 422

不支援的 Question type（`text`、`number`、`date`、`box`、`boxes`、多選 `choice`，也就是 `"multiple": true`）、超過 26 個選項的 `choice`（`lfm3` 用 A–Z 標選項；為了兩個 Decider 行為一致，`d1` 也一樣拒絕）與格式錯誤的請求，一律回 HTTP 422，`detail` 是一句話，指出哪裡錯、該怎麼改，例如：

```json
{"detail": "questions.decision: a score's \"criteria\" lists 2 to 10 levels, lowest first, as [\"Can wait\", \"Today\", \"Now\"]; this one lists 11"}
```

整個請求只要有一個 Question 有問題就整個拒絕，不會先回答其他 Question。用 422 而不是 400，是因為 jevbench 把 422 算成答錯，但不計入「連續三次失敗就停止整個 run」的規則。

### 用 bearer token 保護（選用）

```sh
LOCAL_JEV_TOKEN=換成你的-token uv run local-jev serve --decider d1 --host 0.0.0.0
```

- 有設 `LOCAL_JEV_TOKEN`（非空字串）時，`/v1/systemone` 必須帶 `Authorization: Bearer <token>`，沒帶或錯了都回 401。`/health` 不需要 token，方便 compose 做 health check。
- 沒設時，帶任何 token 或不帶都接受。jevbench 的 `typesafe` adapter 預設一定會送一個 token（環境變數空著它就不肯跑），所以本機跑 benchmark 時隨便設一個假值即可，例如 `TYPESAFE_API_KEY=local`。
- Server 有設 token 時，把同一個值放進 jevbench `--key-env` 指定的環境變數。

## 測試

```sh
uv run pytest                 # 快速測試：in-process 打 HTTP contract，用假的 d1 model 與假的 LM backend，不載入權重
uv run pytest -m real_model   # 真模型測試：載入真正的 LFM3-1.7B 與 d1-2B（第一次會下載），按需執行
uv run pytest -m real_model tests/test_real_lfm3.py   # 只跑其中一個 model
uv run ruff check
```
