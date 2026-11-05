# Local-Jev 可行性報告：≤ 3B 的本機模型能不能當 Jev-compatible system

- 日期：2026-11-05（量測在 10:04–10:14 UTC）
- 量測對象：local-jev v1（branch `ticket-12-benchmark-feasibility`），jevbench commit `609a664`，`LiquidAI/d1-2B@652f920`、`LiquidAI/LFM3-1.7B@0dc4268`
- 硬體：Apple M3 Pro（36 GB），macOS，原生 `uv run`，MPS、bfloat16；torch 2.11.0、transformers 5.22.1
- 比較報告原文（`local-jev report` 的輸出）：[`docs/comparison-report.md`](comparison-report.md)

## 1. 結論摘要

**Verdict：fail。** 照議定的標準（231 個英文 public Tasks、zero-shot 加 calibration、cross-validated 數字），兩個 Decider 都沒有達到「Accuracy ≥ 80% 且 Brier ≤ 0.20」（標準 2）。Valid answers 100%（標準 1）與 Mac 上的延遲記錄（標準 3a）都 pass，GPU 上的延遲（標準 3b）因為還沒有 GPU host 而 pending。`local-jev report` 的判定是 **Overall: fail (fail: 2; pending: 3b)**。所以目前的結論不是「feasible, worth further investment」。

| Decider | Accuracy | Brier（raw → CV） | ECE（raw → CV） | Valid answers | Server p50（MPS） |
| --- | --- | --- | --- | --- | --- |
| `d1`（d1-2B，Decision model） | **76.6%** | 0.285 → **0.275** | 0.072 → 0.060 | 100% | **128 ms** |
| `lfm3`（LFM3-1.7B，Option-ID） | 62.3% | 0.626 → 0.495 | 0.272 → 0.038 | 100% | 162 ms |

（231 個英文 public Tasks；CV 是 5-fold cross-validated calibration。繁體中文另見 7.5 節：d1 78.8%、LFM3 58.0%。）

重點：

- **d1-2B 是可用的起點，但還不夠好。** 76.6% 高於 jevbench 公布的 system-one-open（73.2%）與 decider-2b（71.0%），低於 Jev 1.13.0（86.6%），差距幾乎全在 hard（d1 61/111，Jev 81/111）。95% 信賴區間是 70.8–81.6%，80% 在區間內，但 Brier 0.275 離 0.20 很遠：86% 的 Brier 來自答錯的 54 題，per-type temperature calibration 只降到 0.275，要過關得靠更高的 Accuracy。
- **LFM3-1.7B 加 Option-ID scoring 不可行。** 62.3%，明顯偏向第一個選項、偏向答 yes，答錯時平均 confidence 0.83；calibration 把 ECE 壓到 0.038，但 Brier 仍有 0.495。
- **延遲沒有問題。** d1 在 M3 Pro 的 MPS 上 server 端 p50 128 ms（p95 2.2 s，來自上千 tokens 的長 State）；L4 等級 GPU 上 ≤ 150 ms 的標準 3b 仍待驗證。
- **繁體中文沒有變差。** d1 在中文 State 上 78.8%（英文 76.6%，差異不顯著），LFM3 58.0%。
- **預設 Decider 改成 `d1`**（`local-jev serve` 不給 `--decider` 就是 d1），README 已註明。
- **授權：兩個 model 都是 LFM Open License v1.0，年營收達 1,000 萬美元以上的法人不能商業使用**（6.6 節）。
- 原始提案的前提大多不成立（第 5 節）；它說的「領先指標」也沒有出現：我們量到最好的 Decider（d1）在 public Tasks 上比 Jev 低 10 個百分點，提案主張的 1.7B 模型低 24 個百分點。

## 2. 背景：要回答的問題

開發者手上的原始技術提案是搜尋引擎 AI Overview 生成的初稿，主張「1.7B 的模型加上 xgrammar 限制的 JSON」就能在 jevbench 拿到「領先的指標」。查證後發現它大部分的前提都不成立（逐條見第 5 節）：「JEV」不是「JSON Evaluation」，jevbench 不用提案裡那個四個欄位的 schema、也不量 TTFT，vLLM 目前載入不了 LFM3，而且強的參賽系統也不是靠 grammar-constrained JSON 產生機率。

但問題本身還是值得用數字回答：**一個本機執行、≤ 3B 參數的小模型，能不能成為可信的 Jev-compatible system**，也就是答得夠準、機率校準得夠好、夠快，而且英文和繁體中文的 State 都能用。local-jev v1 就是為了回答這個問題而做的：一個說 TypeSafe `/v1/systemone` wire format 的小型 HTTP server，讓 jevbench 內建的 `typesafe` adapter 不改一行就能評測它，並且用同一套程式在 Mac（MPS）、容器（CPU）與之後的 Linux NVIDIA GPU host 上執行。

這份報告記錄 jevbench 是什麼、wire format 長什麼樣、原始提案逐條的查證結果、模型與 backend 的發現、實際量到的數字，以及照事先議定的標準得出的 verdict。

## 3. jevbench 是什麼、怎麼跑

### 3.1 它是誰的、是哪個版本

- jevbench（JevBench）是**評測 decision model 的公開 benchmark**，量的是有型別的 Decision 的 accuracy、機率的 calibration、latency 與 cost。Jev 本身是 TypeSafe AI 的商用 decision model；jevbench 除了 Jev，也評測任何 Jev-compatible system。
- **維護者是 Benchmark Heaven（Florian Standhartinger），不是 TypeSafe AI。** README 寫明 "Maintained by Benchmark Heaven, independently of TypeSafe AI"，`THIRD-PARTY.md` 也寫 "JevBench is not affiliated with or endorsed by TypeSafe AI"。wire format 是 TypeSafe 公開的 API。
- 程式與資料採 MIT License（Copyright (c) 2026 Florian Standhartinger and contributors），repo 是 <https://github.com/fstandhartinger/jevbench>。另有一個同名、做 metamorphic coherence 的 jevbench.github.io，是不相干的專案。
- local-jev 固定在 commit `609a664e81f46e03731600bd7bd9f1b7a61257c5`（2026-10-15，2026-11-05 時仍是 `main` 的 HEAD）。這個 commit 的 README 寫 release v1.6.1，live 網站列的是 v1.8.7；我們跑的是 repo 裡的程式與資料。
- jevbench 只用 Python 標準函式庫，local-jev 是把它 clone 到 git-ignored 的快取裡執行，不把它當成相依套件安裝。

### 3.2 Tasks

一個 Task 是一筆 jevbench 紀錄：一個 State、一個 Question、它的 Label set 和預期的 Label。公開的 Tasks 共 231 個，分三個部分：

| 部分 | Tasks | `choice` | `noul` | `score` | 說明 |
| --- | --- | --- | --- | --- | --- |
| easy | 48 | 36 | 12 | 0 | 短的客服訊息、事實、抽取、工具選擇 |
| original | 72 | 36 | 24 | 12 | 原始的 72 題（36 組改寫配對），政策、意圖、等級、路由等 |
| hard | 111 | 67 | 38 | 6 | 長政策、multi-hop、judge、時間與數字推理、陷阱、注入攻擊等；35 個 State 是 JSON 物件，最長約 15K 字元 |

- `choice` 的 Label set 有 3 到 6 個選項；`score` 是 4 個等級（17 題）或 5 個等級（1 題）；`noul` 的 Labels 是 `yes`/`no`。
- 線上 board 用的是另外的 sealed（不公開）Tasks，只能請維護者對參賽的 endpoint 跑；local-jev 不送 board（spec 的 Out of Scope），所以這份報告的數字都只在 231 個 public Tasks 上。

### 3.3 怎麼計分

`summarize` 對一個 run 算出的（也就是 local-jev 報告用的）：

- **Accuracy**：取 Distribution 在 Label set 上的 argmax（同分時取字典序最小的 Label）與預期 Label 比對。**無效的 Distribution 一律算錯。**
- **Valid answers**（`schema_validity`）：Distribution 的 key 必須剛好等於 Label set，值是 [0, 1] 的有限數字，加總與 1 的差在 0.02 以內（0.001 以內是 strict）。
- **Brier**：Label set 上的 multi-class sum Σ_k (p_k − y_k)²，範圍 0 到 2（`noul` 等於 2·(p_yes − y)²），所有 Question type 合在一起平均。
- **ECE**：top-label confidence、10 個等寬的 bin。`noul` 的 confidence 是 max(p_yes, 1 − p_yes)。
- **Latency**：jevbench client 端量的整個 request 的 wall-clock，取 p50 與 p95（包含失敗的 request）。**沒有 TTFT 或 tokens/s。**
- Cost：依價格與 `usage` 計算；local-jev 一律傳零價格。

維護者在 board 上另外用的分數（`summarize` 不產生）：

- **Intelligence**：chance-corrected accuracy。
- **Calibration**：ECE 加上 TVD（只在 10 個帶 `gold_probs` 的 hard `probability` Tasks 上算）。spec 提到的 RPS 在這個 commit 沒有實作。
- **Speed**：`100 − 20·log10(s / 0.1 s)`，對 p50 與 p95 各算一次再平均。Self-hosted endpoint 的延遲先乘 2，在維護者自己的機器上再加 0.15 s。
- **Composite**：四個軸（加上 Cost）的加權調和平均，低於門檻時有平方的懲罰。Open-weights board 的主分數是 **Capability**：Intelligence 與 Calibration 的平均，並要在 cost cap（每 1,000 個 Decision USD 0.0646）與調整後的 median-latency cap（1.23 s）之內。

### 3.4 Reference 數字

jevbench 在固定 commit 裡公布的數字（`local-jev report` 也逐行附上出處）：

| System | Suite | Accuracy | Brier | ECE | Client p50 |
| --- | --- | --- | --- | --- | --- |
| Jev 1.13.0 | 231 個 public Tasks | 86.6% | 未公布 | 未公布 | 約 0.67 s |
| system-one-open（Gemma 4 E2B with LoRA，L4） | 231 個 public Tasks | 73.2% | 未公布 | 未公布 | 約 0.65 s |
| decider-2b | 231 個 public Tasks | 71.0% | 未公布 | 未公布 | 約 0.26 s |
| Jev 1.13.0 | 舊的 v1 suite（242 decisions） | 96.3% | 0.056 | 0.027 | 0.65 s |
| system-one-open | 舊的 v1 suite（242 decisions） | 90.1% | 0.138 | 0.068 | 0.65 s |

- **spec 引用的 reference（Jev 96.3%、Brier 0.056；system-one-open 90.1%、Brier 0.138）是舊的 242 題 v1 suite，不是 verdict 用的 231 個 public Tasks。** 在 public Tasks 上 jevbench 只公布每個 Task 的對錯與延遲，沒有 Distribution，所以沒有 Brier、ECE。「Accuracy ≥ 80%」這條標準落在 system-one-open（73.2%）與 Jev（86.6%）之間。
- 這三個系統在 easy 都是 48/48；差距幾乎全在 hard（Jev 81/111、system-one-open 54/111、decider-2b 55/111）。
- Liquid 的 d1-2B model card 宣稱在「JevBench v1.4.1」得到 77.9，沒有說是哪個指標，視為未驗證。
- Reference 的延遲是維護者自己的 client 到遠端 endpoint 的 wall-clock（沒有乘上 self-hosted 的 ×2），不能和本機 server 端的時間直接比。

### 3.5 怎麼跑

用 local-jev（詳見 README「跑 benchmark」）：

```sh
uv run local-jev serve --decider d1 --no-calibration   # 終端機 1：raw server
uv run local-jev bench                                 # 終端機 2：全部 231 個 public Tasks
uv run local-jev bench --task-set zh-tw                # 同樣 231 個 Tasks，State 是繁體中文
uv run local-jev calibrate runs/<run>                  # 擬合 calibration，寫 cross-validated 報告
uv run local-jev report runs/<run> runs/<run> …        # 比較報告與 verdict
```

`local-jev bench` 背後做的事，等同於直接在 jevbench 的 checkout 裡跑：

```sh
cd .cache/jevbench/609a664e81f46e03731600bd7bd9f1b7a61257c5
TYPESAFE_API_KEY=local python -m jevbench.cli run \
  --tasks datasets/public/easy.jsonl,datasets/public/original.jsonl,datasets/public/hard.jsonl \
  --adapter typesafe --endpoint http://127.0.0.1:8000 --model local-jev \
  --key-env TYPESAFE_API_KEY \
  --price-in-per-m 0 --price-out-per-m 0 --reserve-usd 0 \
  --results /abs/run/results.jsonl --raw-dir /abs/run/raw \
  --ledger /abs/run/ledger.jsonl --manifest /abs/run/manifest.json
python -m jevbench.cli summarize --tasks <同上> --results /abs/run/results.jsonl \
  --public-export /abs/run/summary.json
```

會踩到的眉角（`local-jev bench` 都已處理）：

- 一定要給 `--endpoint`：`typesafe` adapter 沒給時會直接打 TypeSafe 的正式 API（`https://api.typesafe.ai`）。給的是 base URL，不加 `/v1`。
- key 的環境變數空著時 adapter 一個 request 都不送：沒有 token 的 server 也要給一個假值。
- 未知 endpoint 沒有價格時，每個 Task 預留 $0.02、累計到 $15 就停：傳零價格與 `--reserve-usd 0`。
- 結果與 raw evidence 只能寫到新檔，而且必須在 jevbench 的 checkout 之外。
- 沒有 retry；401、403、429 立刻停止整個 run，連續三次失敗（含 5xx、連不上、HTTP 200 但答案解析失敗）也停止。所以 server 在模型載入完成前不能接流量，錯誤的 request 回 422（算答錯，但不計入連續失敗）。
- 回應裡任何地方出現 NaN 或 Infinity，整個 run 會 crash。
- jevbench 送出的 `choice` Criteria 的 key 是**字母順序**（資料檔用 `sort_keys` 寫出），作者原本的 Label 順序不會送到 wire 上。

## 4. Jev wire format（TypeSafe `/v1/systemone`）

jevbench 的 `typesafe` adapter 對每個 Task 送一個 `POST {endpoint}/v1/systemone`：

```json
{"state": "…純文字，或 JSON 物件…",
 "model": "…",
 "questions": {"decision": {"type": "choice",
                            "instructions": "Which intent does the user's message express?",
                            "criteria": {"billing_question": "Asks about a charge, invoice or payment",
                                         "track_order": "Wants to know where an order is or when it arrives"}}}}
```

- 每個 request 只有一個 Question，名稱一律是 `decision`。不送 `labels`、不送預期答案、不送 `explain`。
- `noul` 的 `criteria` 是 `{"true": …, "false": …}`，對應 jevbench 的 Labels `yes`/`no`。
- `score` 的 `criteria` 是等級說明的 list，由低到高；Labels 是等級 index 字串 `"0"`、`"1"`…。
- State 可以是字串，也可以是 JSON 物件（hard 有 35 個）。

回應（HTTP 200）：

```json
{"answers": {"decision": {"type": "choice", "choice": "track_order",
                          "probabilities": {"billing_question": 0.03, "track_order": 0.97}}},
 "usage": {"input_tokens": 73, "output_tokens": 0},
 "model": "…"}
```

| Question type | 答案 |
| --- | --- |
| `noul` | `{"type": "noul", "noul": <P(yes)>}`，一個 [0, 1] 的數字（不能是 JSON 的 `true`/`false`） |
| `choice` | `{"type": "choice", "choice": <Label>, "probabilities": {label: p, …}}` |
| `score` | `{"type": "score", "score": <等級>, "probabilities": {"0": p, "1": p, …}}` |

adapter 接受答案的條件：`answers.decision` 存在；`type` 等於 Question type；`noul` 是 [0, 1] 的非布林數字；`choice` 必須是 Label 之一而且 `probabilities` 是 dict；`score` 必須有 `probabilities` dict（`score` 欄位本身不檢查）。之後 scoring 再檢查 Distribution 的 key 與加總（3.3 節）。其他欄位一律忽略，**只有回應頂層的 `runtime` 例外**：jevbench 會把它原樣抄進 `results.jsonl` 的每一筆紀錄。

local-jev 的擴充（都不影響 jevbench）：

- 請求的 `explain`（預設 `false`）：`lfm3` 會在 Distribution 決定之後另外生成一句 Reason；`d1` 照常回答、不帶 Reason。benchmark 一律不開。
- 一個請求可以有好幾個具名的 Question，每個都用同樣的名稱回答。
- 回應的 `usage`、`timing`（server 自己量的 `total_ms`、`decider_ms`）與 `runtime`（Decider、model、revision、device、calibration，加上同樣的時間）。因為 jevbench 會抄 `runtime`，每一筆結果都記著它是在哪種硬體上、有沒有 calibration 跑出來的，報告的 server 端延遲也是從這裡來的。
- 不支援的 Question type（`text`、`number`、`date`、`box`、`boxes`、多選 `choice`）與格式錯誤的請求回 422。

## 5. 原始提案逐條查證

| 提案的說法 | 查證結果 | local-jev 的做法 |
| --- | --- | --- |
| JEV =「JSON Evaluation」，回答固定的 `is_applicable`/`category`/`confidence`/`reason` schema | **錯。** Jev 是 TypeSafe AI 的 decision model，是產品名，不是縮寫。jevbench 問的是有型別的 Question，每個 Task 有自己的 Label set，要的是 Distribution。 | 照 `/v1/systemone` 回答 `noul`、`choice`、`score`（第 4 節） |
| 為 jevbench 最佳化 TTFT 與 tokens/s | **錯。** jevbench 量整個 request 的 wall-clock p50/p95，另外量 accuracy、calibration 與 cost。 | 不生成任何 token：一次 forward pass 就得到整個 Distribution |
| Model 用 `LFM3-1.7B-Instruct` | repo 是 `LiquidAI/LFM3-1.7B`（已 post-trained，沒有後綴），自帶 remote code（`trust_remote_code`），**thinking 預設開啟**，context 256K。 | 固定 revision `0dc4268`，chat template 傳 `enable_thinking=False` |
| 用 vLLM + xgrammar serve | **目前做不到。** 沒有任何 vLLM release 能載入 LFM3：支援在還沒 merge 的 vllm-project/vllm#64959；寫 spec 時最新是 v0.33.0rc3，之後的 v0.33.0 正式版（2026-11-02）registry 裡仍沒有 `Lfm3ForCausalLM`。 | transformers backend，放在 backend seam 後面（ADR-0001） |
| ChatML prompt template | 大致對。需要 BOS `<|startoftext|>` 與空的 think block `<think></think>`，草稿少了一個 `<|im_end|>`。 | 一律用 model 自己的 chat template 產生 prompt |
| Prefill `{` | 與 JSON-schema grammar 衝突：grammar 會要求再生成一次 `{`。 | 不生成 JSON，問題不存在 |
| `reason` 用 `pattern` 加 `maxLength` | xgrammar 在 `pattern` 與 `minLength`/`maxLength` 並用時會丟掉長度限制；vLLM 的 auto 模式會 fall back 到 guidance。 | Reason 是有 token 上限（64）的純文字，server 做 JSON escape |
| `confidence` 用 `minimum`/`maximum` | xgrammar 支援。但這是 Verbalized probabilities：數字是 model 猜的，不是它的 likelihood。 | Native probabilities（ADR-0002） |
| 讓 compiler cache 保持溫熱 | `GrammarCompiler` 預設就有 cache（`cache_limit_bytes`）。 | 不用 xgrammar |
| categorical 欄位用 temperature 0、`reason` 用 0.2 | sampling 參數是 per request 的；要 per-field temperature 得自己寫 logits processor。 | 打分數不 sampling；只有 Reason 用 card 的 sampling 預設 |
| 在 `\n` 或 `}` 停止 | 有 grammar 時是多餘的，而且 `\n` 可能把 JSON 切斷。 | 不需要 |
| JSON 寫到一半 context overflow | 256K context 下不是真正的風險；真正的風險是 `max_tokens` 截斷，用有上限的輸出解決。 | 打分數不生成；Reason 有上限 |
| 這樣就能拿到「領先的指標」 | 實測見第 7、8 節。 | 沒有達到：最好的 d1-2B 是 76.6%，Jev 是 86.6%；1.7B 的 Chat model 只有 62.3% |

## 6. 模型與 backend 的發現

### 6.1 LFM3-1.7B（Chat model，`lfm3` Decider）

- `LiquidAI/LFM3-1.7B`，1.73B 參數，context 262,144，自帶 remote code（transformers 5.22.1 還沒有原生的 LFM3 class），tokenizer 也要 `trust_remote_code=True`，否則 transformers 會跳出互動式的確認。
- Chat template 自己加 BOS；`enable_thinking=False` 時 generation prompt 結尾是 `<|im_start|>assistant\n<think></think>`，後面沒有換行。
- **Option-ID scoring 的 token 事實**：直接接在 `</think>` 之後讀下一個 token 時，`A`–`Z`、`yes`/`Yes`/`no`/`No`、`0`–`9`（都不加前導空白）各自剛好是一個 token；`YES` 是兩個 token，前面加空白的數字也是兩個 token。`lfm3` 啟動時會檢查這件事，換 model 時不成立就 health 回報 `failed`。
- **`noul` 必須合併大小寫**：實測 `Yes` 拿走了整個 vocabulary 0.92 的機率，`yes` 只有 0.08；只讀小寫等於在 7% 的機率上正規化。`lfm3` 對每個 Label 合併兩種寫法（logsumexp）。
- **選項位置偏差**：研究時用同一個三選一問題試了全部 6 種選項順序，LFM3 每次都選兩個合理選項中排在後面的那個，機率 0.85–1.00。在 231 個 public Tasks 上偏的方向反過來，是偏向第一個選項（A），而且偏向答 yes（7.4 節）。兩者都是 spec 預期的 letter-position bias；temperature calibration 修不了 argmax。
- MPS bf16 上一次 forward pass 約 85–100 ms（prompt 約 85–100 tokens），記憶體約 4.1 GB；容器 CPU fp32 約 1.1–1.3 s。
- 在 CUDA 上 LFM3 的 width-7 conv kernel 用 Triton（run time 編譯，所以 runtime image 保留 gcc）；MPS 與 CPU 用 torch 的 `nn.Conv1d`。

### 6.2 d1-2B（Decision model，`d1` Decider）

- `LiquidAI/d1-2B`，用 `AutoModel` 加 `trust_remote_code` 載入，需要 `transformers>=5.19`、`torchvision`、`pillow`。它不是 Chat model：沒有 chat template 可用、不生成文字、沒有 Reason（ADR-0003）。
- `system_one(state, questions)` 一次 forward pass 回答。它內部其實也是在 LFM3 backbone 上做 option-code 的 readout（`A`/` A`、`yes|Yes|YES`、`0`–`9`），只是有專門訓練的 decision head。
- **d1 內部已經先除過一次 temperature T ≈ 1.337**（revision 裡的 `calibration.json`，套在 `noul`/`choice`/`score` 上）。local-jev 的 per-type temperature 疊在 d1 送出的機率之上。
- **d1 的 `score` 欄位是 float 的中位數等級**（第一個累積機率到 0.5 的等級），不是 argmax；local-jev 從 `probabilities` 自己算 argmax 填 wire 的 `score`。
- 同一次 `system_one` 呼叫裡放好幾個 question 時，d1 把它們塞進同一個 sequence，答案會因為其他 question 而偏移約 0.03；local-jev 每個 Question 各呼叫一次。
- d1 在 prompt 裡只顯示 Criteria 的說明，不顯示 Label 名稱；`lfm3` 兩者都顯示。
- 換 transformers 版本時要小心：一旦 transformers 出了原生的 LFM3 class，d1 的 remote code 會自動改用它，固定 revision 也擋不住，所以 transformers 版本鎖在 lockfile 裡，升級後要重跑 `real_model` 測試。

### 6.3 vLLM

- vLLM 0.33.0（2026-11-02）仍不能載入 LFM3；支援在 vllm-project/vllm#64959，尚未 merge。
- jevbench 一次只送一個 request，vLLM 的 continuous batching 與 paged attention 幫助有限。v1 因此用 Hugging Face transformers，backend 介面只有「候選 token 的 next-token 機率」與「有上限的純文字生成」兩個能力，之後 vLLM 支援時可以換掉而不動 decision logic（ADR-0001）。
- macOS 上的 vLLM 只能用 CPU，而且要從 source build。

### 6.4 xgrammar 與 grammar-constrained decoding

- Option-ID scoring 讓 Distribution 在建構時就加總為 1，一次 forward pass 得到整個答案，model 從不寫 JSON，所以 grammar-constrained decoding 沒有事可做，xgrammar 不是相依套件（ADR-0002）。jevbench 自己的結果也顯示，算術出錯（加總不是 1）的只有把機率寫成文字的系統。
- 提案裡 xgrammar 的細節（`pattern` 加長度限制被丟掉、compiler cache、per-field temperature）見第 5 節。

### 6.5 平台與封裝

- **macOS 上沒有任何 container runtime（Docker Desktop、OrbStack、Podman、Lima）能讓 Linux 容器用 Metal/MPS**，OrbStack 完全沒有 GPU 支援。所以延遲只在原生 `uv run`（MPS）上量；容器裡的 CPU 推論每個 Decision 約 1 秒，長 State 要 15–30 秒，只適合開發與驗證。
- OrbStack 給容器 15.7 GiB 與 12 CPUs；兩個 Decider 在 CPU fp32 加起來約 15 GB，所以 compose 一次只跑一個。
- **PyTorch 的 CUDA 12.8 index 最新只到 torch 2.11.0 / torchvision 0.26.0**，所以三個平台都固定在 torch 2.11（Mac 從 PyPI、linux/amd64 從 cu128、linux/arm64 從 CPU index）。**PyPI 的 Linux torch wheel 在 aarch64 上也會帶進 CUDA 13 的 library**，所以 linux/arm64 要明確用 PyTorch 的 CPU index。`uv lock`/`uv sync` 沒有 `--torch-backend`，要用 explicit index 加 platform marker。
- GPU 路徑（`compose.gpu.yaml`、CUDA 12.8 的 image）已交付但**沒有在真的 GPU host 上驗證**，因為目前沒有 GPU host。

### 6.6 授權限制：LFM Open License v1.0

**LFM3-1.7B 與 d1-2B 都採用 LFM Open License v1.0。年營收達 1,000 萬美元（USD 10,000,000）或以上的法人，商業使用（Commercial Use）不在授權範圍內。** 授權第 5 節：商業使用的權利以「你或你的法人沒有超過 Threshold（年營收 1,000 萬美元或以上）」為條件；超過 Threshold 的法人做任何商業使用都沒有授權。符合資格的非營利組織做非商業或研究用途不受這個門檻限制。

這表示：local-jev 的程式碼可以自由使用，但**以這兩個 model 部署到營收達門檻的公司做商業用途，需要另外向 Liquid AI 取得授權**，或換成授權允許的其他 ≤ 3B model（`--model`/`--revision` 不需改程式就能換）。jevbench 的 Tasks 與繁體中文副本是 MIT，不受影響。

## 7. 實測結果

### 7.1 量測方式與環境

- 硬體與軟體：Apple M3 Pro（36 GB）、macOS，原生 `uv run`（不在容器裡），`--device auto` 選到 MPS、bfloat16；torch 2.11.0、transformers 5.22.1、jevbench `609a664`。`HF_HUB_OFFLINE=1`，兩個 model 都已在快取裡（載入加暖機：d1 3.9 s、LFM3 4.1 s）。
- 一次只跑一個 server，一律 `serve --no-calibration`（送 raw Distribution）。每個 Decider 各跑一次 `bench --task-set public`（231 個英文 Tasks）與一次 `bench --task-set zh-tw`（同樣 231 個 Tasks，State 是繁體中文），共 4 個 run，每個約 2–3 分鐘。jevbench 一次送一個 request，沒有失敗、沒有 retry。
- Calibration：對每個 Decider 的英文 public run 跑 `local-jev calibrate`，寫進 committed 的 `src/local_jev/calibrations/d1.json`、`lfm3.json`；對繁體中文 run 用 `calibrate --calibrations-dir`（package 之外的目錄）只取 cross-validated 報告，不覆寫英文的 calibration。最後用 `local-jev report` 把 4 個 run 合成比較報告。
- **同時在跑的東西**：4 個 run 期間沒有任何容器在跑（每 5 秒檢查一次 `docker ps`）；Mac 上開著的是一般桌面程式（VS Code、Chrome、Slack、Postman、Surfshark VPN、閒置的 OrbStack），每個 run 開始與結束時的 `ps` 快照裡，沒有其他程式持續佔用超過一個 CPU core。
- 一個被排除的 run：第一次的 LFM3 繁體中文 run（`20261105T100101Z-zh-tw`）期間，ticket 11 的 dev container 在 10:02:42 到 10:03:23 之間（UTC）在跑，所以它的延遲不採用，另外在沒有容器時重跑。兩次的 231 個 predicted Label 完全一樣（client p50 是 155 ms 對 167 ms）。d1 的英文 run 也和 ticket 07 開發時的 run 得到完全相同的 Accuracy、Brier、ECE 與 temperature：在同一台機器上結果是可重現的。

### 7.2 主要數字

`local-jev report` 的 Results 表（[`docs/comparison-report.md`](comparison-report.md)），Client 是 jevbench 量的 wall-clock，Server 是每個答案 `runtime.total_ms`，單位 ms：

| Decider | Calibration | Language | Accuracy | Brier | ECE | Valid answers | Client p50 / p95 | Server p50 / p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| d1 | raw | English | 76.6% | 0.285 | 0.072 | 100.0% | 129 / 2181 | 128 / 2180 |
| d1 | calibrated, 5-fold CV | English | 76.6% | 0.275 | 0.060 | 100.0% | 129 / 2181 | 128 / 2180 |
| lfm3 | raw | English | 62.3% | 0.626 | 0.272 | 100.0% | 163 / 2342 | 162 / 2340 |
| lfm3 | calibrated, 5-fold CV | English | 62.3% | 0.495 | 0.038 | 100.0% | 163 / 2342 | 162 / 2340 |
| d1 | raw | Traditional Chinese | 78.8% | 0.291 | 0.069 | 100.0% | 153 / 3397 | 151 / 3395 |
| d1 | calibrated, 5-fold CV | Traditional Chinese | 78.8% | 0.287 | 0.057 | 100.0% | 153 / 3397 | 151 / 3395 |
| lfm3 | raw | Traditional Chinese | 58.0% | 0.693 | 0.302 | 100.0% | 167 / 3574 | 165 / 3572 |
| lfm3 | calibrated, 5-fold CV | Traditional Chinese | 58.0% | 0.522 | 0.050 | 100.0% | 167 / 3574 | 165 / 3572 |

- Temperature scaling 不改變 argmax，所以 calibrated 的 Accuracy 與 raw 相同；Valid answers 與延遲沿用同一個 raw run。
- 統計上的不確定性：231 個 Tasks 的 Accuracy，95% 信賴區間（Wilson）約 ±5 個百分點：d1 70.8–81.6%，LFM3 55.9–68.3%。d1 的 raw Brier 用 bootstrap 估的 95% 區間是 0.225–0.346。d1 與 LFM3 的差距很顯著（同一批 Tasks 上 d1 答對而 LFM3 答錯 47 題，反過來 14 題，McNemar p < 0.001）。
- 兩個 Decider 的 Valid answers 都是 100%：Option-ID scoring 與 d1 的 head 都在建構時就讓 Distribution 在 Label set 上加總為 1。

### 7.3 Calibration

用全部 231 個英文 Tasks 擬合、committed 的 temperature（T > 1 代表原本過度自信）：

| Decider | `choice` | `noul` | `score` |
| --- | --- | --- | --- |
| d1 | 1.0321 | 2.1097 | 1.2695 |
| lfm3 | 3.0394 | 12.4246 | 3.7641 |

5-fold cross-validated，raw → calibrated，按 Question type：

| Decider | Question type | Tasks | Accuracy | Brier | ECE |
| --- | --- | --- | --- | --- | --- |
| d1 | `choice` | 139 | 78.4% | 0.253 → 0.258 | 0.056 → 0.077 |
| d1 | `noul` | 74 | 73.0% | 0.357 → 0.313 | 0.160 → 0.070 |
| d1 | `score` | 18 | 77.8% | 0.239 → 0.247 | 0.131 → 0.109 |
| lfm3 | `choice` | 139 | 64.0% | 0.584 → 0.472 | 0.239 → 0.070 |
| lfm3 | `noul` | 74 | 62.2% | 0.660 → 0.489 | 0.315 → 0.096 |
| lfm3 | `score` | 18 | 50.0% | 0.815 → 0.691 | 0.350 → 0.173 |

- d1 本身已經相當校準（它內部先除過 T ≈ 1.337），只有 `noul` 明顯過度自信；calibration 讓整體 Brier 從 0.285 降到 0.275、ECE 從 0.072 降到 0.060。`choice` 與 `score` 的 CV 數字略為變差，是 fold 之間 T 不穩定加上 `score` 只有 18 題的雜訊。
- LFM3 在三種 Question type 都嚴重過度自信。`noul` 的 T = 12.4，等於把 Distribution 幾乎壓平：它的 `noul` 答案接近隨機（calibrated Brier 0.489，而永遠回答 0.5 的 Brier 是 0.5）。繁體中文 run 的 `noul` T 甚至撞到搜尋上限 20。
- **Brier 的瓶頸是 Accuracy，不是 calibration。** d1 的 raw Brier 有 86% 來自答錯的 54 題。一個答錯的 Decision，Brier 至少 0.5（正確 Label 最多和 argmax 平手），所以 54 題錯就讓平均 Brier 至少 0.117；要壓到 0.20，答錯的題目得幾乎都停在五五波，而答對的 177 題又得非常有把握。這是辨別力（discrimination）的問題，per-type temperature 做不到（它只降到 0.275），實際上要靠更高的 Accuracy。

### 7.4 細項：難度與題型

答對的題數，按 Task set 的部分（附 jevbench 公布的 reference）：

| 系統 | easy（48） | original（72） | hard（111） | 合計（231） |
| --- | --- | --- | --- | --- |
| Jev 1.13.0 | 48 | 71 | 81 | 200（86.6%） |
| **d1（local-jev）** | **48** | **68** | **61** | **177（76.6%）** |
| system-one-open | 48 | 67 | 54 | 169（73.2%） |
| decider-2b | 48 | 61 | 55 | 164（71.0%） |
| lfm3（local-jev） | 46 | 49 | 49 | 144（62.3%） |

- d1 在 easy 全對、original 94.4%，跟 Jev 只差 3 題；差距在 hard（55.0% 對 Jev 73.0%）。
- d1 最弱的 hard family：`temporal_numeric` 2/15、`long_policy` 6/19、`probability` 5/10、`judge_hard` 10/17、`multi_hop` 11/18。這些需要日期與數字推理、長文件裡找條款，或多步推論，是 2B 模型一次 forward pass 最難的部分。
- LFM3 連 original 都只有 68.1%（`ordinal` 6/12、`adequacy` 6/12），`trap` 只有 1/8。

**LFM3 的偏差**（從 run 的 Distribution 算出）：

- **位置偏差（primacy）**：jevbench 送來的 Criteria 是字母順序，選項依序標成 A、B、C…。139 個 `choice` 裡 LFM3 選 A 57 次，但 A 是正確答案的只有 30 次；答錯的 50 題裡有 38 題選了比正確答案更前面的選項。研究時的三選一 probe 看到的是 recency（偏向後面的選項）；在真正的 Tasks 上偏的是第一個選項。兩者都是 spec 預期的 letter-position bias。d1 也有輕微的同方向偏差（選 A 38 次，對 30 次）。
- **偏向 yes**：74 個 `noul` 裡 LFM3 答 yes 59 次，正確答案是 yes 的只有 35 次（d1：41 次）。
- **過度自信**：LFM3 答錯時的平均 top-label confidence 是 0.83，有 40 題在 ≥ 0.9 的 confidence 下答錯（d1：平均 0.60，只有 3 題）。

兩個 Decider 的互補性：英文 public Tasks 上，任一個答對的有 191 題（82.7%），但無法事先知道該信哪一個，而且一個 process 只跑一個 Decider（ADR-0003），所以這只是上限的參考，不是可用的數字。

### 7.5 繁體中文

- d1：英文 76.6%、中文 78.8%。兩次 run 的 predicted Label 有 205/231 相同；只有英文答對的 7 題、只有中文答對的 12 題（McNemar p = 0.36），**看不出中文比較差**。Brier 0.291（CV 0.287）與 ECE 也跟英文差不多。
- LFM3：英文 62.3%、中文 58.0%（只有英文答對 19 題、只有中文答對 9 題，p = 0.09），稍差，但本來就不可用。
- 中文 State 的 prompt 比英文長約 27%（d1 的 231 個 prompt 合計 174k 對 137k tokens），所以 p95 延遲較高（d1 server p95 3.4 s 對 2.2 s）。
- 翻譯由 LLM agents 依翻譯指引完成，自動檢查都通過，但**還沒有人工 spot-check**（`src/local_jev/zh_tw/README.md` 列了 10 個要看的 Task）。中文數字在 spot-check 之前只能當參考。

### 7.6 延遲

- Server 端 p50（MPS）：d1 128 ms、LFM3 162 ms。按 Question type：d1 `choice` 133、`noul` 171、`score` 107 ms；LFM3 164、162、112 ms。
- p95 約 2.2–2.3 s，來自 hard 的長 State：prompt 的中位數約 110–140 tokens，但有 38–39 個 Tasks 超過 1,000 tokens（最長約 3,700），它們的中位延遲約 2.1–2.2 s，約每個 prompt token 0.8–0.9 ms；300 tokens 以下的 prompt 中位數是 d1 96 ms、LFM3 114 ms。
- Client 端與 server 端只差 1–2 ms：本機 HTTP 與 jevbench client 的開銷可以忽略。
- 這些數字只代表 M3 Pro 的 MPS。容器裡的 CPU 每個 Decision 約 1 秒、長 State 15–30 秒，不具代表性。jevbench board 的 Speed 分數是維護者從自己的 client 量的（self-hosted 再乘 2），也不能直接拿來比。

### 7.7 跟提案與 reference 的比較

- 提案說的「領先指標」沒有出現：d1 在 public Tasks 上 Accuracy 比 Jev 1.13.0 低 10 個百分點，hard 低 20 題；它高於同樣是小模型的 system-one-open（Gemma 4 E2B with LoRA）與 decider-2b。
- 提案主張的路線（1.7B 的 Chat model 加 grammar-constrained JSON）在 v1 改成了更有利的 Option-ID scoring（Native probabilities、一次 forward pass、Valid 100%），LFM3-1.7B 仍只有 62.3%。Valid answers 100% 表示輸出格式已經不是問題；剩下的是 1.7B Chat model 本身的判斷力與它的位置偏差。
- Liquid 在 d1-2B model card 上宣稱「JevBench v1.4.1」77.9，但沒說是哪個指標，而 jevbench 固定 commit 的 v1.4.x 結果裡沒有 d1-2B 這一行，仍無法驗證。我們量到的 d1 public Tasks Accuracy 是 76.6%。

## 8. Feasibility verdict

`local-jev report` 對 4 個 run 的判定，逐字照錄（只看英文、完整 231 個 public Tasks；繁體中文不影響 verdict）：

| 標準 | Status | Evidence |
| --- | --- | --- |
| 1. Valid answers 100% for every Decider | **pass** | d1 LiquidAI/d1-2B@652f920: 100.0% (20261105T100421Z-public); lfm3 LiquidAI/LFM3-1.7B@0dc4268: 100.0% (20261105T100910Z-public) |
| 2. The best Decider reaches Accuracy ≥ 80% with Brier ≤ 0.20 | **fail** | d1 LiquidAI/d1-2B@652f920: Accuracy 76.6%, Brier 0.275 (calibrated, 5-fold CV, 20261105T100421Z-public); lfm3 LiquidAI/LFM3-1.7B@0dc4268: Accuracy 62.3%, Brier 0.495 (calibrated, 5-fold CV, 20261105T100910Z-public) |
| 3a. Server-side p50 on the Mac (MPS) is recorded for every Decider | **pass** | d1 LiquidAI/d1-2B@652f920: server p50 128 ms (20261105T100421Z-public); lfm3 LiquidAI/LFM3-1.7B@0dc4268: server p50 162 ms (20261105T100910Z-public) |
| 3b. Server-side p50 ≤ 150 ms on an L4-class GPU | **pending** | no run of d1 LiquidAI/d1-2B@652f920 on cuda on the 231 public Tasks |

**Overall: fail** (fail: 2; pending: 3b)

說明：

- **標準 1 pass**：兩個 Decider 的 462 個答案全部被 jevbench 接受。
- **標準 2 fail**：最好的 d1 是 Accuracy 76.6%、cross-validated Brier 0.275，兩個門檻都沒過；LFM3 差更多。Temperature scaling 不會改變 argmax，所以 calibration 不可能讓 Accuracy 過 80%。
- **標準 3a pass**：M3 Pro MPS 上的 server 端 p50 已記錄（d1 128 ms、LFM3 162 ms）。這條沒有門檻。
- **標準 3b pending**：沒有 GPU host，預設 Decider（d1）沒有 CUDA run。M3 Pro 的 MPS 上 d1 已經是 128 ms，L4 上達到 ≤ 150 ms 很可能，但這只是推測，要量過才算數。即使 3b 之後 pass，標準 2 的 fail 也讓 overall 維持 fail。

## 9. 預設 Decider

依議定的順序（在 231 個英文 public Tasks 上：答案全部有效 → 達到標準 2 → Accuracy 較高 → cross-validated Brier 較低），`local-jev report` 選出：

**Default Decider: d1 LiquidAI/d1-2B@652f920**：Accuracy 76.6%，Brier 0.275（calibrated, 5-fold CV，20261105T100421Z-public）。

兩者答案都全部有效、也都沒達到標準 2，所以由 Accuracy 決定（76.6% 對 62.3%）；Brier（0.275 對 0.495）也是 d1 較好。據此：

- `local-jev serve` 的 `--decider` 預設改為 `d1`（`src/local_jev/cli.py` 的 `DEFAULT_DECIDER`，`tests/test_serve_default_decider.py` 驗證沒給 `--decider` 時回答的是 d1），README 的「選擇 Decider」已註明。
- committed 的 calibration（`src/local_jev/calibrations/d1.json`、`lfm3.json`）是這次英文 public run 擬合的，`serve` 預設就會套用。
- `lfm3` 仍然保留：它是唯一能寫 Reason 的 Decider，也是換其他 ≤ 3B Chat model 的入口。

## 10. 與 ADR 不一致之處

**沒有觀察到推翻任何 ADR 所寫理由的事實**，但 ADR-0002 有一個隱含前提與量到的結果不一致，值得重新評估。

- **ADR-0001**（transformers 先、vLLM 之後，放在 backend seam 後面）：成立。vLLM 0.33.0 仍不能載入 LFM3；jevbench 嚴格地一次送一個 request，transformers 在 MPS 上的 server 端 p50 是 128–162 ms。
- **ADR-0002**（Native probabilities via Option-ID scoring）：ADR 寫下的理由都成立：Valid answers 100%、加總為 1、每個 Decision 一次 forward pass、不生成任何 token（`output_tokens` 一律是 0）。但量到的結果與它的一個隱含前提不一致：

  > _與 ADR-0002 的隱含前提「Option-ID scoring 足以讓 1.7B 的 Chat model 成為可用的 Decider」不一致，值得重新評估，因為 LFM3-1.7B 只有 62.3%，而且明顯偏向第一個選項（139 個 `choice` 裡選 A 57 次，A 正確的只有 30 次）與 yes；這正是 ADR 為了維持一次 forward pass 而沒有處理的部分。_

  ADR 放棄的 Label scoring，以及選項順序平均（permutation averaging），如果之後還要走 Chat model 這條路，應該重新評估並量它們的延遲成本。這對預設的 d1 沒有影響（d1 用的是自己訓練過的 head）。
- **ADR-0003**（兩個 Decider、一個 process 一個）：成立。一個 process 一個 Decider 讓延遲數字乾淨（每個 run 只有一種 model、一種硬體）。預設改成 d1 之後，預設的 server 不提供 Reason；這是 ADR-0003 已接受的取捨（不為了 Reason 再配一個 Chat model）。

與 ADR 以外的文件不一致的地方（已在這次修正或註明）：

- README 與 `src/local_jev/option_id.py` 的 docstring 原本把 LFM3 的風險寫成 recency bias（來自一個三選一的 probe）；在 231 個 Tasks 上量到的是偏向第一個選項。兩處都已補上量到的結果。spec 的 Known risks 只寫 letter-position bias、沒有寫方向，與量到的一致。
- spec 的 reference（Jev 96.3%、Brier 0.056）是舊的 242 題 v1 suite，不是 231 個 public Tasks（ticket 09 已在比較報告裡標明；這份報告用 public Tasks 的 86.6%）。
- spec 的 jevbench 說明沒有寫出維護者；維護者是 Benchmark Heaven，不是 TypeSafe AI（3.1 節）。

## 11. 建議的下一步

依議定標準目前是 fail，所以**不建議照原計畫加碼投資**。如果仍要往前走，建議的順序：

1. **先決定要不要用 LFM 授權的 model。** 開發者所屬的法人年營收若達 1,000 萬美元，LFM3 與 d1 都不能商業使用；要嘛向 Liquid AI 取得授權，要嘛改評估其他授權允許的 ≤ 3B model（`--model`/`--revision` 不需改程式，但要自己確認那個 model 的授權）。這一步決定後面的工作值不值得做。
2. **以 d1 為基礎，針對 hard 提升 Accuracy。** 標準 2 只能靠 Accuracy 過關（Brier 的 86% 來自答錯的題目）。d1 在 easy 與 original 已接近 Jev，差在 `temporal_numeric`、`long_policy`、`probability`、`judge_hard`、`multi_hop`。先對這些錯誤做 error analysis；再試新的 d1 checkpoint 或其他 ≤ 3B 的 Decision model（只要換 `--model`/`--revision`，跑 `bench`、`calibrate`、`report`）。Fine-tuning 或 LoRA 不在 v1 範圍內；要做的話不能用 jevbench 的 public Tasks 訓練再拿它們評估（jevbench 也要求揭露在 public split 上的訓練）。注意 system-one-open（Gemma 4 E2B 加 LoRA）在 public Tasks 上也只有 73.2%，LoRA 不保證有用。
3. **在 L4 等級的 GPU host 上跑一次，結束標準 3b 的 pending。** `docker compose -f compose.yaml -f compose.gpu.yaml --profile d1 up -d --wait`，然後 `local-jev bench`、`local-jev report`；同時驗證 `compose.gpu.yaml`、CUDA 版的 image 與 `real_model` 測試（LFM3 在 CUDA 上會用 Triton kernel），這些都還沒在真的 GPU 上跑過。
4. **不再投資 LFM3-1.7B 的 Option-ID Decider 本身。** 它只在需要 Reason 時有用。若 Chat model 這條路仍重要，先重新評估 ADR-0002 的選項（permutation averaging、Label scoring）並量它們的延遲成本，或換其他 ≤ 3B 的 Chat model 再量一次位置偏差。
5. **人工 spot-check 繁體中文 Task set**（`src/local_jev/zh_tw/README.md` 的 10 個 Task），之後中文數字才能正式引用。
6. **注意樣本數。** 231 個 Tasks 的 Accuracy 有約 ±5 個百分點的不確定性，小於這個幅度的改進要更多資料才能確認。正式上 board 需要維護者對 endpoint 跑 sealed Tasks，不在 v1 範圍內。
7. （選用）Mac 上長 State 的延遲：p95 約 2.2 s 來自上千 tokens 的 prefill。若在 Mac 上的延遲重要，可以試 d1 的 `model.compile()`（研究時短 prompt 從約 80–100 ms 降到約 68 ms），但 jevbench 的延遲標準看的是 GPU。

## 附錄：重現與原始資料

重現這份報告的數字（在 repo 根目錄，Mac 上原生執行；一次只開一個 server）：

```sh
export HF_HUB_OFFLINE=1                                    # 兩個 model 都已下載時
uv run local-jev serve --decider d1 --no-calibration       # 終端機 1
uv run local-jev bench --task-set public                   # 終端機 2
uv run local-jev bench --task-set zh-tw
# 停掉 d1，換 LFM3
uv run local-jev serve --decider lfm3 --no-calibration
uv run local-jev bench --task-set public
uv run local-jev bench --task-set zh-tw

uv run local-jev calibrate runs/<d1 public>                # 寫 src/local_jev/calibrations/d1.json
uv run local-jev calibrate runs/<lfm3 public>              # 寫 lfm3.json
uv run local-jev calibrate runs/<d1 zh-tw> --calibrations-dir /tmp/zh-tw-calibrations
uv run local-jev calibrate runs/<lfm3 zh-tw> --calibrations-dir /tmp/zh-tw-calibrations
uv run local-jev report runs/<d1 public> runs/<lfm3 public> runs/<d1 zh-tw> runs/<lfm3 zh-tw> \
  --output docs/comparison-report.md
```

原始資料（不 commit，放在主 repo 的 git-ignored 目錄 `/Users/user/Documents/workspace/local-jev/runs/`）：

| Run | Decider | Task set | 用途 |
| --- | --- | --- | --- |
| `20261105T100421Z-public` | d1 | public | 英文結果、`d1.json` 的來源 |
| `20261105T100910Z-public` | lfm3 | public | 英文結果、`lfm3.json` 的來源 |
| `20261105T100622Z-zh-tw` | d1 | zh-tw | 繁體中文結果 |
| `20261105T101118Z-zh-tw` | lfm3 | zh-tw | 繁體中文結果 |
| `excluded/20261105T100101Z-zh-tw` | lfm3 | zh-tw | 量測時有容器在跑，不採用（7.1 節） |
| `zh-tw-calibrations/` | | | 從繁體中文 run 擬合的 temperature，只供參考，不 commit |

每個 run 目錄裡有 jevbench 的 `results.jsonl`、`summary.json`、`raw/`（每個 Task 完整的 request 與 response）、local-jev 的 `decisions.jsonl`、`run.json` 與 `calibrate` 寫的 `calibration-report.json`（格式見 README）。
