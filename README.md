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
| `--calibrations-dir` | committed 的 `src/local_jev/calibrations/` | 從這裡讀這個 Decider 的 calibration（`<decider>.json`），見下方「Calibration」 |
| `--no-calibration` | 關 | 不套用 calibration，送出 raw Distribution；要給 `local-jev calibrate` 擬合的 benchmark run 必須這樣跑 |

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
             "calibration": null,
             "total_ms": 98.512, "decider_ms": 97.904}}
```

- `usage`：所有 Question 加總的 token 數。沒有要求 Reason 時，兩個 Decider 都只讀 prompt 不生成，所以 `output_tokens` 是 0；`lfm3` 寫 Reason 時，`output_tokens` 是 Reason 的 token 數，`input_tokens` 也加上寫 Reason 時另外讀的 prompt。
- `timing`：server 自己量的時間（毫秒）。`total_ms` 從解析完請求到產生回應為止（包含等前一個請求的時間），`decider_ms` 是其中花在 Decider 上的部分。拿來跟 client 端的 wall-clock 對照，就能分出網路與 client 的額外開銷。
- `runtime`：同樣的時間，再加上 Decider 名稱、model id、revision、device（`cuda`、`mps` 或 `cpu`），以及 `calibration`：server 套用的 calibration（`{"temperatures": {...}, "fitted_from": "<run>"}`），送出 raw Distribution 時是 `null`（見下方「Calibration」）。jevbench 會把回應頂層的 `runtime` 原樣抄進 `results.jsonl` 的每一筆紀錄，所以每筆結果都帶著它是在什麼硬體上、有沒有 calibration 跑的。

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
- 沒設時，帶任何 token 或不帶都接受。jevbench 的 `typesafe` adapter 預設一定會送一個 token（環境變數空著它就不肯跑），所以本機跑 benchmark 時隨便設一個假值即可，例如 `TYPESAFE_API_KEY=local`。`local-jev bench` 會自動處理（見下節）。
- Server 有設 token 時，把同一個值放進 jevbench `--key-env` 指定的環境變數。用 `local-jev bench` 時，只要在同一個 shell 設好 `LOCAL_JEV_TOKEN` 即可。

## 跑 benchmark（jevbench）

`local-jev bench` 用 jevbench 自己的 `typesafe` adapter 評測一個**已經在跑**的 server，一個指令完成 run 與 summarize。請在 repo 根目錄執行（預設路徑都相對於目前目錄）。

```sh
uv run local-jev serve --decider d1      # 終端機 1：先啟動 server（任何 Decider 都可以）
uv run local-jev bench                   # 終端機 2：跑全部 231 個 public Tasks
uv run local-jev bench --task-set hard   # 只跑 hard
uv run local-jev bench --limit 10        # smoke run：10 個 Tasks，不到一分鐘
```

它會依序：

1. 把 jevbench 抓到固定的 commit `609a664e81f46e03731600bd7bd9f1b7a61257c5`，放在 `.cache/jevbench/<commit>/`（git-ignored）。快取的 checkout 若已在這個 commit 就直接重用，不再連網；不對（或抓到一半中斷）就重新抓。要改從 mirror 抓，設 `LOCAL_JEV_JEVBENCH_URL=<git URL>`。
2. 等 server 的 `/health` 回報 ready（Decider 載入中最多等 10 分鐘）。連不上 server、或 Decider 載入失敗，就直接結束、不建立 run 目錄。jevbench 連續三次失敗就會停止整個 run，所以絕不能讓它先碰到還沒就緒的 server。
3. 在 `runs/` 底下開一個新的目錄，用 jevbench 跑選定的 Task set，再 summarize。

| 選項 | 預設 | 說明 |
| --- | --- | --- |
| `--endpoint` | `http://127.0.0.1:8000` | Server 的 base URL（不要加 `/v1`） |
| `--task-set` | `public` | `easy`（48）、`original`（72）、`hard`（111）、`public`（三者合計 231） |
| `--limit N` | 全部 | 只跑 N 個 Tasks：輪流從每個（Task set 的部分 × Question type）群組依檔案順序各取一個，所以 `--limit 10` 也會涵蓋 `choice`、`noul`、`score` 三種。 |
| `--runs-dir` | `runs` | Run 目錄放在這裡（git-ignored） |
| `--cache-dir` | `.cache` | jevbench 的快取放在這裡（git-ignored） |

Server 有設 `LOCAL_JEV_TOKEN` 時，在跑 `bench` 的 shell 設同一個值；沒設時 `bench` 會給 jevbench 一個假的 key。

跑完會印出 run 目錄和一行重點數字，例如：

```text
[local-jev] run: runs/20261105T090000Z-public
[local-jev] d1 LiquidAI/d1-2B@652f920 on mps, raw, Task set public, 231 Tasks: Valid 100.0%, Accuracy …, Brier …, ECE …, client p50 … s, p95 … s
```

`raw` 表示 server 送出的是 raw Distribution；server 有套用 calibration 時會寫 `calibrated`。

### Run 目錄裡有什麼

每次 run 都是一個新目錄 `runs/<UTC 時間>-<Task set>[-limit<N>]`（同一秒內再跑一次會加 `-2`），絕不覆寫之前的 run。

| 檔案 | 內容 |
| --- | --- |
| `tasks.jsonl` | 這次實際跑的 Tasks，逐行照抄 jevbench 的 dataset。`run` 和 `summarize` 都讀這個檔，所以 `--limit` 的 run 一樣是 `complete: true`。 |
| `results.jsonl` | jevbench 的逐 Task 結果（`probs`、`valid`、`correct`、`latency_s`，以及 server 回應頂層的 `runtime`） |
| `raw/` | jevbench 的 raw evidence：每個 Task 一個檔，存完整的 request 與 response |
| `ledger.jsonl`、`manifest.json` | jevbench 的成本 ledger 與 run 設定（含 `dataset_hash`） |
| `summary.json` | jevbench `summarize` 的輸出：`accuracy`、`brier_mean`、`ece`、`schema_validity`（Valid answers）、`latency.p50_s`/`p95_s`（client 端）等 |
| `decisions.jsonl` | 每個 Task 一行：server 給的 Distribution 和預期的 Label 放在一起，給 calibration 用（格式見下） |
| `run.json` | 這次 run 的紀錄：jevbench 的 URL 與 commit、endpoint、Task set、`limit`、Task 數、起訖時間，以及 `server`：server 自己回報的 Decider、model id、revision、硬體（`cuda`、`mps` 或 `cpu`）與 `calibration`（raw 時是 `null`） |
| `calibration-report.json` | 只有跑過 `local-jev calibrate` 才有：cross-validated 的 calibration 報告（格式見「Calibration」） |

`run.json` 只在整個 run 成功跑完才會寫入。jevbench 中途停止（遇到 401、403、429，或連續三次失敗）時，`local-jev bench` 以非零 exit code 結束並說明跑到第幾個 Task；目錄裡留著 jevbench 已寫的結果與 raw evidence 供除錯，但沒有 `summary.json`、`decisions.jsonl`、`run.json`。

`decisions.jsonl` 每行的欄位（jevbench 的結果裡沒有預期的 Label，也沒有 Question type，所以這裡用 `task_id` 跟 Tasks 對起來）：

| 欄位 | 內容 |
| --- | --- |
| `task_id` | jevbench 的 Task id |
| `task_set` | 這個 Task 來自哪個部分：`easy`、`original` 或 `hard` |
| `question_type` | `choice`、`noul` 或 `score` |
| `labels` | 這個 Task 的 Label set（jevbench 作者的順序；`noul` 是 `["no", "yes"]`，`score` 是 `["0", …]`） |
| `expected` | 預期的 Label，一律是字串（`score` 也是，例如 `"2"`） |
| `distribution` | Server 給的 Distribution，以 Label 為 key（jevbench 的 `probs_as_returned`；`noul` 是 `{"yes": p, "no": 1 - p}`）。沒有可用的答案時是 `null` |
| `valid` | jevbench 是否接受這個 Distribution（無效的答案算答錯，calibration 應略過） |
| `predicted`、`correct` | jevbench 的 argmax Label，以及它是否等於 `expected` |
| `latency_s` | jevbench 量的 client 端 wall-clock（秒） |
| `runtime` | Server 這次回應的 `runtime`：Decider、model、revision、device、`calibration`，以及 server 端的 `total_ms`、`decider_ms` |

### jevbench 的眉角（`bench` 已處理）

- `typesafe` adapter 的 key 環境變數空著就一個請求都不送：沒設 token 時給它一個假的 `TYPESAFE_API_KEY`。
- 未知 endpoint 沒有價格時，jevbench 每個 Task 預留 $0.02、累計到 $15 上限就停，而且 cost 記成 null：一律傳 `--price-in-per-m 0 --price-out-per-m 0 --reserve-usd 0`。
- jevbench 的結果與 raw evidence 只能寫到新檔，而且必須在它的 checkout 之外：每次 run 都用新目錄、絕對路徑。
- jevbench 是用目前的 Python（`python -m jevbench.cli`）在它的 checkout 裡執行，不是 `uv run`（checkout 裡有它自己的 `pyproject.toml`）。它只用標準函式庫，不是 local-jev 的相依套件。
- jevbench 不會送 `explain`，所以 benchmark 一律不產生 Reason。

## Calibration

每個 Decider 對每種 Question type（`noul`、`choice`、`score`）各有一個 temperature T。Server 把 Decider 的 raw Distribution 的 log-probabilities 除以 T 再正規化，也就是 p_k 變成 p_k^(1/T) / Σ_j p_j^(1/T)：

- T > 1 讓過度自信的 Distribution 變平，T < 1 讓信心不足的變尖，T = 1 不變。
- Temperature scaling 不會改變各 Label 的排序，所以 argmax 與 Accuracy 不變，只有 Brier、ECE 會動。
- 沒有 calibration 時，Distribution 原樣送出。
- d1 自己內部已經先除過一次 T ≈ 1.337（它 revision 裡的 `calibration.json`）；我們的 T 疊在 d1 送出的機率之上。

### Calibration 檔案

每個 Decider 的 calibration 是一個 committed 的檔案 `src/local_jev/calibrations/<decider>.json`（`d1.json`、`lfm3.json`）。它在 package 裡面，會跟著安裝，所以不管從哪個目錄啟動 server、或在容器裡跑，都讀得到。

```json
{
  "decider": "d1",
  "model": "LiquidAI/d1-2B",
  "revision": "652f920e544ea37f81e4b0ce89b81d4ed0da90b1",
  "temperatures": {"choice": 1.0321, "noul": 2.1097, "score": 1.2695},
  "fitted_from": {"run": "20261105T085252Z-public", "task_set": "public", "n_calibration": 231}
}
```

- Server 啟動時只讀目前這個 Decider 的檔案，所以兩個 Decider 不會共用 calibration。
- 檔案裡的 `decider`、`model`、`revision` 必須和 server 實際載入的完全一致才會套用。例如用 `--model`/`--revision` 換了 model 時，就不套用、log 一個警告，照 raw 送出。
- `temperatures` 裡沒有的 Question type 不做 calibration（T = 1）。
- 檔案壞掉時（不是 JSON、T 不是正數、未知的 Question type），health 會回報 `failed` 並指出是哪個檔案，不會默默照 raw 送出。
- 每個回應的 `runtime.calibration` 都會寫出套用了哪個 calibration（raw 時是 `null`），所以 `run.json` 與 `local-jev bench` 印出的那一行也看得出這次 run 是 raw 還是 calibrated。
- `serve --no-calibration` 一律送 raw；`serve --calibrations-dir DIR` 改從別的目錄讀 calibration。

### 擬合 calibration（`local-jev calibrate`）

Temperature 只能用 raw Distribution 擬合，所以要從一個 `--no-calibration` 的 server 跑出來的 benchmark run 開始：

```sh
uv run local-jev serve --decider d1 --no-calibration   # 終端機 1：raw server
uv run local-jev bench                                 # 終端機 2：全部 231 個 public Tasks
uv run local-jev calibrate runs/20261105T085252Z-public
# 重新啟動 server（不加 --no-calibration）就會套用新的 calibration
```

`calibrate` 讀 run 裡的 `decisions.jsonl`，然後：

1. 對每種 Question type，在 T ∈ [0.05, 20] 裡找出讓預期 Label 的 negative log-likelihood 最小的 T（NLL 對 1/T 是 convex，用 golden-section search）。只用 jevbench 接受的答案；機率剛好是 0 的，算 likelihood 時當成 1e-12。
2. 做 5-fold cross-validation（依 Question type 分層，seed 0）：每個 Task 都用「另外 4 個 fold 擬合出的 T」來 calibrate，再對這些 held-out 的 Distribution 算 raw 與 calibrated 的 Brier、ECE。算法和 jevbench 的 `summarize` 完全一樣：
   - Brier 是 Label set 上的 multi-class sum Σ_k (p_k − y_k)²（`noul` 等於 2·(p_yes − y)²）；
   - ECE 是 top-label confidence、10 個等寬的 bin；
   - 兩者都把所有 Question type 合在一起算，另外也各別列出每種 Question type（這是我們自己加的）；
   - 無效的答案在 Accuracy 裡算錯，但不算進 Brier、ECE。

   所以 raw 的數字會等於 jevbench 對同一個 run 的 `summary.json`。
3. 把用全部 Tasks 擬合出的 T 寫進 `<calibrations-dir>/<decider>.json`（預設就是 committed 的 `src/local_jev/calibrations/`，會覆寫舊的），並把報告寫成 run 目錄裡的 `calibration-report.json`。

報告要引用的是 cross-validated 的數字，不是用同一批 Tasks 擬合再評估的 in-sample 數字（那會偏樂觀）。

`calibrate` 拒絕以下的 run，不寫任何檔案：

- server 有套用 calibration 的 run（`runtime.calibration` 不是 `null`）：請用 `--no-calibration` 重跑 benchmark；
- 由不只一個 Decider、model 或 revision 回答的 run；
- 沒有 `decisions.jsonl` 的目錄（不是跑完的 `local-jev bench` run）。

在 calibration 功能出現之前的 server，`runtime` 裡根本沒有 `calibration` 欄位，這種 run 當成 raw。

用真正的 d1（MPS、全部 231 個 public Tasks）跑出來的結果：

```text
[local-jev] d1 LiquidAI/d1-2B@652f920, run 20261105T085252Z-public, 231 Tasks
[local-jev] temperatures: choice 1.0321, noul 2.1097, score 1.2695
[local-jev] 5-fold cross-validated, raw -> calibrated:
[local-jev]   all     231 Tasks: Accuracy 76.6% -> 76.6%, Brier 0.285 -> 0.275, ECE 0.072 -> 0.060
[local-jev]   choice  139 Tasks: Accuracy 78.4% -> 78.4%, Brier 0.253 -> 0.258, ECE 0.056 -> 0.077
[local-jev]   noul     74 Tasks: Accuracy 73.0% -> 73.0%, Brier 0.357 -> 0.313, ECE 0.160 -> 0.070
[local-jev]   score    18 Tasks: Accuracy 77.8% -> 77.8%, Brier 0.239 -> 0.247, ECE 0.131 -> 0.109
```

d1 的 `noul` 明顯過度自信（T ≈ 2.1），`choice` 本來就差不多（T ≈ 1.03）。`score` 只有 18 個 Tasks，各別 Question type 的 cross-validated 數字雜訊很大，看合計那一行即可。

### `calibration-report.json`

| 欄位 | 內容 |
| --- | --- |
| `run`、`task_set` | Run 目錄名稱與 Task set |
| `decider`、`model`、`revision`、`device` | 回答這個 run 的 server（來自每個答案的 `runtime`） |
| `temperatures` | 用全部 Tasks 擬合、寫進 calibration 檔案的 T（每種 Question type 一個） |
| `calibration_file` | 寫入的 calibration 檔案路徑 |
| `cross_validation.folds`、`seed` | `5`、`0` |
| `cross_validation.fold_temperatures` | 每個 fold 用其他 fold 擬合出的 T（看 T 穩不穩定） |
| `cross_validation.raw`、`calibrated` | 合計：`n`（有預期 Label 的 Tasks）、`n_calibration`（其中有有效 Distribution 的）、`accuracy`、`brier`、`ece` |
| `cross_validation.by_question_type.<type>.raw`、`.calibrated` | 同上，只算這種 Question type |

## 比較報告（`local-jev report`）

`local-jev report` 把幾個跑完的 benchmark run 合併成一份 Markdown 報告：並列各 Decider、raw 與 calibrated、英文與繁體中文的數字，附上 jevbench 的 reference，逐條檢查議定的 feasibility verdict 標準，並選出預設的 Decider。

```sh
uv run local-jev report runs/<d1 的 raw run> runs/<lfm3 的 raw run> runs/<繁體中文的 run> …
```

| 選項 | 預設 | 說明 |
| --- | --- | --- |
| `--output FILE` | 無 | 寫到這個檔案（已存在就覆寫） |
| `--runs-dir` | `runs` | 沒給 `--output` 時，寫成這裡的新檔 `report-<UTC 時間>.md`，絕不覆寫之前的報告 |

不是跑完的 `local-jev bench` run（缺 `run.json`、`summary.json` 或 `decisions.jsonl`）、或同一個 run 混了英文與繁體中文的 Tasks，`report` 會直接拒絕並說明原因。

用真正的 d1（MPS、全部 231 個 public Tasks，跑過 `calibrate`）產生的報告，Results 與 verdict 的前幾行：

```text
| d1 LiquidAI/d1-2B@652f920 | raw | English | public | 231 | 76.6% | 0.285 | 0.072 | 100.0% | 175 / 2620 ms | 173 / 2617 ms | mps | 20261105T085252Z-public |
| d1 LiquidAI/d1-2B@652f920 | calibrated, 5-fold CV | English | public | 231 | 76.6% | 0.275 | 0.060 | 100.0% | 175 / 2620 ms | 173 / 2617 ms | mps | 20261105T085252Z-public |
| 3b. Server-side p50 ≤ 150 ms on an L4-class GPU | **pending** | no run of d1 LiquidAI/d1-2B@652f920 on cuda on the 231 public Tasks |
```

### Results：每個 run 一行

欄位：Decider（含 model 與 revision）、Calibration、Language、Task set、Tasks、Accuracy、Brier、ECE、Valid answers、client p50/p95、server p50/p95、Hardware、Run。

- **raw**：jevbench 自己對這個 run 算的 `summary.json`。
- **calibrated, 5-fold CV**：只有 run 裡有 `calibration-report.json`（跑過 `local-jev calibrate`）才有這一行，用的是 fitter 的 cross-validated 數字（每個 Task 都用其他 fold 擬合的 T calibrate），絕不用 in-sample 數字。Calibration 是事後套上去的，所以 Valid answers 與延遲沿用同一個 raw run。
- **calibrated, served**：server 本身套用 calibration 跑出來的 run。Accuracy、Valid answers、延遲照列，但 Brier 與 ECE 標成 `withheld`：如果 temperature 就是用這批 Tasks 擬合的，那些數字是 in-sample。要 calibrated 的 Brier、ECE，請用 `serve --no-calibration` 跑 raw run，再對它 `calibrate`。
- **Language**：看 `decisions.jsonl` 每一行的 `task_set`：以 `zh-tw-` 開頭（例如 `zh-tw-easy`）是繁體中文，否則是英文。
- **Client 延遲**是 jevbench 量的 wall-clock（`summary.json` 的 `latency`，包含失敗的請求）；**server 延遲**是每個答案 `runtime.total_ms` 的 p50/p95，percentile 的算法和 jevbench 一樣（線性內插）。
- **Hardware** 是答案 `runtime.device` 回報的 device。每一行只來自一個 run、一種硬體，不同硬體的 run 絕不合併成同一個延遲數字。萬一一個 run 的答案來自不只一種 device（例如中途換了 server），Hardware 寫成 `cpu + mps`，兩個延遲欄都寫 `mixed`。

想要繁體中文的 calibrated 行：對繁體中文 Task set 的 raw run 跑 `calibrate`。`calibrate` 一定會寫 calibration 檔案，所以要把 `--calibrations-dir` 指到別處，免得蓋掉 committed 的、用英文 public Tasks 擬合的 calibration：

```sh
uv run local-jev calibrate runs/<繁體中文的 raw run> --calibrations-dir /tmp/zh-tw-calibrations
```

### jevbench reference points

Jev 1.13.0 與 system-one-open（Gemma 4 E2B with LoRA, on an L4），每一行都標明 suite 與出處（jevbench 固定 commit `609a664` 裡的檔案與欄位）：

| System | Suite | Accuracy | Brier | ECE | 出處 |
| --- | --- | --- | --- | --- | --- |
| Jev 1.13.0 | 231 個 public Tasks | 86.6% | 未公布 | 未公布 | `results/v1.2/jevbench-v1.2-per-task.json` |
| system-one-open | 231 個 public Tasks | 73.2% | 未公布 | 未公布 | 同上 |
| Jev 1.13.0 | 舊的 v1 suite（242 decisions） | 96.3% | 0.056 | 0.027 | `results/jevbench-v1-results.json`（`RESULTS.md` 第 32 行） |
| system-one-open | 舊的 v1 suite（242 decisions） | 90.1% | 0.138 | 0.068 | `results/jevbench-v1-results.json`（`RESULTS.md` 第 36 行） |

- spec 列的 reference（Jev 96.3%、Brier 0.056；system-one-open 90.1%）是舊的 v1 suite，不是 verdict 用的 231 個 public Tasks，報告裡明確標成 older suite。
- 在 public Tasks 上，jevbench 只公布每個 Task 的對錯與延遲，沒有 Distribution，所以 Brier、ECE、Valid answers 都沒有公布。Accuracy ≥ 80% 這條標準落在 system-one-open（73.2%）與 Jev（86.6%）之間。
- reference 的延遲是 jevbench 維護者從自己的 client 量到遠端 endpoint 的 raw wall-clock（沒有乘上他們對 self-hosted endpoint 的 ×2），不能拿來和本機 server 端的時間比。
- `tests/test_comparison_report.py` 會拿這些數字去比對 jevbench 固定 commit 裡的原始檔案。

### Feasibility verdict

只看英文、完整 231 個 public Tasks 的 run（`--task-set public`、沒有 `--limit`）。每條標準是 pass、fail 或 pending：

| 標準 | pass | fail | pending |
| --- | --- | --- | --- |
| 1. 每個 Decider 的 Valid answers 都是 100% | `d1`、`lfm3` 都有 run，而且全部 100% | 任一個 run 低於 100% | 還有 Decider 沒有 run |
| 2. 最好的 Decider 達到 Accuracy ≥ 80% 且 Brier ≤ 0.20 | 有 Decider 的 raw 或 cross-validated 數字同時達標 | 沒有 Decider 能達標 | 還有 Decider 沒有 run，或 Accuracy 已達標但還沒 calibrate |
| 3a. 在 Mac（MPS）上記錄每個 Decider 的 server 端 p50 | 每個 Decider 都有 MPS 的 run | （沒有門檻） | 還有 Decider 沒有 MPS 的 run |
| 3b. L4 等級 GPU 上 server 端 p50 ≤ 150 ms | 預設 Decider 的每個 CUDA run 都達標 | 有 CUDA run 超過 150 ms | 沒有預設 Decider 的 CUDA run |

- 標準 2 只用 raw 或 cross-validated 的 Brier，calibrated, served 的 Brier 不算。Temperature scaling 不會改變 argmax，所以 Accuracy 不到 80% 的 Decider calibrate 也沒用，直接算 fail。
- 標準 3b：run 只記錄 `cuda`，看不出是不是 L4 等級的 GPU，evidence 會註明沒有檢查。目前沒有 GPU 主機，所以它一直是 pending。
- **Overall**：全部 pass 才是 pass（feasible, worth further investment）；任一條 fail 就是 fail；其餘是 pending，並列出是哪幾條。
- 繁體中文的 run 另外列在 Results，不影響 verdict。
- **Default Decider**：在 231 個 public Tasks 上依序比較：答案全部有效、達到標準 2、Accuracy 較高、Brier 較低。還有 Decider 沒跑時標成 provisional。

## 測試

```sh
uv run pytest                 # 快速測試：in-process 打 HTTP contract，用假的 d1 model 與假的 LM backend，不載入權重
uv run pytest -m real_model   # 真模型測試：載入真正的 LFM3-1.7B 與 d1-2B（第一次會下載），按需執行
uv run pytest -m real_model tests/test_real_lfm3.py   # 只跑其中一個 model
uv run ruff check
```

快速測試裡的 `tests/test_bench.py`、`tests/test_comparison_report.py` 等會跑真正的 jevbench：第一次執行時把它抓進 `.cache/`（需要網路），之後離線也能跑。`real_model` 裡的 `tests/test_real_bench.py` 用 jevbench 的 `typesafe` adapter 對真正的 d1 server 跑 10 個 Tasks，10 個答案都必須有效。
