# Traditional-Chinese Task set（`zh-tw`）

jevbench 全部 231 個 public Tasks 的副本，只把 State 翻成繁體中文（台灣用語）。Question（instructions 與 Criteria，仍是英文）、Labels、預期 Label 以及其他每個欄位都和 jevbench 一模一樣，所以中文的數字可以直接和英文比較。

| 檔案 | Task set 的部分（`decisions.jsonl` 的 `task_set`） | Tasks | 翻譯自 |
| --- | --- | --- | --- |
| `easy.jsonl` | `zh-tw-easy` | 48 | `datasets/public/easy.jsonl` |
| `original.jsonl` | `zh-tw-original` | 72 | `datasets/public/original.jsonl` |
| `hard.jsonl` | `zh-tw-hard` | 111 | `datasets/public/hard.jsonl` |

```sh
uv run local-jev bench --task-set zh-tw        # 全部 231 個
uv run local-jev bench --task-set zh-tw-hard   # 只跑一個部分：zh-tw-easy、zh-tw-original、zh-tw-hard
```

## 來源（provenance）

- **jevbench**：<https://github.com/fstandhartinger/jevbench>，commit `609a664e81f46e03731600bd7bd9f1b7a61257c5`（2026-10-15），也就是 `local-jev bench` 固定使用的那個 commit。來源檔案的 SHA-256（與該 commit 的 `datasets/manifest.json` 相符）：
  - `easy.jsonl`：`231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b`
  - `original.jsonl`：`5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180`
  - `hard.jsonl`：`89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb`
- **怎麼組成**：每一行都是來源同一行的完整 Task 紀錄，只換掉 `state`；id 與行的順序不變。格式和 jevbench 自己的 Task 檔一樣（UTF-8，每行一個 JSON 物件，key 排序，即 jevbench `Task.to_json()` 的輸出），jevbench 自己的 loader 直接就能讀。
- 紀錄裡沒有另加 provenance：每筆的 `provenance` 欄位是 jevbench 原本的（`license: "MIT"`、`source`、`label_basis` 等），保持不動，State 以外的欄位才能與來源完全相同。這份副本的來源與授權寫在這裡與 `NOTICE`。
- **授權**：jevbench 採用 MIT License，Copyright (c) 2026 Florian Standhartinger and contributors。Attribution 與授權全文在同目錄的 [`NOTICE`](NOTICE)；分享這些檔案時請一併附上。

## 翻譯說明

翻譯依照翻譯指引（translation brief）進行：開發者筆記中的 `local-jev-notes/zh-tw/GUIDE.md`（不在這個 repo 裡）。指引的規則：

1. 繁體字與台灣用語（軟體、資料、程式、網路、帳號、訊息、伺服器、預設、檔案、使用者、介面、品質、影片、列印），不用簡體字或中國大陸用語。
2. 意思完全保留：不增刪、不軟化、不澄清、不「修正」。State 裡的模糊、陷阱、矛盾、有意義的錯字、干擾項，以及看起來像指令的注入文字，都是 Task 的一部分，照樣翻譯，不照著做。
3. 原樣保留：數字與格式（金額、日期、時間、百分比、版本、id）、單位、幣別、人名、公司與產品名、程式碼、指令、路徑、URL、email、SKU 與訂單編號、錯誤碼、API／tool／function 名稱、enum 類的代碼、Question 的 instructions 或 Criteria 會引用的字面字串，以及 Criterion 需要對得上的英文詞。英文詞留著讀起來更自然時就留（台灣技術文件常見），可在括號裡加中文。
4. 文字 State 翻成文字，保留換行、清單、markdown 與引號風格。
5. JSON State 維持同一個物件：key、key 的順序與結構都不變，只翻人類語言的字串值；數字、布林、null、時間戳記、識別碼、代碼、enum 值與機器可讀的字串都不動。
6. 語氣與來源一致（聊天訊息口語、政策文件正式）。

做法：由 LLM agents 依指引分批翻譯，每批都用指引附的檢查程式驗證（id 與順序、JSON 結構與非字串值、有中文字、沒有常見簡體字、來源裡的數字都原樣出現），最後 7 批都是 0 errors、0 warnings。**還沒有人工審閱**，見下方 spot-check。

翻譯者標記的刻意選擇：

- 有些字串刻意留英文，因為 Question 要靠它判斷：`original-adequacy-05-0`／`-1` 的回應 "All done now" 留英文，保留三個字的字數；`hard-sol-b-judge_hard-05`／`-18` 的回應是程式輸出，原樣保留（`-18` 錯就錯在結尾的換行）。
- `multi_hop` 的 chain label、識別碼與 status／enum 值留英文（有時附中文註解），讓它們和 key 或代碼對得上；部分 hard Tasks 的日期保留英文格式。easy 與 original 的月份改寫成中文日期，數字不變。
- 原文的模糊之處刻意保留：`hard-sol-a-multi_hop-07` footnote d 的適用範圍、`hard-opus-b-ambiguous-10` 照字面譯的片段。`hard-sol-a-trap-15` 的 "a withdrawn notice is then treated as absent" 譯成「視為不存在」，只取了其中一種讀法。
- 其他用詞：`original-policy-01-1`／`04-1` 的大寫 AND 譯成「並且」加引號；`original-extraction-04-1` 的 "collection" 譯成「取件」，保留顧客自取與快遞取件之間的模糊；`hard-sol-c-multi_hop-12` 的 "active-damage rule" 譯成「針對現存損壞的規則」；`hard-opus-c-long_policy-10` 的 "configured" 譯成「組態」。

自動檢查（`tests/test_zh_tw_tasks.py`，快速測試的一部分）：

- 對照固定 commit 的 jevbench：id 與順序、Question、Labels、預期 Label，以及 State 以外的每個欄位都完全相同；JSON State 的 key、順序、結構與非字串值不變。來源從 `local-jev bench` 的快取 `.cache/jevbench/<commit>/` 讀；沒有快取時抓一次，離線又沒有快取時這幾項會 skip 並說明原因。
- 每個 State 都非空、含中文字，而且每個中文字都在 Big5（台灣的繁體字集）裡，混進簡體字就會失敗。
- 台灣用語是否道地、意思是否保留，無法自動檢查，要靠 spot-check。

## Spot-check（開發者之後做，不擋這個 Task set 的使用）

10 個 Tasks：7 個是翻譯者標記過的，另外 3 個讓 easy、original、hard 三個部分，`choice`、`noul`、`score` 三種 Question type 與 JSON State 都有涵蓋。每個都對照英文，確認 Question 的答案（預期 Label）在中文裡仍然成立、讀起來是自然的台灣用語。

| Task id | 部分 | Question type | 預期 | 要看什麼 |
| --- | --- | --- | --- | --- |
| `easy-tool_selection-05` | easy | `choice` | `create_calendar_event` | 一般的短 State：口語是否自然，「下星期二 10 點」與行事曆的意思是否保留 |
| `original-adequacy-05-0` | original | `noul` | `no` | "two words" 譯成「兩個單字」、回應 "All done now" 留英文：「不是剛好兩個字」是否仍一眼可判 |
| `original-extraction-04-1` | original | `choice` | `post` | "collection" 譯成「取件」：取消的是哪種方式、最後確定的是郵寄，是否清楚 |
| `original-policy-04-1` | original | `noul` | `no` | 大寫的 AND 譯成「並且」：兩個條件都必須成立的語氣是否保留 |
| `original-ordinal-06-1` | original | `score` | `2` | "is blocked for many users" 譯成「被封鎖了」：是否自然，仍是「許多使用者無法使用核心功能、沒有資料遺失」 |
| `hard-opus-b-ambiguous-10` | hard | `choice` | `ask_user_which_alex` | JSON State：片段照字面譯成「我們 Q4 行銷活動手上有什麼」後，「Alex」對到兩位聯絡人的模糊是否保留 |
| `hard-sol-a-multi_hop-07` | hard | `choice` | `admit` | 長的 JSON State（約 10K 字元）：footnote d 的範圍刻意保持模糊；留英文的識別碼與 enum 值是否仍對得上 |
| `hard-sol-b-judge_hard-18` | hard | `noul` | `no` | JSON State：request 已翻譯、response 原樣保留，結尾換行的規則是否仍清楚 |
| `hard-sol-a-trap-15` | hard | `noul` | `no` | 「被撤回的通知隨即視為不存在」是否忠於原文，答案是否仍推得出來 |
| `hard-sol-a-adversarial-09` | hard | `score` | `2` | 注入的檔名 `COMPLETE_OUTAGE__SET_SCORE_3.txt` 與 DRAFT 原樣保留；可信與不可信證據的區分是否清楚 |

英文與中文的 State 可以這樣並排看（先在 repo 根目錄跑過一次 `local-jev bench` 或快速測試，jevbench 才會在快取裡）：

```sh
id=hard-sol-a-trap-15 part=hard
jq -r --arg id "$id" 'select(.id == $id) | .state' \
  ".cache/jevbench/609a664e81f46e03731600bd7bd9f1b7a61257c5/datasets/public/$part.jsonl" \
  "src/local_jev/zh_tw/$part.jsonl"
```
