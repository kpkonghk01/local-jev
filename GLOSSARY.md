# Local-Jev

A locally hosted small-model decision system that answers jevbench-style typed Questions with Native probabilities, built to be measured by jevbench as a Jev-compatible system.

## Language

### Decisions

**Decision**:
One answer to one Question about one State, expressed as a Distribution over that Question's Label set.
_Avoid_: classification, prediction, verdict

**Decider**:
A way of turning a State and a Question into a Decision: either Option-ID scoring over a chat model, or a Decision model answering natively.
_Avoid_: strategy, engine, backend

**State**:
The text (or structured data) describing the situation a Decision is about.
_Avoid_: input, document, user_input

**Question**:
A typed ask about a State, made of a Question type, instructions and Criteria.
_Avoid_: prompt, query

**Question type**:
One of `noul` (yes/no), `choice` (one of several named options) or `score` (one level on an ordered scale).
_Avoid_: category, task type

**Label set**:
The exact answers a Question allows; it differs from Question to Question.
_Avoid_: categories, classes, enum

**Criteria**:
The description of what each Label means (an ordered list of level descriptions for `score`).
_Avoid_: rubric, definitions

**Distribution**:
A probability for every Label in the Label set, summing to 1.
_Avoid_: probabilities map, confidence, scores

**Native probabilities**:
A Distribution read from the model's own likelihood of each Label.
_Avoid_: logprob confidence, real probabilities

**Verbalized probabilities**:
A Distribution the model writes out as numbers in its output text.
_Avoid_: self-reported confidence

**Option-ID scoring**:
Producing Native probabilities from the model's likelihood of the short identifier that stands in for each Label (a letter for `choice` options; the Labels themselves for `noul` and `score`).
_Avoid_: letter scoring, MCQ scoring, label scoring

**Reason**:
An optional short explanation of a Decision, written in the State's language after the Distribution is already fixed.
_Avoid_: rationale, justification

### Models

**Decision model**:
A model trained to answer Questions directly with a Distribution in one pass, without generating text.
_Avoid_: classifier, decision LLM

**Chat model**:
A general instruction-following model that answers by generating text; it becomes a Decider only through Option-ID scoring.
_Avoid_: instruct model, LLM

**d1**:
Liquid AI's family of open Decision models.
_Avoid_: LFM3 d1, LFM3-D1

### Benchmark

**Jev**:
TypeSafe AI's commercial decision model; a product name, not an acronym.
_Avoid_: JEV, JSON Evaluation

**Jev-compatible system**:
Any decision system jevbench can evaluate in place of Jev.
_Avoid_: JEV API, JEV-compliant API

**jevbench**:
The public benchmark that scores Jev-compatible systems on accuracy, calibration, speed and cost.
_Avoid_: jev-benchmark, JEV benchmark

**Task**:
One jevbench record: a State, a Question, its Label set and the expected Label.
_Avoid_: item, sample, example
