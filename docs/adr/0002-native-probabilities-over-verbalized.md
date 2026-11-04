# Native probabilities via Option-ID scoring, no grammar-constrained decoding

The original proposal had the model write its answer as JSON under an xgrammar schema. We instead present a Question's Labels as options with short identifiers and read the model's next-token likelihood of each identifier, normalised over the Label set: the Distribution sums to 1 by construction, one forward pass yields the whole answer, and jevbench's own results note that only systems which write their probabilities out ever fail the arithmetic. Because the model never writes the JSON, grammar-constrained decoding has nothing left to do, so xgrammar is not a dependency; an optional Reason is generated as plain text and escaped by the server when it builds the response.

## Considered Options

- **Verbalized probabilities under an xgrammar JSON schema** (the proposal): dropped; it needs dozens of generated tokens per Decision and its numbers are the model's guess, not its likelihood.
- **Label scoring** (likelihood of each Label's full name): dropped for simplicity and to keep exactly one forward pass per Decision.
