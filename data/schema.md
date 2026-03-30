# V2X-QA JSONL Schema

Each line in the released JSONL files is a single JSON object.

## Core fields

### `sample_id`
- Type: `string`
- Description: Unique sample identifier for the QA instance.

### `task_id`
- Type: `string`
- Allowed values:
  - `VS1`, `VS2`, `VS3`, `VS4`
  - `IS1`, `IS2`, `IS3`, `IS4`
  - `CO1`, `CO2`, `CO3`, `CO4`
- Description: Viewpoint-aligned task identifier.

### `question_id`
- Type: `string`
- Description: Identifier of the question template within the task bank.

### `question`
- Type: `string`
- Description: The multiple-choice question text.

### `options`
- Type: `object`
- Required keys: `A`, `B`, `C`, `D`
- Value type: `string`
- Description: Candidate answer options.

### `selected_option_id`
- Type: `string`
- Allowed values: `A`, `B`, `C`, `D`
- Description: Correct option letter.
- Note: This field is used by the provided training and evaluation scripts as the gold label.

## View-specific image identifier fields

### `image_id_vs`
- Type: `string` or `null`
- Description: Vehicle-side image identifier corresponding to the original V2X-Seq-SPD raw image filename stem.
- Expected for: `VS*` and `CO*` tasks.

### `image_id_is`
- Type: `string` or `null`
- Description: Infrastructure-side image identifier corresponding to the original V2X-Seq-SPD raw image filename stem.
- Expected for: `IS*` and `CO*` tasks.

## Optional fields
Depending on the exact released file, additional fields may appear.

### `canonical_answer`
- Type: `string`
- Description: Canonical natural-language rendering of the correct answer.
- Recommendation: For public test files, this field is usually unnecessary if `selected_option_id` is already present.

## Example
```json
{
  "sample_id": "000096",
  "task_id": "CO1",
  "question_id": "CO1_Q1",
  "question": "With both views, what best describes the ego path?",
  "options": {
    "A": "The path looks clear.",
    "B": "Cross traffic affects the path.",
    "C": "The path is constrained.",
    "D": "The path is still unclear."
  },
  "selected_option_id": "B",
  "image_id_vs": "000096",
  "image_id_is": "000087"
}
```

## Validation recommendations
Before training or evaluation, verify that:
- every line is valid JSON
- `selected_option_id` is one of `A/B/C/D`
- `selected_option_id` exists in `options`
- `task_id` matches one of the defined task names
- the corresponding raw image IDs can be resolved under `data/raw_external/`
