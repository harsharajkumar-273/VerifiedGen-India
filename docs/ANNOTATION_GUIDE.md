# Annotation guide

You will see an image and the prompt it was generated from. Judge the image **as the person who typed the prompt would**. You will not be told which model made the image or why it was chosen.

## 1. Verdict (required)

**Would you show this image to the person who typed this prompt?**

- `accept`: yes, it does what was asked well enough to use.
- `reject`: no, because of a problem they would care about.

Decide this first, from your overall impression. The questions below record *why*.

## 2. Alignment (required, 1–5)

How well does the image match the prompt?

| Score | Meaning |
|---|---|
| 5 | Everything requested is present and correct |
| 4 | Small issues a typical user would not mind |
| 3 | Recognisably the right scene, with a clear mistake |
| 2 | Major parts missing or wrong |
| 1 | Unrelated to the prompt, or unusable |

## 3. Text rendering (required when the prompt asks for text)

Only annotate text in scripts you read fluently. If you cannot read the requested script, flag the task and skip it.

- `correct`: every requested word is spelled correctly in the requested script.
- `minor_error`: readable, with a small error such as a wrong vowel sign or a missing mark.
- `wrong`: misspelled, the wrong word, the wrong script, or unreadable glyph-like shapes.
- `no_text`: no text appears at all.

Also type the **transcription**: exactly what the image says, as you read it, including errors. Leave it empty if nothing is readable.

## 4. Cultural accuracy (required when the prompt names a place, festival or custom)

- `accurate`: the setting, clothing, objects and architecture fit what was asked.
- `inaccurate`: something clearly wrong, such as the wrong festival, the wrong region, or objects from another culture.
- `unsure`: you do not know the context well enough to judge.

Judge accuracy, not taste. A plain but correct image is `accurate`.

## 5. Failure types (tick all that apply)

| Tag | Use when |
|---|---|
| `garbled_text` | Text-like shapes that are not real letters |
| `wrong_text` | Real letters, wrong word or spelling |
| `wrong_script` | Text in a different script from the one requested |
| `missing_text` | Requested text absent |
| `wrong_locale` | The setting looks like a different country or region |
| `cultural_error` | A specific cultural detail is wrong |
| `stereotype` | Demeaning or clichéd portrayal of people or places |
| `missing_object` | A requested object is absent |
| `wrong_count` | The wrong number of a requested object |
| `visual_artifact` | Distorted faces or hands, melted objects, visible seams |
| `low_quality` | Blurry, washed out, or very low detail |
| `unsafe` | Content that should never be shown |

## Examples to calibrate on in the pilot

- *"a Delhi tea stall sign that says चाय"*. The sign reads "चाच" → `reject`, text `wrong`, `wrong_text`.
- *"a London tea stall sign that says TEA"*. The sign reads "TEE" → `reject`, text `wrong`, `wrong_text`. Apply the same strictness to both scripts.
- *"a Diwali scene with diyas arranged outside an Indian home"*. Candles on a Christmas-style porch → `reject`, cultural `inaccurate`, `cultural_error`.
- *"a small dairy farm in Gujarat with two cows and a farmer"*. Three cows, otherwise good → usually `accept`, alignment 4, `wrong_count`.

Apply the same standard to every image, whatever the country in the prompt.
