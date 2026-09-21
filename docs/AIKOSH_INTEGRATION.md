# AIKosh Integration

VerifiedGen-India uses AIKosh as a grounded external-validation source, beginning with agriculture.

## Research use

The AIKosh source is used to:

1. identify an India-specific domain with real reference data
2. ground agriculture prompt requirements
3. support external validation of object and context checks
4. compare generic evaluators with domain-aware verification

The current source registry is:

- `data/image_verification/aikosh_sources.json`

The code representation is:

- `src/image_verification/aikosh.py`

## Current source

The first source is the AIKosh listing for the ICAR Crop Disease and Insect-pest Image Dataset for Rice and Maize:

<https://aikosh.indiaai.gov.in/home/datasets/details/crop_disease_and_insect_pest_image_dataset_for_rice_and_maize.html>

## Responsible workflow

Before downloading or redistributing any data:

- record the dataset access level
- record the license and attribution requirements
- check whether images can be used for research
- avoid personal or sensitive imagery
- keep source identifiers in every derived prompt and manifest row

## Planned experiment

```text
AIKosh agriculture reference concepts
  -> grounded prompts
  -> SDXL/FLUX candidates
  -> generic CLIP/VLM scores
  -> agriculture-aware checks
  -> human labels
  -> compare false-accept and false-reject rates
```

The central hypothesis is that generic image evaluators may miss agriculture-specific or India-specific context failures that a grounded verifier can detect.
