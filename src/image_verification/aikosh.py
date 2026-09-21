"""AIKosh provenance and grounded-domain helpers.

This module does not download AIKosh data automatically. Access, licensing,
and redistribution must be checked for each dataset before use.
"""
from typing import Dict, List

AIKOSH_ICAR_RICE_MAIZE = {
    "source_id": "aikosh_icar_rice_maize_crop_disease",
    "title": "ICAR Crop Disease and Insect-pest Image Dataset for Rice and Maize",
    "url": "https://aikosh.indiaai.gov.in/home/datasets/details/crop_disease_and_insect_pest_image_dataset_for_rice_and_maize.html",
    "domain": "agriculture",
    "reference_role": "prompt_grounding_and_external_validation",
    "license_status": "verify_on_download",
}

def agriculture_source_metadata() -> Dict:
    return {"aikosh_reference": dict(AIKOSH_ICAR_RICE_MAIZE)}

def source_registry() -> List[Dict]:
    return [dict(AIKOSH_ICAR_RICE_MAIZE)]

