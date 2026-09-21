import argparse, json, random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List
from .prompt_parser import extract_prompt_requirements
from .aikosh import agriculture_source_metadata

@dataclass
class PromptSpec:
    prompt_id: str
    prompt: str
    category: str
    expected_objects: Dict[str, int] = field(default_factory=dict)
    difficulty: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)
    @classmethod
    def from_dict(cls, data):
        data = dict(data); data["metadata"] = dict(data.get("metadata") or {}); data["expected_objects"] = dict(data.get("expected_objects") or extract_prompt_requirements(data["prompt"]).objects); return cls(**data)

INDIA = [
 ("india_multilingual_text", "a street food stall sign in Delhi that says चाय", {"expected_text":["चाय"],"language":"Hindi","script":"Devanagari","cultural_markers":["street food stall","Delhi","chai"]}),
 ("india_multilingual_text", "a Chennai bus stop sign that says பேருந்து நிறுத்தம்", {"expected_text":["பேருந்து நிறுத்தம்"],"language":"Tamil","script":"Tamil","cultural_markers":["bus stop","Chennai"]}),
 ("india_multilingual_text", "a Kolkata bookshop sign that says বই", {"expected_text":["বই"],"language":"Bengali","script":"Bengali","cultural_markers":["bookshop","Kolkata"]}),
 ("india_multilingual_text", "a Hyderabad metro sign that says మెట్రో", {"expected_text":["మెట్రో"],"language":"Telugu","script":"Telugu","cultural_markers":["metro sign","Hyderabad"]}),
 ("india_multilingual_text", "a bilingual Indian road sign that says SCHOOL and स्कूल", {"expected_text":["SCHOOL","स्कूल"],"language":"English+Hindi","script":"Latin+Devanagari","cultural_markers":["road sign","India"]}),
 ("india_public_service", "an Indian public health poster about vaccination at a rural clinic", {"expected_objects":{"poster":1,"clinic":1},"cultural_markers":["rural clinic","public health poster","India"]}),
 ("india_public_service", "a clean illustration of an Aadhaar enrollment desk in a government office", {"expected_objects":{"desk":1,"person":1},"cultural_markers":["Aadhaar","government office","India"]}),
 ("india_public_service", "a public-service poster encouraging water conservation in Rajasthan", {"expected_objects":{"poster":1},"cultural_markers":["Rajasthan","water conservation","public-service poster"]}),
 ("india_public_service", "an Indian railway platform information board with passengers waiting", {"expected_objects":{"person":2,"train":1},"cultural_markers":["Indian railway","platform","information board"]}),
 ("india_public_service", "a Swachh Bharat cleanliness poster on a city wall", {"expected_objects":{"poster":1},"cultural_markers":["Swachh Bharat","cleanliness poster","city wall"]}),
 ("india_agriculture", "an Indian farmer inspecting rice paddy fields during monsoon", {"expected_objects":{"person":1},"cultural_markers":["farmer","rice paddy","monsoon","India"]}),
 ("india_agriculture", "a cotton field in Maharashtra with a farmer holding a mobile phone", {"expected_objects":{"person":1,"phone":1},"cultural_markers":["cotton field","Maharashtra","farmer"]}),
 ("india_agriculture", "a small dairy farm in Gujarat with two cows and a farmer", {"expected_objects":{"cow":2,"person":1},"cultural_markers":["dairy farm","Gujarat","cow"]}),
 ("india_agriculture", "a millet field in rural Karnataka under a clear sky", {"expected_objects":{"field":1},"cultural_markers":["millet","Karnataka","rural India"]}),
 ("india_agriculture", "a vegetable mandi in India with crates of tomatoes and onions", {"expected_objects":{"tomato":1,"onion":1},"cultural_markers":["mandi","tomatoes","onions","India"]}),
 ("india_cultural_context", "a Diwali scene with diyas arranged outside an Indian home", {"expected_objects":{"diya":2},"cultural_markers":["Diwali","diyas","Indian home"]}),
 ("india_cultural_context", "a Pongal celebration with a decorated clay pot and sugarcane", {"expected_objects":{"pot":1,"sugarcane":1},"cultural_markers":["Pongal","clay pot","sugarcane"]}),
 ("india_cultural_context", "a Durga Puja pandal decorated with flowers in Kolkata", {"expected_objects":{"pandal":1},"cultural_markers":["Durga Puja","Kolkata","flowers"]}),
 ("india_cultural_context", "an Onam pookalam flower arrangement outside a Kerala home", {"expected_objects":{"flower arrangement":1},"cultural_markers":["Onam","pookalam","Kerala"]}),
 ("india_cultural_context", "a traditional Indian wedding entrance decorated with marigold garlands", {"expected_objects":{"garland":1},"cultural_markers":["Indian wedding","marigold","entrance"]}),
 ("india_everyday_scene", "a small kirana store with shelves of snacks and a shopkeeper", {"expected_objects":{"store":1,"person":1},"cultural_markers":["kirana store","snacks","shopkeeper"]}),
 ("india_everyday_scene", "an auto rickshaw waiting beside a busy street in Mumbai", {"expected_objects":{"auto rickshaw":1},"cultural_markers":["auto rickshaw","Mumbai","street"]}),
 ("india_everyday_scene", "a chai stall beside an Indian railway station in the morning", {"expected_objects":{"stall":1},"cultural_markers":["chai stall","railway station","India"]}),
 ("india_everyday_scene", "a family shopping at a colorful Indian street market", {"expected_objects":{"person":2,"market":1},"cultural_markers":["street market","India","family"]}),
 ("india_everyday_scene", "a school bus arriving at a neighborhood road in Bengaluru", {"expected_objects":{"bus":1},"cultural_markers":["school bus","Bengaluru","neighborhood road"]}),
]

INDIC_SCRIPT_PROMPTS = [
 ("Hindi", "Devanagari", "Delhi", "चाय", "a Delhi tea stall sign that says चाय"),
 ("Hindi", "Devanagari", "Jaipur", "विद्यालय", "a Jaipur school sign that says विद्यालय"),
 ("Hindi", "Devanagari", "Lucknow", "बाजार", "a Lucknow market sign that says बाजार"),
 ("Hindi", "Devanagari", "Bhopal", "अस्पताल", "a Bhopal clinic sign that says अस्पताल"),
 ("Hindi", "Devanagari", "Patna", "स्वागत", "a Patna station sign that says स्वागत"),
 ("Tamil", "Tamil", "Chennai", "தேநீர்", "a Chennai tea shop sign that says தேநீர்"),
 ("Tamil", "Tamil", "Madurai", "பள்ளி", "a Madurai school sign that says பள்ளி"),
 ("Tamil", "Tamil", "Coimbatore", "மருத்துவமனை", "a Coimbatore clinic sign that says மருத்துவமனை"),
 ("Tamil", "Tamil", "Salem", "சந்தை", "a Salem market sign that says சந்தை"),
 ("Tamil", "Tamil", "Tiruchirappalli", "வரவேற்பு", "a Tiruchirappalli station sign that says வரவேற்பு"),
 ("Telugu", "Telugu", "Hyderabad", "టీ", "a Hyderabad tea stall sign that says టీ"),
 ("Telugu", "Telugu", "Vijayawada", "పాఠశాల", "a Vijayawada school sign that says పాఠశాల"),
 ("Telugu", "Telugu", "Warangal", "ఆసుపత్రి", "a Warangal clinic sign that says ఆసుపత్రి"),
 ("Telugu", "Telugu", "Visakhapatnam", "మార్కెట్", "a Visakhapatnam market sign that says మార్కెట్"),
 ("Telugu", "Telugu", "Tirupati", "స్వాగతం", "a Tirupati station sign that says స్వాగతం"),
 ("Bengali", "Bengali", "Kolkata", "চা", "a Kolkata tea shop sign that says চা"),
 ("Bengali", "Bengali", "Siliguri", "বিদ্যালয়", "a Siliguri school sign that says বিদ্যালয়"),
 ("Bengali", "Bengali", "Howrah", "হাসপাতাল", "a Howrah clinic sign that says হাসপাতাল"),
 ("Bengali", "Bengali", "Durgapur", "বাজার", "a Durgapur market sign that says বাজার"),
 ("Bengali", "Bengali", "Kolkata", "স্বাগতম", "a Kolkata station sign that says স্বাগতম"),
 ("Kannada", "Kannada", "Bengaluru", "ಚಹಾ", "a Bengaluru tea stall sign that says ಚಹಾ"),
 ("Kannada", "Kannada", "Mysuru", "ಶಾಲೆ", "a Mysuru school sign that says ಶಾಲೆ"),
 ("Kannada", "Kannada", "Hubballi", "ಆಸ್ಪತ್ರೆ", "a Hubballi clinic sign that says ಆಸ್ಪತ್ರೆ"),
 ("Kannada", "Kannada", "Mangaluru", "ಮಾರುಕಟ್ಟೆ", "a Mangaluru market sign that says ಮಾರುಕಟ್ಟೆ"),
 ("Kannada", "Kannada", "Shivamogga", "ಸ್ವಾಗತ", "a Shivamogga station sign that says ಸ್ವಾಗತ"),
 ("Malayalam", "Malayalam", "Kochi", "ചായ", "a Kochi tea shop sign that says ചായ"),
 ("Malayalam", "Malayalam", "Thiruvananthapuram", "സ്കൂൾ", "a Thiruvananthapuram school sign that says സ്കൂൾ"),
 ("Malayalam", "Malayalam", "Kozhikode", "ആശുപത്രി", "a Kozhikode clinic sign that says ആശുപത്രി"),
 ("Malayalam", "Malayalam", "Thrissur", "ചന്ത", "a Thrissur market sign that says ചന്ത"),
 ("Malayalam", "Malayalam", "Alappuzha", "സ്വാഗതം", "an Alappuzha station sign that says സ്വാഗതം"),
 ("Marathi", "Devanagari", "Pune", "चहा", "a Pune tea stall sign that says चहा"),
 ("Marathi", "Devanagari", "Nagpur", "शाळा", "a Nagpur school sign that says शाळा"),
 ("Marathi", "Devanagari", "Nashik", "रुग्णालय", "a Nashik clinic sign that says रुग्णालय"),
 ("Marathi", "Devanagari", "Kolhapur", "बाजार", "a Kolhapur market sign that says बाजार"),
 ("Marathi", "Devanagari", "Aurangabad", "स्वागत", "an Aurangabad station sign that says स्वागत"),
 ("Gujarati", "Gujarati", "Ahmedabad", "ચા", "an Ahmedabad tea stall sign that says ચા"),
 ("Gujarati", "Gujarati", "Surat", "શાળા", "a Surat school sign that says શાળા"),
 ("Gujarati", "Gujarati", "Vadodara", "હોસ્પિટલ", "a Vadodara clinic sign that says હોસ્પિટલ"),
 ("Gujarati", "Gujarati", "Rajkot", "બજાર", "a Rajkot market sign that says બજાર"),
 ("Gujarati", "Gujarati", "Bhavnagar", "સ્વાગત", "a Bhavnagar station sign that says સ્વાગત"),
 ("Punjabi", "Gurmukhi", "Amritsar", "ਚਾਹ", "an Amritsar tea stall sign that says ਚਾਹ"),
 ("Punjabi", "Gurmukhi", "Ludhiana", "ਸਕੂਲ", "a Ludhiana school sign that says ਸਕੂਲ"),
 ("Punjabi", "Gurmukhi", "Patiala", "ਹਸਪਤਾਲ", "a Patiala clinic sign that says ਹਸਪਤਾਲ"),
 ("Punjabi", "Gurmukhi", "Jalandhar", "ਬਾਜ਼ਾਰ", "a Jalandhar market sign that says ਬਾਜ਼ਾਰ"),
 ("Punjabi", "Gurmukhi", "Bathinda", "ਸਵਾਗਤ", "a Bathinda station sign that says ਸਵਾਗਤ"),
]

CODE_MIXED_PROMPTS = [
 ("English+Hindi", "Latin+Devanagari", "a bilingual Delhi road sign that says BUS STOP and बस स्टॉप"),
 ("English+Tamil", "Latin+Tamil", "a bilingual Chennai sign that says HOSPITAL and மருத்துவமனை"),
 ("English+Telugu", "Latin+Telugu", "a bilingual Hyderabad sign that says MARKET and మార్కెట్"),
 ("English+Bengali", "Latin+Bengali", "a bilingual Kolkata sign that says SCHOOL and বিদ্যালয়"),
 ("English+Kannada", "Latin+Kannada", "a bilingual Bengaluru sign that says METRO and ಮೆಟ್ರೋ"),
 ("English+Malayalam", "Latin+Malayalam", "a bilingual Kochi sign that says TEA and ചായ"),
 ("English+Marathi", "Latin+Devanagari", "a bilingual Pune sign that says WELCOME and स्वागत"),
 ("English+Gujarati", "Latin+Gujarati", "a bilingual Ahmedabad sign that says MARKET and બજાર"),
 ("English+Punjabi", "Latin+Gurmukhi", "a bilingual Amritsar sign that says HOTEL and ਹੋਟਲ"),
 ("English+Hindi", "Latin+Devanagari", "an Indian railway sign that says PLATFORM and प्लेटफॉर्म"),
]

def _expanded_india(limit):
    rows = []
    for language, script, region, text, prompt in INDIC_SCRIPT_PROMPTS:
        rows.append(("india_script_rendering", prompt, {"expected_text":[text], "language":language, "script":script, "region":region, "cultural_markers":[region, "Indian signage"], "track":"ocr_by_script"}))
    for language, script, prompt in CODE_MIXED_PROMPTS:
        texts = [part for part in prompt.split(" says ", 1)[-1].split(" and ")]
        rows.append(("india_code_mixed", prompt, {"expected_text":texts, "language":language, "script":script, "cultural_markers":["Indian bilingual signage"], "track":"code_mixed_ocr"}))
    # Keep the application benchmark balanced: 10 agriculture, governance,
    # and cultural prompts. Variants are explicit prompt instances, not extra
    # images, so they can be independently annotated.
    category_bases = {
        "india_agriculture": INDIA[10:15],
        "india_public_service": INDIA[5:10],
        "india_cultural_context": INDIA[15:20],
    }
    for category, bases in category_bases.items():
        for index, base in enumerate(bases * 2):
            cat, prompt, meta = base
            suffix = " with a documentary reference composition" if index >= len(bases) else ""
            rows.append((cat, prompt + suffix, {**meta, "track": category, "variant": index + 1}))
    while len(rows) < limit:
        base = rows[len(rows) % 5]
        rows.append((base[0], base[1] + " in a documentary-style reference scene", {**base[2], "augmentation":"context_variant"}))
    return rows[:limit]

def build_prompt_bank(limit=25, preset="india", seed=0, categories=None):
    random.seed(seed); rows=[]
    items = _expanded_india(limit) if preset in {"india", "sarvam", "aikosh_india"} else [("general", x, {}) for x in ["a red car on a road", "three apples on a plate", "a scientist holding a microscope"]]
    for i, (cat, prompt, meta) in enumerate(items[:limit], 1):
        if categories and cat not in categories: continue
        if cat == "india_agriculture":
            meta = {**meta, **agriculture_source_metadata()}
        rows.append(PromptSpec(f"india_v1_{cat}_{i:03d}" if preset == "india" else f"prompt_{i:03d}", prompt, cat, (meta.get("expected_objects") or {}), "hard" if preset == "india" else "medium", meta))
    return rows

def save_prompt_bank(prompts, path):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_text("\n".join(json.dumps(asdict(p), ensure_ascii=False) for p in prompts)+"\n", encoding="utf-8")

def load_prompt_bank(path): return [PromptSpec.from_dict(json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--preset",default="india"); p.add_argument("--limit",type=int,default=25); p.add_argument("--output",default="data/image_verification/india_prompt_bank_v1.jsonl"); args=p.parse_args(); save_prompt_bank(build_prompt_bank(args.limit,args.preset),args.output); print(f"[prompt-bank] wrote {args.limit} prompts -> {args.output}")
if __name__ == "__main__": main()
