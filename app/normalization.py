from typing import List

SYMPTOM_SYNONYMS = {
    'lagnat': 'fever',
    'ubo': 'cough',
    'sipon': 'runny nose',
    'sakit ng ulo': 'headache',
    'hirap huminga': 'shortness of breath',
    'pananakit ng dibdib': 'chest pain'
}

def normalize_symptom(raw: str) -> str:
    base = raw.strip().lower()
    base = base.replace('  ', ' ')
    if base in SYMPTOM_SYNONYMS:
        return SYMPTOM_SYNONYMS[base]
    return base

def normalize_list(list_in: List[str]) -> List[str]:
    out = []
    for s in list_in:
        n = normalize_symptom(s)
        if n and n not in out:
            out.append(n)
    return out