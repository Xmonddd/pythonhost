from typing import List, Tuple

RED_FLAG_RULES = [
    {
        "id": "cardiac",
        "any": ["chest pain", "shortness of breath", "sweating"],
        "message": "Possible cardiac concern. Seek urgent medical evaluation.",
        "severity": "high"
    },
    {
        "id": "meningitis_like",
        "all": ["stiff neck", "fever", "headache"],
        "message": "Combination may indicate serious infection. Get medical attention.",
        "severity": "high"
    },
    {
        "id": "respiratory",
        "all": ["wheezing", "shortness of breath"],
        "message": "Breathing difficulty. Monitor closely; seek care if worsening.",
        "severity": "medium"
    }
]

def evaluate_red_flags(symptoms: List[str]) -> Tuple[List[str], str | None]:
    triggered = []
    final_severity = None
    for rule in RED_FLAG_RULES:
        any_ok = any(s in symptoms for s in rule.get("any", [])) if rule.get("any") else False
        all_ok = all(s in symptoms for s in rule.get("all", [])) if rule.get("all") else False
        if any_ok or all_ok:
            triggered.append(rule["message"])
            sev = rule["severity"]
            if sev == "high":
                final_severity = "high"
            elif sev == "medium" and final_severity != "high":
                final_severity = "medium"
    return triggered, final_severity