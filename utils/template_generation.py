import random

# Base skeletons
BASE_TEMPLATES = [
    "{verb} from {s} to {g}, avoiding {obstacles} at altitude {altitude}.",
    "{verb} from {s} to {g}, maintaining altitude {altitude}.",
    "Mission: {verb} the drone from {s} to {g} at altitude {altitude}, then {land}.",
    "Objective: {verb} from {s} to {g} and {land}, flying at altitude {altitude}.",
    "Task: {verb} between {s} and {g}, stay {safety}, altitude {altitude}.",
    "{verb} the UAV from {s} to {g} at altitude {altitude}.",
    "Start at {s}, {verb} to {g} at altitude {altitude}, then {land}.",
    "Head from {s} to {g} at altitude {altitude}, replan if {blocked}.",
    "Survey the area from {s} until reaching {g}, altitude {altitude}.",
    "Scout path from {s} to {g}, keep altitude {altitude}."
]

# Synonym pools
SYNONYMS = {
    "verb": [
        "Fly", "Navigate", "Go", "Move", "Travel", "Head", "Pilot", "Dispatch",
        "Send", "Operate", "Control", "Direct"
    ],
    "obstacles": [
        "obstacles", "hazards", "barriers", "no-fly zones", "restricted regions", "danger areas"
    ],
    "altitude": [
        "altitude 1.0 meters", "low altitude", "a steady altitude", "constant height",
        "safe altitude", "1 meter height"
    ],
    "land": [
        "land", "descend", "touch down", "finish by landing", "return safely"
    ],
    "safety": [
        "safe", "cautious", "clear of hazards", "within safe zones", "away from danger"
    ],
    "blocked": [
        "blocked", "obstructed", "occupied", "unsafe", "not clear"
    ]
}

def expand_templates(n=100):
    results = []
    for _ in range(n):
        template = random.choice(BASE_TEMPLATES)
        filled = template
        for key, values in SYNONYMS.items():
            if f"{{{key}}}" in template:
                filled = filled.replace(f"{{{key}}}", random.choice(values))
        # Add random altitude and abstract map info
        altitude = round(random.uniform(1.0, 5.0), 2)
        filled = filled.replace("{altitude}", str(altitude))
        # Example abstract map info
        filled += " Abstract map: obstacles and no-fly zones described."
        results.append(filled)
    return results

if __name__ == "__main__":
    examples = expand_templates(1000)  # generate 100 variations
    print("TEMPLATES = [")
    for ex in examples:
        print(f'    "{ex}",')
    print("]")