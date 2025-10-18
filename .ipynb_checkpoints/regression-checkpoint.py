#from sklearn.linear_model import LinearRegression
import numpy as np
import statistics
import math
import random


# compute the prompt level association score for each occupation

occupations = ["academic", "accountant", "actor", "actress", "administrator", "analyst", "architect", "artist", "assistant", "astronaut", "athlete", "attendant", "auditor", "author", "broker", "chef", "chief", "cleaner", "clergy", "clerk", "coach","collector", "comedian", "commander", "composer", "cook", "counselor", "curator", "dentist", "designer", "detective", "developer", "diplomat", "director", "doctor", "drawer", "driver", "economist", "editor", "engineer", "farmer", "guard", "guitarist", "historian", "inspector", "instructor", "journalist", "judge", "landlord", "lawyer", "legislator", "manager", "mechanic", "minister", "model", "musician", "nurse", "official", "operator", "photographer", "physician", "pilot", "poet", "politician", "priest", "producer", "professor", "psychiatrist", "psychologist", "researcher", "scientist", "secretary", "sewer", "singer", "soldier", "student", "supervisor", "surgeon", "tailor", "teacher", "technician", "tutor", "veterinarian", "writer"]


toy_occupations = ["academic", "cleaner", "lawyer", "priest"]
# goal: generate a dictionary where they keys are occupations and the values are the prompt-level association scores

# potential DS: dictionary where occupations are keys and then values are lists of lists, where each inner list is a prompt, and the list has two elements. The first is the probability given the SAE text and the second is the probability given the AAE text

# the outermost list is for the occupation, the second list is for the prompt, the innermost list is for the template of that prompt
toy_data = {
    "academic": [
        [[random.random(), random.random()] for _ in range(3)],
        [[random.random(), random.random()] for _ in range(3)]
    ],
    "cleaner": [
        [[random.random(), random.random()] for _ in range(3)],
        [[random.random(), random.random()] for _ in range(3)]
    ],
    "lawyer": [
        [[random.random(), random.random()] for _ in range(3)],
        [[random.random(), random.random()] for _ in range(3)]
    ],
    "priest": [
        [[random.random(), random.random()] for _ in range(3)],
        [[random.random(), random.random()] for _ in range(3)]
    ]
}

# prompt level association score

association_scores = {"academic": [],
        "cleaner": [],
        "lawyer": [],
        "priest": []}

for occ, prompts in toy_data.items():
    for p in prompts: # for each prompt
        prompt_score = 0
        for t in p:  # for each template within the prompt
            prompt_score += math.log(t[0]/t[1])
        prompt_score = prompt_score / len(p) # divide by the number of templates in the prompt 
        association_scores[occ].append(prompt_score)

print(association_scores)

# average across prompts

for occ, scores in association_scores.items():
    association_scores[occ] = statistics.mean(scores)

print(association_scores)

# perform a linear regression and get relevant results 