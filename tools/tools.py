from io import BytesIO
import string
import re
import base64
import json
from PIL import Image
import random
import numpy as np
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def im_2_b64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def im_resize(image, size):
    """Resize the shortest edge to `size`, keeping the aspect ratio unchanged."""
    if size == 0:
        return image
    w, h = image.size
    if w < h:
        w_new = size
        h_new = int(h * w_new / w)
    else:
        h_new = size
        w_new = int(w * h_new / h)
    return image.resize((w_new, h_new), Image.LANCZOS)


def remove_list_format(s):
    # s = s.replace(".\n", ". ").replace("\n", ". ").replace("*", "").replace("- ", "")
    s = s.strip()
    s = re.sub("\.+\s*\n\d\.\s*", ". ", s)
    s = re.sub("\.+\s*\n\-\s*", ". ", s)
    s = re.sub("^\-\s*", "", s)
    s = re.sub("^\d\.\s*", "", s)
    s = re.sub("\n\d\.\s*", "", s)
    s = re.sub("\*", "", s)
    s = s.replace("The most significant object is a", "A")
    return s.strip()


def none_pipeline(*args):
    if isinstance(args[0], list):
        return [None for _ in args[0]]
    else:
        return None


def get_str_from_brackets(s):
    m = re.search(r"<(.*?)>", s)
    if m:
        res = m.group(1)
        return res
    else:
        return "None"


str_action_map = {
    0: "stop",
    1: "move forward",
    2: "turn left",
    3: "turn right",
}


def read_gt_action(s, smooth=True, action_file="rgb/actions.json"):
    with open(s / action_file, "r") as f:
        actions = json.load(f)
    if smooth:
        i = 0
        while i < len(actions) - 2:
            if (
                actions[i] == 1
                and actions[i + 2] == 1
                and (actions[i + 1] == 2 or actions[i + 1] == 3)
            ):
                actions[i + 1] = 1
                i = i + 2
            else:
                i += 1

    actions = [str_action_map[v] for v in actions]
    return actions[:-1]  # remove stop


# move_forward_phrases = [
#     "walk",
#     "walk forward",
#     "go",
#     "go to",
#     "go forward",
#     "continue",
#     "continue straight",
#     "head",
#     "head forward",
#     "head towards",
#     "move",
#     "move forward",
#     "move ahead",
#     "move straight",
#     "forward",
#     "keep going straight",
#     "keep moving forward",
#     "lead",
#     "follow",
#     "proceed",
#     "proceed straight",
#     "proceed onward",
#     "travel",
#     "ascend",
#     "remain",
#     "approach",
#     "run",
#     "advance",
#     "advance ahead",
#     "press forward",
#     "march forward",
#     "progress",
#     "progress straight",
#     "move straight ahead",
#     "push ahead",
#     "forge ahead",
#     "move onward",
#     "maintain direction",
#     "step forward",
#     "climb",
#     "line straight",
# ]
# turn_left_phrases = [
#     "turn",
#     "turn left",
#     "make a turn",
#     "take a left turn",
#     "veer",
#     "curve",
#     "curve left",
#     "left",
#     "go left",
#     "go in the left direction",
#     "rotate",
#     "take a left",
#     "twist",
#     "turn to your left",
#     "head left",
#     "shift left",
#     "move leftward",
#     "head in the left direction",
#     "veer left",
#     "shift to the left",
#     "navigate left",
#     "bear left",
#     "angle left",
#     "steer left",
#     "bear",
#     "angle",
# ]
# turn_right_phrases = [
#     "turn",
#     "turn right",
#     "make a turn",
#     "take a right turn",
#     "veer",
#     "curve",
#     "curve right",
#     "right",
#     "go right",
#     "go in the right direction",
#     "rotate",
#     "take a right",
#     "twist",
#     "turn to your right",
#     "head right",
#     "shift right",
#     "move rightward",
#     "head in the right direction",
#     "veer right",
#     "shift to the right",
#     "navigate right",
#     "bear right",
#     "angle right",
#     "steer right",
#     "bear",
#     "angle",
# ]
# stop_phrases = [
#     "stop",
#     "stop moving",
#     "wait",
#     "stop immediately",
#     "stop right there",
#     "stop now",
#     "reach",
#     "come to a stop",
#     "sit",
#     "stay",
#     "locate",
#     "find",
#     "arrive",
#     "meet",
#     "hold on",
#     "hold position",
#     "halt",
#     "halt action",
#     "end",
#     "put an end to",
#     "pause",
#     "finish",
#     "quit",
#     "cease",
#     "cease activity",
#     "freeze",
#     "freeze in place",
#     "terminate",
#     "suspend",
#     "stand still",
#     "stand firm",
#     "bring to a halt",
#     "face",
# ]
# enter_phrases = [
#     "walk into",
#     "walk in",
#     "go into",
#     "go in",
#     "go to",
#     "enter",
#     "reach",
#     "walk near",
#     "walk towards",
#     "arrive at",
#     "walk inside",
#     "move to",
#     "move into",
#     "move in",
#     "advance to",
#     "step into",
#     "head into",
#     "move towards",
#     "go inside",
#     "proceed to",
# ]
# leave_phrases = [
#     "walk out",
#     "walk outside",
#     "go out",
#     "go outside",
#     "leave",
#     "exit",
#     "move out",
#     "move outside",
#     "walk off",
#     "get off",
#     "move off",
#     "get out",
#     "go away",
#     "walk away",
#     "step out",
#     "head out",
#     "depart",
#     "depart from",
#     "step away",
#     "leave the room",
#     "withdraw",
#     "move away",
#     "run away",
#     "break away",
#     "away from",
# ]
# pass_phrases = [
#     "walk through",
#     "walk past",
#     "walk by",
#     "walk along",
#     "go through",
#     "go past",
#     "go by",
#     "go across",
#     "walk around",
#     "pass through",
#     "pass",
#     "pass by",
#     "move through",
#     "move past",
#     "move between",
#     "cross",
#     "cross over",
#     "course",
#     "traverse",
#     "via",
#     "navigate through",
#     "proceed past",
#     "continue past",
# ]
# diversed_action_map = {
#     "move forward": move_forward_phrases,
#     "turn left": turn_left_phrases,
#     "turn right": turn_right_phrases,
#     "stop": stop_phrases,
#     "enter": enter_phrases,
#     "pass": pass_phrases,
#     "leave": leave_phrases,
# }


# move_forward_phrases = [
#     "move forward",
#     "go",
#     "move",
#     "continue",
#     "head",
#     "advance",
#     "proceed",
#     "forward",
#     "push",
#     "drive",
#     "go forward",
#     "move ahead",
#     "proceed straight",
#     "continue straight",
#     "head forward",
#     "walk forward",
#     "advance ahead",
#     "keep going straight",
#     "press forward",
#     "march forward",
#     "progress",
#     "progress straight",
#     "stay on course",
#     "move straight ahead",
#     "keep moving forward",
#     "proceed onward",
#     "push ahead",
#     "forge ahead",
#     "carry on",
#     "move onward",
#     "maintain direction",
#     "step forward",
# ]
# turn_left_phrases = [
#     "turn left",
#     "left",
#     "turn",
#     "go left",
#     "take a left",
#     "make a left turn",
#     "head left",
#     "shift left",
#     "veer",
#     "steer",
#     "pivot",
#     "swing",
#     "bear",
#     "curve",
#     "angle",
#     "move leftward",
#     "head in the left direction",
#     "turn to your left",
#     "veer left",
#     "shift to the left",
#     "navigate left",
#     "bear left",
#     "angle left",
#     "pivot left",
#     "steer left",
#     "swing left",
#     "curve left",
#     "turn leftward",
#     "take a left turn",
#     "go in the left direction",
# ]
# turn_right_phrases = [
#     "turn right",
#     "right",
#     "turn",
#     "go right",
#     "take a right",
#     "make a right turn",
#     "head right",
#     "shift right",
#     "veer",
#     "steer",
#     "pivot",
#     "swing",
#     "bear",
#     "curve",
#     "angle",
#     "move rightward",
#     "head in the right direction",
#     "turn to your right",
#     "veer right",
#     "shift to the right",
#     "navigate right",
#     "bear right",
#     "angle right",
#     "pivot right",
#     "steer right",
#     "swing right",
#     "curve right",
#     "turn rightward",
#     "take a right turn",
#     "go in the right direction",
# ]
# stop_phrases = [
#     "stop",
#     "end",
#     "hold",
#     "pause",
#     "finish",
#     "quit",
#     "halt",
#     "cease",
#     "freeze",
#     "wait",
#     "terminate",
#     "discontinue",
#     "suspend",
#     "break",
#     "conclude",
#     "stand still",
#     "stop moving",
#     "refrain",
#     "come to a stop",
#     "desist",
#     "stop immediately",
#     "stop right there",
#     "put an end to",
#     "bring to a halt",
#     "hold position",
#     "cease activity",
#     "freeze in place",
#     "stand firm",
#     "halt action",
#     "stop now",
# ]
# enter_phrases = [
#     "enter",
#     "go in",
#     "walk in",
#     "reach",
#     "approach",
#     "arrive at",
#     "go into",
#     "move into",
#     "walk into",
#     "step into",
#     "head into",
#     "advance to",
#     "move towards",
#     "go inside",
#     "proceed to",
# ]
# leave_phrases = [
#     "leave",
#     "get out",
#     "go away",
#     "exit",
#     "walk away",
#     "walk out",
#     "step out",
#     "head out",
#     "go out",
#     "move out",
#     "depart",
#     "step away",
#     "leave the room",
#     "withdraw",
#     "move away",
# ]
# pass_phrases = [
#     "pass",
#     "walk through",
#     "go past",
#     "move through",
#     "pass by",
#     "walk past",
#     "go by",
#     "move past",
#     "cross",
#     "traverse",
#     "walk by",
#     "pass through",
#     "proceed past",
#     "continue past",
#     "navigate through",
# ]
# diversed_action_map = {
#     "move forward": move_forward_phrases,
#     "turn left": turn_left_phrases,
#     "turn right": turn_right_phrases,
#     "stop": stop_phrases,
#     "enter": enter_phrases,
#     "pass": pass_phrases,
#     "leave": leave_phrases,
# }

diversed_action_map_1 = {
    "move forward": [
        "move forward",
        "go",
        "move",
        "continue",
        "head",
        "advance",
        "proceed",
        "forward",
        "push",
        "drive",
        "go forward",
        "move ahead",
        "proceed straight",
        "continue straight",
        "head forward",
        "walk forward",
        "advance ahead",
        "keep going straight",
        "press forward",
        "march forward",
        "progress",
        "progress straight",
        "stay on course",
        "move straight ahead",
        "keep moving forward",
        "proceed onward",
        "push ahead",
        "forge ahead",
        "carry on",
        "move onward",
        "maintain direction",
        "step forward",
    ],
    "turn left": [
        "turn left",
        "left",
        "turn",
        "go left",
        "take a left",
        "make a left turn",
        "head left",
        "shift left",
        "veer",
        "steer",
        "pivot",
        "swing",
        "bear",
        "curve",
        "angle",
        "move leftward",
        "head in the left direction",
        "turn to your left",
        "veer left",
        "shift to the left",
        "navigate left",
        "bear left",
        "angle left",
        "pivot left",
        "steer left",
        "swing left",
        "curve left",
        "turn leftward",
        "take a left turn",
        "go in the left direction",
    ],
    "turn right": [
        "turn right",
        "right",
        "turn",
        "go right",
        "take a right",
        "make a right turn",
        "head right",
        "shift right",
        "veer",
        "steer",
        "pivot",
        "swing",
        "bear",
        "curve",
        "angle",
        "move rightward",
        "head in the right direction",
        "turn to your right",
        "veer right",
        "shift to the right",
        "navigate right",
        "bear right",
        "angle right",
        "pivot right",
        "steer right",
        "swing right",
        "curve right",
        "turn rightward",
        "take a right turn",
        "go in the right direction",
    ],
    "stop": [
        "stop",
        "end",
        "hold",
        "pause",
        "finish",
        "quit",
        "halt",
        "cease",
        "freeze",
        "wait",
        "terminate",
        "discontinue",
        "suspend",
        "break",
        "conclude",
        "stand still",
        "stop moving",
        "refrain",
        "come to a stop",
        "desist",
        "stop immediately",
        "stop right there",
        "put an end to",
        "bring to a halt",
        "hold position",
        "cease activity",
        "freeze in place",
        "stand firm",
        "halt action",
        "stop now",
    ],
    "enter": [
        "enter",
        "go in",
        "walk in",
        "reach",
        "approach",
        "arrive at",
        "go into",
        "move into",
        "walk into",
        "step into",
        "head into",
        "advance to",
        "move towards",
        "go inside",
        "proceed to",
    ],
    "leave": [
        "leave",
        "get out",
        "go away",
        "exit",
        "walk away",
        "walk out",
        "step out",
        "head out",
        "go out",
        "move out",
        "depart",
        "step away",
        "leave the room",
        "withdraw",
        "move away",
    ],
    "pass": [
        "pass",
        "walk through",
        "go past",
        "move through",
        "pass by",
        "walk past",
        "go by",
        "move past",
        "cross",
        "traverse",
        "walk by",
        "pass through",
        "proceed past",
        "continue past",
        "navigate through",
    ],
}

# GPT
diversed_action_map_2 = {
    "move forward": [
        "go forward",
        "move forward",
        "keep going",
        "keep moving",
        "walk ahead",
        "continue forward",
        "head straight",
        "proceed forward",
        "move ahead",
        "walk straight",
        "go straight",
        "keep walking",
        "carry on",
        "step forward",
        "continue straight",
        "proceed",
    ],
    "turn left": [
        "turn left",
        "go left",
        "take a left",
        "make a left turn",
        "head left",
        "move left",
        "veer left",
        "steer left",
        "pivot left",
        "swing left",
        "curve left",
        "angle left",
        "shift left",
        "walk left",
        "step left",
    ],
    "turn right": [
        "turn right",
        "go right",
        "take a right",
        "make a right turn",
        "head right",
        "move right",
        "veer right",
        "steer right",
        "pivot right",
        "swing right",
        "curve right",
        "angle right",
        "shift right",
        "walk right",
        "step right",
    ],
    "stop": [
        "stop",
        "pause",
        "halt",
        "freeze",
        "wait",
        "stand still",
        "stop moving",
        "stop now",
        "stop immediately",
        "hold still",
        "come to a stop",
        "stay there",
        "stand by",
        "stay still",
        "hold position",
    ],
    "enter": [
        "enter",
        "go in",
        "walk in",
        "step in",
        "go inside",
        "walk inside",
        "move in",
        "head inside",
    ],
    "pass": [
        "pass by",
        "go past",
        "walk past",
        "move past",
        "walk through",
        "go through",
        "cross",
        "keep going",
    ],
    "leave": [
        "leave",
        "go out",
        "get out",
        "walk out",
        "step out",
        "head out",
        "move out",
        "exit",
    ],
}

# Qwen
diversed_action_map_3 = {
    "move forward": [
        "move forward",
        "go",
        "continue",
        "head forward",
        "proceed",
        "go straight",
        "keep going",
        "move ahead",
        "go straight ahead",
        "continue straight",
        "walk forward",
        "keep going straight",
        "progress forward",
        "move straight ahead",
        "keep moving forward",
        "proceed forward",
    ],
    "turn left": [
        "turn left",
        "left",
        "go left",
        "take a left",
        "make a left turn",
        "head left",
        "turn to the left",
        "veer left",
        "bear left",
        "angle left",
        "pivot left",
        "steer left",
        "swing left",
        "curve left",
        "navigate left",
    ],
    "turn right": [
        "turn right",
        "right",
        "go right",
        "take a right",
        "make a right turn",
        "head right",
        "turn to the right",
        "veer right",
        "bear right",
        "angle right",
        "pivot right",
        "steer right",
        "swing right",
        "curve right",
        "navigate right",
    ],
    "stop": [
        "stop",
        "end",
        "pause",
        "finish",
        "halt",
        "freeze",
        "wait",
        "terminate",
        "come to a stop",
        "stop moving",
        "stop immediately",
        "put an end to",
        "bring to a halt",
        "stop now",
    ],
    "enter": [
        "enter",
        "go in",
        "walk in",
        "reach",
        "arrive at",
        "step into",
        "go inside",
        "approach",
    ],
    "pass": [
        "pass",
        "go past",
        "move past",
        "cross",
        "pass by",
        "walk by",
        "pass through",
        "go through",
    ],
    "leave": [
        "leave",
        "exit",
        "get out",
        "go away",
        "depart",
        "move away",
        "walk away",
    ],
}

diversed_action_map = diversed_action_map_2


def diversify_action(a, limit=1.0, diversify_level=0):
    if (not diversify_level) or (a not in diversed_action_map):
        return a
    elif diversify_level == 1:  # Double uniform
        N = len(diversed_action_map[a])
        random_range = min(N, max(1, int(round(limit * N + 0.5))))
        return random.choice(diversed_action_map[a][:random_range])
    elif diversify_level == 2:  # Uniform
        return random.choice(diversed_action_map[a])
    elif diversify_level == 3:  # Exponential decay
        N = len(diversed_action_map[a])
        random_weights = np.exp(-np.arange(N) / N * 10 * limit)
        return random.choices(diversed_action_map[a], k=1, weights=random_weights)[0]


def get_synonyms(word, tag):
    # self.wordnet_tag_map = {
    #     'n': 'NN',
    #     's': 'JJ',
    #     'a': 'JJ',
    #     'r': 'RB',
    #     'v': 'VB'
    # }
    syns = wordnet.synsets(word)
    if not syns:
        return [word]
    word_type = ""
    word_type2 = ""
    if tag[1].startswith("NN"):
        word_type = wordnet.NOUN
    elif tag[1].startswith("VB"):
        word_type = wordnet.VERB
    elif tag[1].startswith("JJ"):
        word_type = wordnet.ADJ
        word_type2 = wordnet.ADJ_SAT
    elif tag[1].startswith("RB"):
        word_type = wordnet.ADV

    replacements = set()
    for syn in syns:
        for lemma in syn.lemmas():
            replacements.add(lemma.name())
    syn = syns[0]
    if (
        syn.name().find("." + word_type + ".") >= 0
        or syn.name().find("." + word_type2 + ".") >= 0
    ):
        # extract the word only
        # print(syn)
        # print(syn.hypernyms())
        # print(syn.hyponyms())
        for hyper in syn.hypernyms():
            r = hyper.name()[0 : hyper.name().find(".")]
            if r not in replacements:
                replacements.add(r)
        for hypo in syn.hyponyms():
            r = hypo.name()[0 : hypo.name().find(".")]
            if r not in replacements:
                replacements.add(r)
    replacements = [r.replace("_", " ") for r in replacements]
    return replacements


def diversify_element(a, limit=1.0, diversify_level=0):
    if not diversify_level:
        return a
    words = word_tokenize(a)
    tags = nltk.pos_tag(words)
    indecies = [
        i
        for i, tag in enumerate(tags)
        if (tag[1] != "NNP" and tag[1] != "DT" and tag[1] not in string.punctuation)
    ]
    if not indecies:
        return a
    idx = random.choice(indecies)
    replacements = get_synonyms(words[idx], tags[idx])
    if diversify_level == 1:  # Double uniform
        N = len(replacements)
        random_range = min(N, max(1, int(round(limit * N + 0.5))))
        new_word = random.choice(replacements[:random_range])
    elif diversify_level == 2:  # Uniform
        new_word = random.choice(replacements)
    elif diversify_level == 3:  # Exponential decay
        N = len(replacements)
        random_weights = np.exp(-np.arange(N) / N * 10 * limit)
        new_word = random.choices(replacements, weights=random_weights, k=1)[0]
    print(words[idx], new_word)
    words[idx] = new_word
    return TreebankWordDetokenizer().detokenize(words)


if __name__ == "__main__":
    a = "living room"
    s = diversify_element(a, diversify_level=2)
    print(get_synonyms("door", ("", "NN")))
    print(a)
    print(s)
    print(diversify_action("turn right", limit=0.5, diversify_level=3))
