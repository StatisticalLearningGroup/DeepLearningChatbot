"""
Utilities for Chatbot 
"""

import unicodedata
import re
   
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def clean_resp(raw_resp, tokens):
    resp = [w for w in raw_resp if not w in tokens]
    return " ".join(resp)