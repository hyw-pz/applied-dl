"""
vocabulary.py
-------------
ARPAbet phoneme vocabulary shared across all modules.
"""

LOGIT_TO_PHONEME = [
    'BLANK',
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B',  'CH', 'D',  'DH',
    'EH', 'ER', 'EY', 'F',  'G',  'HH', 'IH', 'IY', 'JH', 'K',
    'L',  'M',  'N',  'NG', 'OW', 'OY', 'P',  'R',  'S',  'SH',
    'T',  'TH', 'UH', 'UW', 'V',  'W',  'Y',  'Z',  'ZH', '|',
]

NUM_PHONEMES = len(LOGIT_TO_PHONEME)  # 41

PHONEME_TO_LOGIT = {p: i for i, p in enumerate(LOGIT_TO_PHONEME)}

VOCAB_PHONEMES = list(range(1, 40))  # excludes BLANK (0) and '|' (40)

PHONEME_CONFUSION_MAP = {
    # Plosives/Stops
    27: [7, 31, 20],   # P -> B, T, K
    7:  [27, 9, 15],   # B -> P, D, G
    31: [9, 27, 20],   # T -> D, P, K
    9:  [31, 7, 15],   # D -> T, B, G
    20: [15, 31, 27],  # K -> G, T, P
    15: [20, 9, 7],    # G -> K, D, B
    # Fricatives
    14: [35, 32, 29],  # F  -> V, TH, S
    35: [14, 10, 38],  # V  -> F, DH, Z
    32: [10, 14, 29],  # TH -> DH, F, S
    10: [32, 35, 38],  # DH -> TH, V, Z
    29: [38, 30, 32],  # S  -> Z, SH, TH
    38: [29, 39, 10],  # Z  -> S, ZH, DH
    30: [39, 29, 8],   # SH -> ZH, S, CH
    39: [30, 38, 19],  # ZH -> SH, Z, JH
    16: [32, 14],      # HH -> TH, F
    # Affricates
    8:  [19, 30, 31],  # CH -> JH, SH, T
    19: [8, 39, 9],    # JH -> CH, ZH, D
    # Nasals
    22: [23, 24],      # M  -> N, NG
    23: [22, 24],      # N  -> M, NG
    24: [23, 22],      # NG -> N, M
    # Liquids and Glides
    21: [28, 36, 37],  # L  -> R, W, Y
    28: [21, 36],      # R  -> L, W
    36: [21, 28, 34],  # W  -> L, R, UW
    37: [21, 18],      # Y  -> L, IY
    # Front Vowels
    18: [17, 13],      # IY -> IH, EY
    17: [18, 11],      # IH -> IY, EH
    11: [17, 2],       # EH -> IH, AE
    2:  [11, 1],       # AE -> EH, AA
    # Back / Central Vowels
    1:  [4, 3],        # AA -> AO, AH
    4:  [1, 33],       # AO -> AA, UH
    33: [34, 4],       # UH -> UW, AO
    34: [33, 25],      # UW -> UH, OW
    3:  [11, 33, 12],  # AH -> EH, UH, ER
    12: [3, 28],       # ER -> AH, R
    # Diphthongs
    13: [11, 18, 6],   # EY -> EH, IY, AY
    6:  [1, 18, 13],   # AY -> AA, IY, EY
    26: [4, 18],       # OY -> AO, IY
    5:  [1, 34, 25],   # AW -> AA, UW, OW
    25: [4, 34, 5],    # OW -> AO, UW, AW
}
