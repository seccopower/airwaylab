"""Stable anatomical identifiers, separate from display labels.

Display labels (Italian, shown in reports) may change; the `aid` codes are the
stable contract used programmatically (ALR4, cohort tables, tests). labels.py
assigns both; downstream code must match on `aid`, never on display strings.
"""

DISPLAY_TO_AID = {
    'trachea': 'TRACHEA',
    'bronco principale dx': 'RMB',
    'bronco principale sx': 'LMB',
    'bronco intermedio': 'BI',
    'lobare sup dx': 'RUL',
    'lobare medio': 'RML',
    'lobare inf dx': 'RLL',
    'lobare sup sx': 'LUL',
    'lingulare': 'LING',
    'lobare inf sx': 'LLL',
    'tronco basale dx': 'TB_R',
    'tronco basale sx': 'TB_L',
}

def to_aid(display_name):
    """Map a display label to its stable anatomical id (or None)."""
    if not display_name:
        return None
    if display_name in DISPLAY_TO_AID:
        return DISPLAY_TO_AID[display_name]
    # segmental: 'B6 dx' -> 'B6_R', 'B10 sx' -> 'B10_L', 'B1+2 sx' -> 'B1+2_L'
    parts = display_name.split()
    if len(parts) == 2 and parts[0].startswith('B'):
        side = {'dx': 'R', 'sx': 'L'}.get(parts[1])
        if side:
            return f'{parts[0]}_{side}'
    return None

# airway set for the Shimada ALR4 (J Appl Physiol 2025)
ALR4_AIDS = {'TRACHEA', 'RMB', 'LMB', 'BI'}


class QualityError(SystemExit):
    """Pipeline abort with a diagnostic, phase-specific message.

    Raised when an intermediate artifact fails minimal plausibility checks
    (empty mask, implausible volume, ...). Exits non-zero so the CLI stops
    instead of producing a polished report built on bad inputs.
    """

    def __init__(self, phase, message):
        super().__init__(f'[{phase}] QUALITY ERROR: {message}')
