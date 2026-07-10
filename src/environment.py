import logging
import numpy as np
import torch
import copy
from typing import Dict, Tuple

logger = logging.getLogger("SmellRL.env")

# --- Refactoring Actions & Smells indices ---
# Actions:
STRATEGY = 0
OBSERVER = 1
FACADE = 2
MEDIATOR = 3
EXTRACT = 4
NO_REFACTOR = 5

# Smells:
GOD_CLASS = 0
FEATURE_ENVY = 1
LONG_METHOD = 2
DATA_CLASS = 3
NO_SMELL = 4

# --- Fowler (1999) grounded Soft Reward Matrix ---
# Mapping true smell index (0-4) to refactoring action index (0-5)
SOFT_REWARD_MATRIX = {
    # GodClass: Facade=+2 (Fowler §12), Mediator=+1 (Brown 1998), ExtractClass=+0.5
    GOD_CLASS: {STRATEGY: -1.0, OBSERVER: -0.5, FACADE: +2.0, MEDIATOR: +1.0, EXTRACT: +0.5, NO_REFACTOR: -2.0},
    # FeatureEnvy: Strategy=+2 (Fowler §7), ExtractClass=+0.5
    FEATURE_ENVY: {STRATEGY: +2.0, OBSERVER: -1.0, FACADE: -1.0, MEDIATOR: -0.5, EXTRACT: +0.5, NO_REFACTOR: -2.0},
    # LongMethod: ExtractClass=+2 (Fowler §6)
    LONG_METHOD: {STRATEGY: -1.0, OBSERVER: -1.0, FACADE: -0.5, MEDIATOR: -1.0, EXTRACT: +2.0, NO_REFACTOR: -2.0},
    # DataClass: Observer=+2 (Fowler §11)
    DATA_CLASS: {STRATEGY: -1.0, OBSERVER: +2.0, FACADE: -0.5, MEDIATOR: -0.5, EXTRACT: -1.0, NO_REFACTOR: -2.0},
    # NoSmell: NoRefactor=+2
    NO_SMELL: {STRATEGY: -1.0, OBSERVER: -1.0, FACADE: -1.0, MEDIATOR: -1.0, EXTRACT: -1.0, NO_REFACTOR: +2.0},
}

# State Transitions (Murphy-Hill et al. 2012 ICSE grounded perturbation bounds)
TRANSITIONS = {
    FACADE:      {"cyclomatic": (0.60, 0.85), "lloc": (0.70, 0.90)},
    STRATEGY:    {"D": (0.70, 0.90)},
    EXTRACT:     {"cyclomatic": (0.55, 0.75), "lloc": (0.55, 0.75)},
    OBSERVER:    {"D": (0.75, 0.90)},
    MEDIATOR:    {"cyclomatic": (0.70, 0.90), "D": (0.80, 0.95)},
    NO_REFACTOR: {},
}

# Halstead Metric bounds for normalisation bounds (aligned with config / data)
_HALSTEAD_MAX = {
    "cyclomatic": 50.0,
    "lloc": 5000.0,
    "V": 5000.0,
    "D": 100.0,
    "E": 1000000.0,
    "B": 5.0,
    "dit": 1.0,  # kept for placeholder alignment
    "noc": 5.0,  # mapped to B
    "cbo": 100.0, # mapped to D
    "rfc": 5000.0, # mapped to V
    "loc": 5000.0, # mapped to lloc
    "wmc": 50.0,  # mapped to cyclomatic
}

def _norm(val: float, key: str) -> float:
    return float(np.clip(val, 0, _HALSTEAD_MAX.get(key, 100.0)) / _HALSTEAD_MAX.get(key, 100.0))

def apply_simulated_refactoring(item: dict, action: int) -> dict:
    """
    Simulate code refactoring by perturbing Halstead features in the structural vector.
    Grounded in Murphy-Hill et al. (2012) ICSE empirical study.
    No fallbacks. Raises ValueError on invalid action.
    """
    if action not in TRANSITIONS:
        raise ValueError(f"Invalid refactoring action index {action}. Must be 0-5.")

    perturbations = TRANSITIONS[action]
    if not perturbations:
        logger.debug(f"[Env] Action NoRefactor applied; state unchanged.")
        return item

    new_item = copy.deepcopy(item)
    
    # We update both the unnormalized "halstead" tensor and the normalized structural features in new_item["x"]
    # x node features shape [N, 783]
    # For class node (index 0), structural features are located at x[0, 0:15]
    # idx 3 = cyclomatic (wmc), idx 4 = lloc (loc), idx 5 = V (rfc), idx 6 = D (cbo), idx 7 = E, idx 8 = B (noc)
    # raw halstead tensor layout: [cyclomatic, lloc, V, D, E, B]
    
    halstead = new_item["halstead"].clone().float() # [6]
    
    # Read current values
    curr_cyclomatic = halstead[0].item()
    curr_lloc = halstead[1].item()
    curr_V = halstead[2].item()
    curr_D = halstead[3].item()
    curr_E = halstead[4].item()
    curr_B = halstead[5].item()
    
    # Apply perturbations (randomly sampled within the grounded range)
    if "cyclomatic" in perturbations:
        mult = np.random.uniform(*perturbations["cyclomatic"])
        curr_cyclomatic *= mult
    if "lloc" in perturbations:
        mult = np.random.uniform(*perturbations["lloc"])
        curr_lloc *= mult
        # Volume, Effort, and Bugs scale with LOC reduction
        curr_V *= mult
        curr_E *= mult
        curr_B *= mult
    if "D" in perturbations:
        mult = np.random.uniform(*perturbations["D"])
        curr_D *= mult
        curr_E *= mult # Effort depends on Difficulty too
        
    # Write back to unnormalized halstead
    halstead[0] = curr_cyclomatic
    halstead[1] = curr_lloc
    halstead[2] = curr_V
    halstead[3] = curr_D
    halstead[4] = curr_E
    halstead[5] = curr_B
    new_item["halstead"] = halstead
    
    # Update normalized values in the graph features tensor new_item["x"]
    # Structural features are on class node (index 0) and method/field nodes
    # Let's update the class node (row 0 in x)
    x = new_item["x"].clone()
    
    x[0, 3] = _norm(curr_cyclomatic, "cyclomatic")
    x[0, 4] = _norm(curr_lloc, "lloc")
    x[0, 5] = _norm(curr_V, "V")
    x[0, 6] = _norm(curr_D, "D")
    x[0, 7] = _norm(curr_E, "E")
    x[0, 8] = _norm(curr_B, "B")
    
    new_item["x"] = x
    logger.debug(
        f"[Env] Action {action} applied: cyclomatic={curr_cyclomatic:.1f}, "
        f"lloc={curr_lloc:.1f}, V={curr_V:.1f}, D={curr_D:.1f}, E={curr_E:.1f}, B={curr_B:.1f}"
    )
    return new_item

def quality_delta_reward(item_before: dict, item_after: dict) -> float:
    """
    Calculate the reward based on the sum of normalized metric improvements.
    Uses structural class features index 3 to 8.
    Positive values represent a reduction in complexity, coupling, or size.
    Negative values indicate a penalty.
    """
    cf_before = item_before["x"][0, 3:9].cpu().numpy()
    cf_after  = item_after["x"][0, 3:9].cpu().numpy()
    # Positive delta means value decreased (good refactoring)
    delta = float((cf_before - cf_after).sum())
    reward = float(np.clip(delta * 2.0, -2.0, +2.0))
    logger.debug(f"[Env] Computed quality delta reward: {reward:.4f} (delta sum: {delta:.4f})")
    return reward
