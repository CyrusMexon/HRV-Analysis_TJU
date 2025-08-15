import numpy as np

def poincare(rr: list[float]):
    """Compute Poincaré SD1 and SD2.
    TODO: Add option to plot the Poincaré scatter diagram.
    """
    rr = np.array(rr)
    rr1 = rr[:-1]
    rr2 = rr[1:]
    diff = rr2 - rr1
    sd1 = np.sqrt(np.var(diff) / 2)
    sd2 = np.sqrt(2*np.var(rr) - (np.var(diff) / 2))
    return sd1, sd2

def sample_entropy(rr: list[float], m: int = 2, r: float = 0.2) -> float:
    """Calculate Sample Entropy.
    TODO: Implement actual Sample Entropy calculation.
    """
    raise NotImplementedError("Sample entropy not yet implemented.")
