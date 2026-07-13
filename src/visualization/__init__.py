import numpy as np

STATE_COLORS = {
    "Progenitor": "#7f7f7f",
    "Branch_A": "#1f77b4",
    "Branch_B": "#d62728",
    "Trunk": "#2ca02c",
}
MODALITY_COLORS = {"RNA": "#1f77b4", "Protein": "#ff7f0e"}

TAB10_FALLBACK_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def state_color_map(states) -> dict[str, str]:
    unique = sorted({str(s) for s in np.unique(states)})
    colors: dict[str, str] = {}
    fallback_i = 0
    for state in unique:
        if state in STATE_COLORS:
            colors[state] = STATE_COLORS[state]
        else:
            colors[state] = TAB10_FALLBACK_COLORS[fallback_i % len(TAB10_FALLBACK_COLORS)]
            fallback_i += 1
    return colors
