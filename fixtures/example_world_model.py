# Example agent-authored world-model program for the GridPursuit toy game.
#
# This is what a model would converge to after a few transitions: a single small
# program that grounds the grid, encodes the move mechanism, defines the goal,
# and renders state back to a grid. It is loaded into the sandbox, which injects
# `Action` and `Frame`, so it needs no imports. It is intentionally simple —
# the "simplest program that fits the observations", per Schema's MDL-like bias.

_MOVES = {"ACTION1": (-1, 0), "ACTION2": (1, 0), "ACTION3": (0, -1), "ACTION4": (0, 1)}


def _find(grid, colour):
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == colour:
                return (r, c)
    return None


def ground(frame):
    grid = frame.grid
    return {
        "cursor": _find(grid, 2),
        "target": _find(grid, 3),
        "h": len(grid),
        "w": len(grid[0]) if grid else 0,
    }


def step(state, action):
    dr, dc = _MOVES.get(action.name, (0, 0))
    cr, cc = state["cursor"]
    nr = max(0, min(state["h"] - 1, cr + dr))
    nc = max(0, min(state["w"] - 1, cc + dc))
    out = dict(state)
    out["cursor"] = (nr, nc)
    return out


def is_goal(state):
    return state["cursor"] == state["target"]


def render(state):
    h, w = state["h"], state["w"]
    grid = [[0 for _ in range(w)] for _ in range(h)]
    tr, tc = state["target"]
    grid[tr][tc] = 3
    cr, cc = state["cursor"]
    grid[cr][cc] = 2
    return grid


def actions(state):
    return [Action("ACTION1"), Action("ACTION2"), Action("ACTION3"), Action("ACTION4")]
