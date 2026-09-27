"""Snake game core.

rust/src/snake.rs must match this file exactly: same RNG, same rules, same state text.
With the same seed both sides feed Laya byte-identical input, so decisions can be compared step by step.
"""

import math

DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
ORDER = ["up", "down", "left", "right"]
DEADLY = ("wall", "body")
CELL_TEXT = {"wall": "wall (deadly)", "body": "own body (deadly)", "food": "FOOD", "free": "empty"}
MASK32 = 0xFFFFFFFF


class XorShift32:
    """Tiny reproducible RNG; the Rust side uses the same algorithm."""

    def __init__(self, seed: int):
        self.state = (seed & MASK32) or 1

    def next(self) -> int:
        s = self.state
        s ^= (s << 13) & MASK32
        s ^= s >> 17
        s ^= (s << 5) & MASK32
        self.state = s
        return s


class Game:
    def __init__(self, size: int = 10, seed: int = 42):
        self.size = size
        self.rng = XorShift32(seed)
        c = size // 2
        self.body = [(c, c), (c - 1, c), (c - 2, c)]  # body[0] is the head
        self.heading = "right"
        self.score = 0
        self.steps = 0
        self.hunger = 0
        self.alive = True
        self.death = None
        self.food = self._place_food()

    def _place_food(self):
        occupied = set(self.body)
        free = [(x, y) for y in range(self.size) for x in range(self.size) if (x, y) not in occupied]
        if not free:
            return None
        return free[self.rng.next() % len(free)]

    def cell_status(self, d: str) -> str:
        hx, hy = self.body[0]
        dx, dy = DIRS[d]
        nx, ny = hx + dx, hy + dy
        if not (0 <= nx < self.size and 0 <= ny < self.size):
            return "wall"
        if (nx, ny) == self.food:
            return "food"
        # The tail moves away this step, so it is not an obstacle
        if (nx, ny) in self.body[:-1]:
            return "body"
        return "free"

    def step(self, d: str):
        status = self.cell_status(d)
        self.steps += 1
        if status in DEADLY:
            self.alive = False
            self.death = status
            return
        hx, hy = self.body[0]
        dx, dy = DIRS[d]
        self.body.insert(0, (hx + dx, hy + dy))
        self.heading = d
        if status == "food":
            self.score += 1
            self.hunger = 0
            self.food = self._place_food()
            if self.food is None:
                self.alive = False
                self.death = "win"
        else:
            self.body.pop()
            self.hunger += 1
            if self.hunger > self.size * self.size * 2:
                self.alive = False
                self.death = "starved"


def _offset(n: int, pos: str, neg: str, same: str) -> str:
    if n > 0:
        return "%d cells %s" % (n, pos)
    if n < 0:
        return "%d cells %s" % (-n, neg)
    return same


def state_text(g: Game) -> str:
    """Describe the game state in English for Laya to read."""
    hx, hy = g.body[0]
    fx, fy = g.food
    cells = ", ".join("%s = %s" % (d, CELL_TEXT[g.cell_status(d)]) for d in ORDER)
    return "\n".join([
        "Snake game on a %dx%d grid. x increases to the right, y increases downward." % (g.size, g.size),
        "Snake head at (%d,%d), currently moving %s, length %d." % (hx, hy, g.heading, len(g.body)),
        "Food at (%d,%d), which is %s and %s from the head." % (
            fx, fy,
            _offset(fx - hx, "right", "left", "same column"),
            _offset(fy - hy, "down", "up", "same row"),
        ),
        "Next cell for each move: %s." % cells,
    ])


def move_question(g: Game, base: dict, safe: bool = False, hints: bool = False) -> dict:
    """The Laya question for this step, built from shared/question.json.

    safe:  only offer moves that do not end the game (all four if every move is deadly).
    hints: append each move's outcome to its option text, e.g. "hits the wall, game over".
    """
    hx, hy = g.body[0]
    fx, fy = g.food
    status = {d: g.cell_status(d) for d in ORDER}
    moves = [d for d in ORDER if status[d] not in DEADLY] if safe else list(ORDER)
    if not moves:
        moves = list(ORDER)
    criteria = {}
    for d in moves:
        text = base["criteria"][d]
        if hints:
            dx, dy = DIRS[d]
            closer = abs(hx + dx - fx) + abs(hy + dy - fy) < abs(hx - fx) + abs(hy - fy)
            outcome = {
                "wall": "hits the wall, game over",
                "body": "hits its own body, game over",
                "food": "eats the food",
                "free": "moves closer to the food" if closer else "moves away from the food",
            }[status[d]]
            text = "%s; %s" % (text, outcome)
        criteria[d] = text
    return dict(base, criteria=criteria)


def rule_move(g: Game) -> str:
    """Baseline: among safe moves pick the one closest to the food; ties follow ORDER."""
    hx, hy = g.body[0]
    fx, fy = g.food
    best, best_dist = ORDER[0], math.inf
    for d in ORDER:
        if g.cell_status(d) in DEADLY:
            continue
        dx, dy = DIRS[d]
        dist = abs(hx + dx - fx) + abs(hy + dy - fy)
        if dist < best_dist:
            best, best_dist = d, dist
    return best


def render(g: Game) -> str:
    rows = []
    head, body = g.body[0], set(g.body[1:])
    border = "+" + "--" * g.size + "+"
    rows.append(border)
    for y in range(g.size):
        line = []
        for x in range(g.size):
            if (x, y) == head:
                line.append("\x1b[1;32m@@\x1b[0m")
            elif (x, y) in body:
                line.append("\x1b[32moo\x1b[0m")
            elif (x, y) == g.food:
                line.append("\x1b[1;31m**\x1b[0m")
            else:
                line.append(" .")
        rows.append("|" + "".join(line) + "|")
    rows.append(border)
    return "\n".join(rows)
