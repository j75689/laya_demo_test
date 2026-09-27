//! Snake game core. Must match python/snake.py exactly (same RNG, rules and state text)
//! so both runners feed Laya identical input and decisions can be compared step by step.

use std::collections::HashSet;

pub const ORDER: [&str; 4] = ["up", "down", "left", "right"];

fn delta(d: &str) -> (i32, i32) {
    match d {
        "up" => (0, -1),
        "down" => (0, 1),
        "left" => (-1, 0),
        _ => (1, 0),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    Wall,
    Body,
    Food,
    Free,
}

impl Cell {
    pub fn deadly(self) -> bool {
        matches!(self, Cell::Wall | Cell::Body)
    }

    fn text(self) -> &'static str {
        match self {
            Cell::Wall => "wall (deadly)",
            Cell::Body => "own body (deadly)",
            Cell::Food => "FOOD",
            Cell::Free => "empty",
        }
    }
}

/// Tiny reproducible RNG; the Python side uses the same algorithm.
pub struct XorShift32(u32);

impl XorShift32 {
    pub fn new(seed: u64) -> Self {
        let s = (seed & 0xFFFF_FFFF) as u32;
        Self(if s == 0 { 1 } else { s })
    }

    pub fn next(&mut self) -> u32 {
        let mut s = self.0;
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        self.0 = s;
        s
    }
}

pub struct Game {
    pub size: i32,
    rng: XorShift32,
    /// body[0] is the head
    pub body: Vec<(i32, i32)>,
    pub heading: &'static str,
    pub score: u32,
    pub steps: u32,
    hunger: u32,
    pub alive: bool,
    pub death: Option<&'static str>,
    pub food: Option<(i32, i32)>,
    detect_loops: bool,
    seen: HashSet<(Vec<(i32, i32)>, Option<(i32, i32)>)>,
}

impl Game {
    /// detect_loops: end the game as "loop" when a state repeats. Only valid for a deterministic
    /// player, which would then repeat the same moves forever.
    pub fn new(size: i32, seed: u64, detect_loops: bool) -> Self {
        let c = size / 2;
        let mut g = Self {
            size,
            rng: XorShift32::new(seed),
            body: vec![(c, c), (c - 1, c), (c - 2, c)],
            heading: "right",
            score: 0,
            steps: 0,
            hunger: 0,
            alive: true,
            death: None,
            food: None,
            detect_loops,
            seen: HashSet::new(),
        };
        g.food = g.place_food();
        g.seen.insert(g.key());
        g
    }

    fn key(&self) -> (Vec<(i32, i32)>, Option<(i32, i32)>) {
        (self.body.clone(), self.food)
    }

    fn place_food(&mut self) -> Option<(i32, i32)> {
        let mut free = Vec::new();
        for y in 0..self.size {
            for x in 0..self.size {
                if !self.body.contains(&(x, y)) {
                    free.push((x, y));
                }
            }
        }
        if free.is_empty() {
            return None;
        }
        let i = self.rng.next() as usize % free.len();
        Some(free[i])
    }

    pub fn cell_status(&self, d: &str) -> Cell {
        let (hx, hy) = self.body[0];
        let (dx, dy) = delta(d);
        let (nx, ny) = (hx + dx, hy + dy);
        if !(0..self.size).contains(&nx) || !(0..self.size).contains(&ny) {
            return Cell::Wall;
        }
        if Some((nx, ny)) == self.food {
            return Cell::Food;
        }
        // The tail moves away this step, so it is not an obstacle
        if self.body[..self.body.len() - 1].contains(&(nx, ny)) {
            return Cell::Body;
        }
        Cell::Free
    }

    pub fn step(&mut self, d: &'static str) {
        let status = self.cell_status(d);
        self.steps += 1;
        match status {
            Cell::Wall => {
                self.alive = false;
                self.death = Some("wall");
                return;
            }
            Cell::Body => {
                self.alive = false;
                self.death = Some("body");
                return;
            }
            _ => {}
        }
        let (hx, hy) = self.body[0];
        let (dx, dy) = delta(d);
        self.body.insert(0, (hx + dx, hy + dy));
        self.heading = d;
        if status == Cell::Food {
            self.score += 1;
            self.hunger = 0;
            self.food = self.place_food();
            if self.food.is_none() {
                self.alive = false;
                self.death = Some("win");
                return;
            }
            // The snake is longer now, so earlier states can never come back
            self.seen.clear();
        } else {
            self.body.pop();
            self.hunger += 1;
            if self.hunger > (self.size * self.size * 2) as u32 {
                self.alive = false;
                self.death = Some("starved");
                return;
            }
        }
        let key = self.key();
        if self.detect_loops && self.seen.contains(&key) {
            self.alive = false;
            self.death = Some("loop");
        }
        self.seen.insert(key);
    }
}

fn offset(n: i32, pos: &str, neg: &str, same: &str) -> String {
    if n > 0 {
        format!("{} cells {}", n, pos)
    } else if n < 0 {
        format!("{} cells {}", -n, neg)
    } else {
        same.to_string()
    }
}

/// Describe the game state in English for Laya to read.
pub fn state_text(g: &Game) -> String {
    let (hx, hy) = g.body[0];
    let (fx, fy) = g.food.expect("a running game always has food");
    let cells: Vec<String> = ORDER
        .iter()
        .map(|d| format!("{} = {}", d, g.cell_status(d).text()))
        .collect();
    [
        format!(
            "Snake game on a {}x{} grid. x increases to the right, y increases downward.",
            g.size, g.size
        ),
        format!(
            "Snake head at ({},{}), currently moving {}, length {}.",
            hx,
            hy,
            g.heading,
            g.body.len()
        ),
        format!(
            "Food at ({},{}), which is {} and {} from the head.",
            fx,
            fy,
            offset(fx - hx, "right", "left", "same column"),
            offset(fy - hy, "down", "up", "same row")
        ),
        format!("Next cell for each move: {}.", cells.join(", ")),
    ]
    .join("\n")
}

/// Options for this step's Laya question; mirrors move_question() in python/snake.py.
///
/// safe:  only offer moves that do not end the game (all four if every move is deadly).
/// hints: append each move's outcome to its option text, e.g. "hits the wall, game over".
pub fn move_options(
    g: &Game,
    base: impl Fn(&str) -> String,
    safe: bool,
    hints: bool,
) -> Vec<(&'static str, String)> {
    let (hx, hy) = g.body[0];
    let (fx, fy) = g.food.expect("a running game always has food");
    let mut moves: Vec<&'static str> = ORDER.into_iter().filter(|d| !safe || !g.cell_status(d).deadly()).collect();
    if moves.is_empty() {
        moves = ORDER.to_vec();
    }
    moves
        .into_iter()
        .map(|d| {
            let mut text = base(d);
            if hints {
                let (dx, dy) = delta(d);
                let closer = (hx + dx - fx).abs() + (hy + dy - fy).abs() < (hx - fx).abs() + (hy - fy).abs();
                let outcome = match g.cell_status(d) {
                    Cell::Wall => "hits the wall, game over",
                    Cell::Body => "hits its own body, game over",
                    Cell::Food => "eats the food",
                    Cell::Free if closer => "moves closer to the food",
                    Cell::Free => "moves away from the food",
                };
                text = format!("{text}; {outcome}");
            }
            (d, text)
        })
        .collect()
}

/// Draw a move from Laya's probabilities (listed in ORDER). Mirrors sample_move() in python/snake.py.
///
/// Probabilities are rounded to 4 decimals first, the precision both runners report, so Python and
/// Rust draw the same move from the same numbers.
pub fn sample_move(rng: &mut XorShift32, probs: &[f32]) -> &'static str {
    let p: Vec<f64> = probs.iter().map(|&v| (v as f64 * 10000.0 + 0.5).floor() / 10000.0).collect();
    let target = rng.next() as f64 / 4294967296.0 * p.iter().sum::<f64>();
    let mut acc = 0.0;
    for (d, v) in ORDER.iter().zip(&p) {
        acc += v;
        if target < acc {
            return d;
        }
    }
    let best = (0..ORDER.len()).fold(0, |b, i| if probs[i] > probs[b] { i } else { b });
    ORDER[best]
}

/// Baseline: among safe moves pick the one closest to the food; ties follow ORDER.
pub fn rule_move(g: &Game) -> &'static str {
    let (hx, hy) = g.body[0];
    let (fx, fy) = g.food.expect("a running game always has food");
    let mut best = ORDER[0];
    let mut best_dist = i32::MAX;
    for d in ORDER {
        if g.cell_status(d).deadly() {
            continue;
        }
        let (dx, dy) = delta(d);
        let dist = (hx + dx - fx).abs() + (hy + dy - fy).abs();
        if dist < best_dist {
            best = d;
            best_dist = dist;
        }
    }
    best
}

pub fn render(g: &Game) -> String {
    let border = format!("+{}+", "--".repeat(g.size as usize));
    let mut rows = vec![border.clone()];
    for y in 0..g.size {
        let mut line = String::from("|");
        for x in 0..g.size {
            let p = (x, y);
            if p == g.body[0] {
                line.push_str("\x1b[1;32m@@\x1b[0m");
            } else if g.body[1..].contains(&p) {
                line.push_str("\x1b[32moo\x1b[0m");
            } else if Some(p) == g.food {
                line.push_str("\x1b[1;31m**\x1b[0m");
            } else {
                line.push_str(" .");
            }
        }
        line.push('|');
        rows.push(line);
    }
    rows.push(border);
    rows.join("\n")
}
