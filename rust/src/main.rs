//! Rust runner: snake game driven by Laya decisions (ONNX Runtime). Same flags as python/run.py.
//!
//! Examples:
//!     cargo run --release -- --render
//!     cargo run --release -- --onnx models/laya.int8.onnx --episodes 5
//!     cargo run --release -- --backend rule --render --delay 0.05

mod laya;
mod snake;

use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use ort::ep::coreml::ComputeUnits;
use serde_json::{Value, json};

use laya::{CoreMLConfig, Laya, Question};
use snake::{Game, ORDER, render, rule_move, state_text};

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Backend {
    Onnx,
    Rule,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Provider {
    Cpu,
    Coreml,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Units {
    All,
    /// CPU + GPU
    Gpu,
    /// CPU + Neural Engine
    Ane,
    Cpu,
}

impl Units {
    fn to_ort(self) -> (ComputeUnits, &'static str) {
        match self {
            Units::All => (ComputeUnits::All, "ALL"),
            Units::Gpu => (ComputeUnits::CPUAndGPU, "CPUAndGPU"),
            Units::Ane => (ComputeUnits::CPUAndNeuralEngine, "CPUAndNeuralEngine"),
            Units::Cpu => (ComputeUnits::CPUOnly, "CPUOnly"),
        }
    }
}

#[derive(Parser)]
#[command(about = "Rust snake + Laya")]
struct Args {
    #[arg(long, value_enum, default_value = "onnx")]
    backend: Backend,
    /// onnx backend: model path, relative to the project root
    #[arg(long, default_value = "models/laya.onnx")]
    onnx: String,
    #[arg(long, value_enum, default_value = "cpu")]
    provider: Provider,
    /// coreml provider: compute units
    #[arg(long, value_enum, default_value = "all")]
    coreml_units: Units,
    /// CPU threads for inference (0 = library default)
    #[arg(long, default_value_t = 0)]
    threads: usize,
    #[arg(long, default_value_t = 3)]
    episodes: u32,
    #[arg(long, default_value_t = 10)]
    size: i32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 300)]
    max_steps: u32,
    /// Untimed inferences before measuring
    #[arg(long, default_value_t = 3)]
    warmup: u32,
    /// Draw the game in the terminal
    #[arg(long)]
    render: bool,
    /// Extra pause per step in seconds, for watching
    #[arg(long, default_value_t = 0.0)]
    delay: f64,
    /// Write every decision as JSONL, to diff against the Python runner
    #[arg(long)]
    trace: Option<PathBuf>,
    /// Print a one-line JSON summary at the end
    #[arg(long)]
    json: bool,
}

/// Project root (the parent of rust/), so models/ and shared/ resolve from any working directory.
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

enum Decider {
    Laya { model: Laya, questions: Vec<Question> },
    Rule,
}

impl Decider {
    /// Returns the move and, for Laya, its probabilities in ORDER.
    fn decide(&mut self, g: &Game) -> Result<(&'static str, Option<Vec<f32>>)> {
        match self {
            Decider::Rule => Ok((rule_move(g), None)),
            Decider::Laya { model, questions } => {
                let answers = model.predict(&state_text(g), questions)?;
                let a = &answers[0];
                let mv = ORDER
                    .into_iter()
                    .find(|d| *d == a.choice())
                    .context("model returned an unknown direction")?;
                let probs = ORDER
                    .iter()
                    .map(|d| a.labels.iter().position(|l| l == d).map_or(0.0, |i| a.probs[i]))
                    .collect();
                Ok((mv, Some(probs)))
            }
        }
    }
}

fn build_decider(args: &Args) -> Result<(Decider, String)> {
    if args.backend == Backend::Rule {
        return Ok((Decider::Rule, "rule".into()));
    }
    let root = root();
    let qjson: Value = serde_json::from_str(&std::fs::read_to_string(root.join("shared/question.json"))?)?;
    let questions = qjson
        .as_object()
        .context("question.json must be a JSON object")?
        .iter()
        .map(|(id, q)| Question::from_json(id, q))
        .collect::<Result<Vec<_>>>()?;
    let onnx_path = root.join(&args.onnx);
    let coreml = if args.provider == Provider::Coreml {
        Some(coreml_config(&root, &onnx_path, args.coreml_units)?)
    } else {
        None
    };
    let provider = if coreml.is_some() { "coreml" } else { "cpu" };
    let model = Laya::load(
        &onnx_path,
        &root.join("models/tokenizer.json"),
        &root.join("models/meta.json"),
        coreml,
        args.threads,
    )?;
    let file = onnx_path.file_name().unwrap().to_string_lossy();
    Ok((Decider::Laya { model, questions }, format!("onnx-{provider}({file})")))
}

/// Same cache layout as python/run.py. ONNX Runtime does not notice a re-exported model,
/// so the directory name includes the model's mtime.
fn coreml_config(root: &std::path::Path, onnx_path: &std::path::Path, units: Units) -> Result<CoreMLConfig> {
    let (units, units_name) = units.to_ort();
    let stem = onnx_path.file_name().unwrap().to_string_lossy();
    let stem = stem.strip_suffix(".onnx").unwrap_or(&stem);
    let mtime = std::fs::metadata(onnx_path)?
        .modified()?
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    let cache = root.join("models/coreml_cache").join(format!("{stem}-{mtime}-{units_name}"));
    std::fs::create_dir_all(&cache)?;
    Ok(CoreMLConfig { units, cache_dir: cache.to_string_lossy().into_owned() })
}

/// Nearest-rank percentile, same definition as python/run.py.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0 * sorted.len() as f64).ceil() as usize).max(1) - 1;
    sorted[idx]
}

fn fmt_probs(probs: &Option<Vec<f32>>) -> String {
    match probs {
        None => String::new(),
        Some(p) => format!(
            "({})",
            ORDER.iter().zip(p).map(|(d, v)| format!("{d} {v:.2}")).collect::<Vec<_>>().join(" | ")
        ),
    }
}

/// Laya's Python API rounds probabilities to 4 decimals; do the same so traces diff cleanly.
fn round4(v: f32) -> f64 {
    (v as f64 * 10000.0).round() / 10000.0
}

fn main() -> Result<()> {
    let args = Args::parse();

    let t0 = Instant::now();
    let (mut decider, name) = build_decider(&args)?;
    let load_s = t0.elapsed().as_secs_f64();

    let warm = Game::new(args.size, args.seed);
    for _ in 0..args.warmup {
        decider.decide(&warm)?;
    }

    let mut trace = match &args.trace {
        Some(p) => Some(std::io::BufWriter::new(std::fs::File::create(p)?)),
        None => None,
    };
    let mut latencies: Vec<f64> = Vec::new();
    let mut scores: Vec<u32> = Vec::new();
    let mut deaths: BTreeMap<&str, u32> = BTreeMap::new();

    for ep in 0..args.episodes {
        let mut g = Game::new(args.size, args.seed + ep as u64);
        while g.alive && g.steps < args.max_steps {
            let t = Instant::now();
            let (mv, probs) = decider.decide(&g)?;
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            latencies.push(ms);
            if let Some(w) = trace.as_mut() {
                let probs_json = probs.as_ref().map(|p| {
                    ORDER
                        .iter()
                        .zip(p)
                        .map(|(d, v)| (d.to_string(), json!(round4(*v))))
                        .collect::<serde_json::Map<_, _>>()
                });
                writeln!(w, "{}", json!({"ep": ep, "step": g.steps, "move": mv, "probs": probs_json}))?;
            }
            g.step(mv);
            if args.render {
                let avg = latencies.iter().sum::<f64>() / latencies.len() as f64;
                print!("\x1b[H\x1b[2J");
                println!(
                    "[rust / {}]  episode {}/{}  step {}  score {}",
                    name,
                    ep + 1,
                    args.episodes,
                    g.steps,
                    g.score
                );
                println!("{}", render(&g));
                println!("decision: {:<5} {}", mv, fmt_probs(&probs));
                println!("latency : {ms:.2} ms (avg {avg:.2} ms)");
                if !g.alive {
                    println!("GAME OVER: {}", g.death.unwrap_or(""));
                    std::thread::sleep(Duration::from_secs(1));
                }
            }
            if args.delay > 0.0 {
                std::thread::sleep(Duration::from_secs_f64(args.delay));
            }
        }
        scores.push(g.score);
        *deaths.entry(g.death.unwrap_or("max_steps")).or_default() += 1;
    }
    if let Some(mut w) = trace {
        w.flush()?;
    }

    let mut s = latencies;
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = if s.is_empty() { 0.0 } else { s.iter().sum::<f64>() / s.len() as f64 };
    let avg_score = scores.iter().sum::<u32>() as f64 / scores.len().max(1) as f64;
    println!();
    println!("=== rust / {name} ===");
    println!("model load   : {load_s:.2} s");
    println!("episodes     : {}   scores {:?}   avg {:.2}", args.episodes, scores, avg_score);
    println!(
        "game over    : {}",
        deaths.iter().map(|(k, v)| format!("{k} {v}")).collect::<Vec<_>>().join(", ")
    );
    println!("decisions    : {}", s.len());
    println!(
        "latency (ms) : mean {:.3}   p50 {:.3}   p95 {:.3}   min {:.3}",
        mean,
        percentile(&s, 50.0),
        percentile(&s, 95.0),
        s.first().copied().unwrap_or(0.0)
    );
    println!("throughput   : {:.1} decisions/s", if mean > 0.0 { 1000.0 / mean } else { 0.0 });
    if args.json {
        let r3 = |v: f64| (v * 1000.0).round() / 1000.0;
        let r6 = |v: f64| (v * 1e6).round() / 1e6;
        println!(
            "{}",
            json!({
                "impl": "rust", "backend": name, "load_s": r3(load_s), "scores": scores,
                "decisions": s.len(), "mean_ms": r6(mean), "p50_ms": r6(percentile(&s, 50.0)),
                "p95_ms": r6(percentile(&s, 95.0)),
            })
        );
    }
    Ok(())
}
