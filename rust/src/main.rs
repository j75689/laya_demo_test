//! Rust runner: snake game driven by Laya decisions (ONNX Runtime). Same flags as python/run.py.
//!
//! Examples:
//!     cargo run --release -- --render
//!     cargo run --release -- --onnx models/laya.int8.onnx --episodes 5
//!     cargo run --release -- --onnx models/finetuned/laya.static256.onnx --provider coreml
//!     cargo run --release -- --backend coreml --render      # pre-compiled Core ML model
//!     cargo run --release -- --backend rule --render --delay 0.05

mod coreml_laya;
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

use coreml_laya::CoreMLLaya;
use laya::{Answer, CoreMLConfig, Laya, Predict, Question};
use objc2_core_ml::MLComputeUnits;
use snake::{Game, ORDER, XorShift32, move_options, render, rule_move, sample_move, state_text};

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Backend {
    Onnx,
    /// Compiled Core ML model (--coreml-dir; python/fetch_coreml.py or python/convert_coreml.py)
    Coreml,
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

    fn to_coreml(self) -> MLComputeUnits {
        match self {
            Units::All => MLComputeUnits::All,
            Units::Gpu => MLComputeUnits::CPUAndGPU,
            Units::Ane => MLComputeUnits::CPUAndNeuralEngine,
            Units::Cpu => MLComputeUnits::CPUOnly,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Units::All => "all",
            Units::Gpu => "gpu",
            Units::Ane => "ane",
            Units::Cpu => "cpu",
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
    /// coreml backend: directory from fetch_coreml.py or convert_coreml.py
    #[arg(long, default_value = "models/coreml")]
    coreml_dir: String,
    /// CoreML compute units, for --provider coreml and --backend coreml
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
    /// Only offer Laya moves that do not end the game
    #[arg(long)]
    safe: bool,
    /// Add each move's outcome to its option text
    #[arg(long)]
    hints: bool,
    /// Draw each move from Laya's probabilities instead of taking the top one
    #[arg(long)]
    sample: bool,
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
    /// The question is rebuilt every step from `base` (see snake::move_options)
    Laya { model: Box<dyn Predict>, id: String, base: Value, safe: bool, hints: bool, sample: Option<XorShift32> },
    Rule,
}

/// Picks the move from the first answer and lists its probabilities in ORDER.
fn to_move(answers: &[Answer]) -> Result<(&'static str, Option<Vec<f32>>)> {
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

impl Decider {
    /// Deterministic players repeat themselves forever once a state repeats, so the game can stop them.
    fn deterministic(&self) -> bool {
        !matches!(self, Decider::Laya { sample: Some(_), .. })
    }

    fn new_episode(&mut self, seed: u64) {
        if let Decider::Laya { sample: Some(rng), .. } = self {
            // Separate stream from the game's food RNG
            *rng = XorShift32::new((seed & 0xFFFF_FFFF) ^ 0x5EED_5EED);
        }
    }

    /// Returns the move and, for Laya, its probabilities in ORDER.
    fn decide(&mut self, g: &Game) -> Result<(&'static str, Option<Vec<f32>>)> {
        match self {
            Decider::Rule => Ok((rule_move(g), None)),
            Decider::Laya { model, id, base, safe, hints, sample } => {
                let options = move_options(
                    g,
                    |d| base["criteria"][d].as_str().unwrap_or_default().to_string(),
                    *safe,
                    *hints,
                );
                if let [(only, _)] = options.as_slice() {
                    // Forced move: nothing to ask the model
                    let probs = ORDER.iter().map(|d| if d == only { 1.0 } else { 0.0 }).collect();
                    return Ok((*only, Some(probs)));
                }
                let mut q = base.clone();
                q["criteria"] = Value::Object(options.into_iter().map(|(d, t)| (d.to_string(), Value::String(t))).collect());
                let (mv, probs) = to_move(&model.predict(&state_text(g), &[Question::from_json(id, &q)?])?)?;
                match (sample, &probs) {
                    (Some(rng), Some(p)) => Ok((sample_move(rng, p), probs)),
                    _ => Ok((mv, probs)),
                }
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
    let (id, base) = qjson
        .as_object()
        .and_then(|m| m.iter().next())
        .context("question.json must be an object with one question")?;
    // Validate the file up front instead of on the first move
    Question::from_json(id, base)?;
    let suffix = format!(
        "{}{}{}",
        if args.safe { "+safe" } else { "" },
        if args.hints { "+hints" } else { "" },
        if args.sample { "+sample" } else { "" }
    );
    let laya = |model: Box<dyn Predict>| Decider::Laya {
        model,
        id: id.clone(),
        base: base.clone(),
        safe: args.safe,
        hints: args.hints,
        sample: args.sample.then(|| XorShift32::new(0)),
    };
    if args.backend == Backend::Coreml {
        let model = CoreMLLaya::load(&root.join(&args.coreml_dir), args.coreml_units.to_coreml())?;
        let dir = std::path::Path::new(&args.coreml_dir).file_name().unwrap().to_string_lossy();
        let name = format!("coreml-{}({dir}){suffix}", args.coreml_units.name());
        return Ok((laya(Box::new(model)), name));
    }
    let onnx_path = root.join(&args.onnx);
    let coreml = if args.provider == Provider::Coreml {
        Some(coreml_config(&root, &onnx_path, args.coreml_units)?)
    } else {
        None
    };
    let provider = if coreml.is_some() { "coreml" } else { "cpu" };
    let model = Laya::load(
        &onnx_path,
        // export_onnx.py writes tokenizer.json and meta.json next to the model
        &onnx_path.with_file_name("tokenizer.json"),
        &onnx_path.with_file_name("meta.json"),
        coreml,
        args.threads,
    )?;
    let file = onnx_path.file_name().unwrap().to_string_lossy();
    Ok((laya(Box::new(model)), format!("onnx-{provider}({file}){suffix}")))
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

    let warm = Game::new(args.size, args.seed, false);
    decider.new_episode(args.seed);
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
        let mut g = Game::new(args.size, args.seed + ep as u64, decider.deterministic());
        decider.new_episode(args.seed + ep as u64);
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
