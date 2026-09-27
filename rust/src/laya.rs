//! Rust port of Laya inference. Depends only on ONNX Runtime (ort) and HF tokenizers, no Python.
//!
//! Mirrors the official Python code:
//! - input assembly   laya/common.py       render_options / build_sequence / collate_items
//! - post-processing  laya/onnx_agent.py   ONNXAgent._infer (temperature scaling + softmax)

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use ndarray::{Array1, Array2};
use ort::ep::coreml::{ComputeUnits, ModelFormat};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

/// models/meta.json written by python/export_onnx.py
#[derive(Deserialize)]
pub struct Meta {
    cls_id: i64,
    sep_id: i64,
    mask_id: i64,
    pad_id: i64,
    mask_token: String,
    max_len: usize,
    head_max_len: usize,
    temperature: Vec<f32>,
    temperature_by_options: HashMap<String, f32>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum QType {
    Choice = 0,
    Score = 1,
    Noul = 2,
}

impl QType {
    fn name(self) -> &'static str {
        match self {
            QType::Choice => "choice",
            QType::Score => "score",
            QType::Noul => "noul",
        }
    }
}

pub struct Question {
    pub id: String,
    qtype: QType,
    instructions: String,
    /// Option names: criteria keys for choice, "0".."k-1" for score, false/true for noul
    labels: Vec<String>,
    /// Option text fed to the model, as render_options() builds it
    options: Vec<String>,
}

/// Mirrors render_criterion(): strings pass through, anything else is rendered like
/// Python json.dumps(separators=(", ", ": ")).
fn render_criterion(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(a) => format!("[{}]", a.iter().map(render_json).collect::<Vec<_>>().join(", ")),
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter()
                .map(|(k, v)| format!("{}: {}", Value::String(k.clone()), render_json(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        other => other.to_string(),
    }
}

fn render_json(v: &Value) -> String {
    match v {
        Value::String(_) => v.to_string(),
        _ => render_criterion(v),
    }
}

fn is_blank(v: &Value) -> bool {
    v.is_null() || v.as_str() == Some("")
}

impl Question {
    /// Build from the same question JSON the Python API takes (one entry of shared/question.json).
    pub fn from_json(id: &str, q: &Value) -> Result<Self> {
        let qtype = match q["type"].as_str() {
            Some("choice") => QType::Choice,
            Some("score") => QType::Score,
            Some("noul") => QType::Noul,
            other => bail!("question {id:?}: unknown type {other:?}"),
        };
        if q.get("labels").is_some() {
            bail!("question {id:?}: custom noul labels are not supported in this port");
        }
        let instructions = match &q["instructions"] {
            Value::String(s) => s.clone(),
            other => bail!("question {id:?}: instructions must be a string, got {other}"),
        };
        let crit = &q["criteria"];
        let (labels, options) = match qtype {
            QType::Choice => match crit {
                Value::Object(m) => m
                    .iter()
                    .map(|(k, v)| {
                        let opt = if is_blank(v) { k.clone() } else { format!("{k}: {}", render_criterion(v)) };
                        (k.clone(), opt)
                    })
                    .unzip(),
                Value::Array(a) => a
                    .iter()
                    .map(|k| {
                        let k = k.as_str().unwrap_or_default().to_string();
                        (k.clone(), k)
                    })
                    .unzip(),
                _ => bail!("question {id:?}: choice criteria must be an object or a list"),
            },
            QType::Score => {
                let a = crit.as_array().ok_or_else(|| anyhow!("question {id:?}: score criteria must be a list"))?;
                a.iter()
                    .enumerate()
                    .map(|(i, c)| (i.to_string(), format!("level {i}: {}", render_criterion(c))))
                    .unzip()
            }
            QType::Noul => {
                let get = |key: &str, default: &str| match crit.get(key) {
                    Some(v) if !is_blank(v) => render_criterion(v),
                    _ => default.to_string(),
                };
                (
                    vec!["false".into(), "true".into()],
                    vec![
                        format!("false: {}", get("false", "no, the statement does not hold")),
                        format!("true: {}", get("true", "yes, the statement holds")),
                    ],
                )
            }
        };
        Ok(Self { id: id.to_string(), qtype, instructions, labels, options })
    }
}

pub struct Answer {
    pub labels: Vec<String>,
    pub probs: Vec<f32>,
}

impl Answer {
    pub fn choice(&self) -> &str {
        let best = self
            .probs
            .iter()
            .enumerate()
            .fold(0, |best, (i, p)| if *p > self.probs[best] { i } else { best });
        &self.labels[best]
    }
}

/// CoreML settings. MLProgram is the format that takes nearly the whole Laya graph.
pub struct CoreMLConfig {
    pub units: ComputeUnits,
    /// Compiling takes about a minute, so the compiled model is cached here
    pub cache_dir: String,
}

pub struct Laya {
    session: Session,
    tok: Tokenizer,
    meta: Meta,
    /// Fixed sequence length of a static-shape model; inputs are right-padded to it
    pad_to: Option<usize>,
}

impl Laya {
    /// `threads == 0` keeps ONNX Runtime's default intra-op thread count.
    pub fn load(
        onnx_path: &Path,
        tokenizer_path: &Path,
        meta_path: &Path,
        coreml: Option<CoreMLConfig>,
        threads: usize,
    ) -> Result<Self> {
        let meta: Meta = serde_json::from_str(
            &std::fs::read_to_string(meta_path).with_context(|| format!("reading {}", meta_path.display()))?,
        )?;
        let mut tok = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow!("failed to load tokenizer: {e}"))?;
        tok.with_truncation(None).map_err(|e| anyhow!("{e}"))?;
        tok.with_padding(None);

        // SessionBuilder errors are not Send, so stringify them for anyhow
        let err = |e: ort::Error<_>| anyhow!("ONNX Runtime: {e}");
        let mut builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(err)?;
        if threads > 0 {
            builder = builder.with_intra_threads(threads).map_err(err)?;
        }
        if let Some(cfg) = coreml {
            let ep = ort::ep::CoreML::default()
                .with_model_format(ModelFormat::MLProgram)
                .with_compute_units(cfg.units)
                .with_model_cache_dir(cfg.cache_dir);
            builder = builder.with_execution_providers([ep.build()]).map_err(err)?;
        }
        let session = builder
            .commit_from_file(onnx_path)
            .with_context(|| format!("loading {}", onnx_path.display()))?;
        let pad_to = session
            .inputs()
            .iter()
            .find(|i| i.name() == "input_ids")
            .and_then(|i| i.dtype().tensor_shape())
            .and_then(|shape| shape.get(1).copied())
            .filter(|&len| len > 0)
            .map(|len| len as usize);
        Ok(Self { session, tok, meta, pad_to })
    }

    fn encode(&self, text: &str) -> Result<Vec<i64>> {
        let enc = self.tok.encode(text, false).map_err(|e| anyhow!("tokenize failed: {e}"))?;
        Ok(enc.get_ids().iter().map(|&i| i as i64).collect())
    }

    /// Mirrors build_sequence():
    /// [CLS] <type> question: instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
    fn build_sequence(&self, state_ids: &[i64], q: &Question) -> Result<(Vec<i64>, Vec<usize>)> {
        let m = &self.meta;
        let ins = q.instructions.replace(&m.mask_token, " ");
        let mut head_ids = self.encode(&format!("{} question: {}", q.qtype.name(), ins))?;
        let mut opt_ids = Vec::with_capacity(q.options.len());
        for opt in &q.options {
            let mut toks = self.encode(&format!(" {}", opt.replace(&m.mask_token, " ")))?;
            toks.truncate(48);
            let mut o = vec![m.mask_id];
            o.extend(toks);
            opt_ids.push(o);
        }
        let head_max = m.head_max_len as i64;
        let mut opt_budget = head_max - opt_ids.iter().map(|o| o.len() as i64).sum::<i64>();
        if opt_budget < 16 {
            let per = 4.max((head_max - 16) / (opt_ids.len().max(1) as i64)) as usize;
            opt_ids.iter_mut().for_each(|o| o.truncate(per));
            opt_budget = head_max - opt_ids.iter().map(|o| o.len() as i64).sum::<i64>();
        }
        head_ids.truncate(8.max(opt_budget) as usize);

        let mut ids = vec![m.cls_id];
        ids.extend(head_ids);
        ids.push(m.sep_id);
        let mut markers = Vec::with_capacity(opt_ids.len());
        for o in opt_ids {
            markers.push(ids.len());
            ids.extend(o);
        }
        ids.push(m.sep_id);
        let room = (m.max_len as i64 - ids.len() as i64 - 1).max(0) as usize;
        ids.extend(&state_ids[..room.min(state_ids.len())]);
        ids.push(m.sep_id);
        ids.truncate(m.max_len);
        markers.retain(|&p| p < m.max_len);
        if markers.len() != q.options.len() {
            bail!("question {:?} options exceed head_max_len={}", q.id, m.head_max_len);
        }
        Ok((ids, markers))
    }

    /// Mirrors ONNXAgent.predict(): all questions go through the model in one batch.
    pub fn predict(&mut self, state: &str, questions: &[Question]) -> Result<Vec<Answer>> {
        let state_ids = self.encode(&state.replace(&self.meta.mask_token, " "))?;
        let items = questions
            .iter()
            .map(|q| self.build_sequence(&state_ids, q))
            .collect::<Result<Vec<_>>>()?;

        // collate_items(): pad to a common length
        let n = items.len();
        let mut seq_len = items.iter().map(|(ids, _)| ids.len()).max().unwrap_or(0);
        if let Some(pad_to) = self.pad_to {
            if seq_len > pad_to {
                bail!("input is {seq_len} tokens, longer than the model's static length {pad_to}");
            }
            seq_len = pad_to;
        }
        let kmax = items.iter().map(|(_, mk)| mk.len()).max().unwrap_or(0);
        let mut input_ids = Array2::<i64>::from_elem((n, seq_len), self.meta.pad_id);
        let mut attention = Array2::<i64>::zeros((n, seq_len));
        let mut marker_pos = Array2::<i64>::zeros((n, kmax));
        let mut marker_mask = Array2::<bool>::from_elem((n, kmax), false);
        let mut qtype = Array1::<i64>::zeros(n);
        for (r, ((ids, markers), q)) in items.iter().zip(questions).enumerate() {
            for (c, &id) in ids.iter().enumerate() {
                input_ids[[r, c]] = id;
                attention[[r, c]] = 1;
            }
            for (c, &p) in markers.iter().enumerate() {
                marker_pos[[r, c]] = p as i64;
                marker_mask[[r, c]] = true;
            }
            qtype[r] = q.qtype as i64;
        }

        let outputs = self.session.run(ort::inputs![
            "input_ids" => TensorRef::from_array_view(&input_ids)?,
            "attention_mask" => TensorRef::from_array_view(&attention)?,
            "marker_pos" => TensorRef::from_array_view(&marker_pos)?,
            "marker_mask" => TensorRef::from_array_view(&marker_mask)?,
            "qtype" => TensorRef::from_array_view(&qtype)?,
        ])?;
        let logits = outputs["logits"].try_extract_array::<f32>()?;

        let mut answers = Vec::with_capacity(n);
        for (r, q) in questions.iter().enumerate() {
            let k = q.options.len();
            let size = match k {
                0..=2 => "2",
                3..=5 => "3-5",
                6..=10 => "6-10",
                _ => "11+",
            };
            let bucket = format!("{}:{}", q.qtype.name(), size);
            let t = *self
                .meta
                .temperature_by_options
                .get(&bucket)
                .unwrap_or(&self.meta.temperature[q.qtype as usize]);
            let z: Vec<f32> = (0..k).map(|j| logits[[r, j]] / t).collect();
            let zmax = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = z.iter().map(|v| (v - zmax).exp()).collect();
            let sum: f32 = exp.iter().sum();
            answers.push(Answer {
                labels: q.labels.clone(),
                probs: exp.iter().map(|v| v / sum).collect(),
            });
        }
        Ok(answers)
    }
}
