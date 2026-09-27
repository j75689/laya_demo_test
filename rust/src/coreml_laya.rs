//! Laya on a pre-compiled Core ML model, called through Apple's Core ML framework (objc2-core-ml).
//! No ONNX Runtime and no Python. The model lives in models/coreml/ (see python/fetch_coreml.py).
//!
//! Fixed-shape inputs, one question per prediction:
//!     input_ids      int32   [1, L]
//!     attention_mask int32   [1, L]
//!     marker_map     float32 [1, max_options, L]   one-hot [MASK] position per option
//!     question_type  float32 [1, 3]                one-hot choice / score / noul

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::AnyThread;
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel, MLModelConfiguration,
    MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSError, NSNumber, NSString, NSURL, ns_string};

use crate::laya::{Answer, Encoder, Predict, Question};

fn ns_err(e: Retained<NSError>) -> anyhow::Error {
    anyhow!("Core ML: {}", e.localizedDescription())
}

/// A zero-filled multi-array. Core ML leaves new arrays uninitialized.
fn zeros(shape: &[usize], dtype: MLMultiArrayDataType) -> Result<Retained<MLMultiArray>> {
    let dims: Vec<Retained<NSNumber>> = shape.iter().map(|&d| NSNumber::new_usize(d)).collect();
    let arr = unsafe {
        MLMultiArray::initWithShape_dataType_error(MLMultiArray::alloc(), &NSArray::from_retained_slice(&dims), dtype)
    }
    .map_err(ns_err)?;
    let bytes = shape.iter().product::<usize>() * 4; // int32 and float32 are both 4 bytes
    unsafe { std::ptr::write_bytes(data_ptr::<u8>(&arr), 0, bytes) };
    Ok(arr)
}

/// Arrays created with initWithShape use a contiguous first-major layout, so a flat pointer is valid.
#[allow(deprecated)]
unsafe fn data_ptr<T>(arr: &MLMultiArray) -> *mut T {
    unsafe { arr.dataPointer().as_ptr().cast() }
}

pub struct CoreMLLaya {
    model: Retained<MLModel>,
    enc: Encoder,
    len: usize,
    max_options: usize,
}

impl CoreMLLaya {
    pub fn load(dir: &Path, units: MLComputeUnits) -> Result<Self> {
        let enc = Encoder::load(&dir.join("tokenizer.json"), &dir.join("meta.json"))?;
        let len = enc.meta.max_len;
        let max_options = enc.meta.max_options.context("models/coreml/meta.json has no max_options")?;
        let path = dir
            .join("model.mlmodelc")
            .canonicalize()
            .context("models/coreml/model.mlmodelc not found; run python/fetch_coreml.py")?;
        let url = NSURL::fileURLWithPath_isDirectory(&NSString::from_str(&path.to_string_lossy()), true);
        let model = unsafe {
            let config = MLModelConfiguration::new();
            config.setComputeUnits(units);
            MLModel::modelWithContentsOfURL_configuration_error(&url, &config)
        }
        .map_err(ns_err)?;
        Ok(Self { model, enc, len, max_options })
    }

    fn run(&self, state_ids: &[i64], q: &Question) -> Result<Answer> {
        let (ids, markers) = self.enc.build_sequence(state_ids, q)?;
        if markers.len() > self.max_options {
            bail!("{} options, the model supports at most {}", markers.len(), self.max_options);
        }
        let l = self.len;
        let input_ids = zeros(&[1, l], MLMultiArrayDataType::Int32)?;
        let attention = zeros(&[1, l], MLMultiArrayDataType::Int32)?;
        let marker_map = zeros(&[1, self.max_options, l], MLMultiArrayDataType::Float32)?;
        let question_type = zeros(&[1, 3], MLMultiArrayDataType::Float32)?;
        unsafe {
            let id_buf = std::slice::from_raw_parts_mut(data_ptr::<i32>(&input_ids), l);
            let att_buf = std::slice::from_raw_parts_mut(data_ptr::<i32>(&attention), l);
            id_buf.fill(self.enc.meta.pad_id as i32);
            for (i, &id) in ids.iter().enumerate() {
                id_buf[i] = id as i32;
                att_buf[i] = 1;
            }
            let mm = data_ptr::<f32>(&marker_map);
            for (j, &p) in markers.iter().enumerate() {
                *mm.add(j * l + p) = 1.0;
            }
            *data_ptr::<f32>(&question_type).add(q.qtype as usize) = 1.0;
        }

        // Core ML returns autoreleased objects; drain them every call instead of once at exit
        autoreleasepool(|_| {
            let values: Vec<Retained<AnyObject>> = [&input_ids, &attention, &marker_map, &question_type]
                .into_iter()
                .map(|a| unsafe { MLFeatureValue::featureValueWithMultiArray(a) }.into_super().into_super())
                .collect();
            let names = [
                ns_string!("input_ids"),
                ns_string!("attention_mask"),
                ns_string!("marker_map"),
                ns_string!("question_type"),
            ];
            let dict = NSDictionary::from_retained_objects(&names, &values);
            let provider = unsafe {
                MLDictionaryFeatureProvider::initWithDictionary_error(MLDictionaryFeatureProvider::alloc(), &dict)
            }
            .map_err(ns_err)?;
            let out = unsafe { self.model.predictionFromFeatures_error(ProtocolObject::from_ref(&*provider)) }
                .map_err(ns_err)?;
            let logits = unsafe { out.featureValueForName(ns_string!("logits")) }
                .and_then(|v| unsafe { v.multiArrayValue() })
                .context("model output has no logits")?;
            let vals: Vec<f32> = (0..markers.len())
                .map(|j| unsafe { logits.objectAtIndexedSubscript(j as isize) }.floatValue())
                .collect();
            Ok(self.enc.answer(q, |j| vals[j]))
        })
    }
}

impl Predict for CoreMLLaya {
    fn predict(&mut self, state: &str, questions: &[Question]) -> Result<Vec<Answer>> {
        let state_ids = self.enc.encode_state(state)?;
        questions.iter().map(|q| self.run(&state_ids, q)).collect()
    }
}
