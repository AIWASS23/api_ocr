//! Modelos de texto e imagem (CLIP, SigLIP) em ONNX: dois grafos (`text_model.onnx` e
//! `vision_model.onnx`, os exports do transformers.js) e o pré-processamento de imagem do
//! `preprocessor_config.json` (RGB, redimensionamento bicúbico, recorte central, escala e
//! normalização), como o `CLIPProcessor`/`SiglipProcessor` do Python.

use image::imageops::FilterType;
use image::{DynamicImage, RgbImage};
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::core::hub::ModelFiles;
use crate::core::onnx::{Feed, OnnxModel};
use crate::core::text_model::{load_tokenizer, pad_id};
use crate::embedder::ImageInput;
use crate::error::{EmbeddingError, Result};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Resize {
    None,
    ShortestEdge(u32),
    Exact(u32, u32),
}

/// Pré-processamento de imagem do Hugging Face.
#[derive(Clone, Debug)]
pub struct ImageProcessor {
    resize: Resize,
    filter: FilterType,
    crop: Option<(u32, u32)>,
    rescale: Option<f32>,
    mean: [f32; 3],
    std: [f32; 3],
    normalize: bool,
}

fn triple(v: Option<&Value>, default: [f32; 3]) -> [f32; 3] {
    match v {
        Some(Value::Array(a)) if a.len() == 3 => {
            let f = |i: usize| a[i].as_f64().map_or(default[i], |x| x as f32);
            [f(0), f(1), f(2)]
        }
        Some(Value::Number(n)) => [n.as_f64().unwrap_or(0.0) as f32; 3],
        _ => default,
    }
}

fn size(v: Option<&Value>) -> Option<Resize> {
    match v? {
        Value::Number(n) => n.as_u64().map(|s| Resize::ShortestEdge(s as u32)),
        Value::Object(m) => {
            if let Some(s) = m.get("shortest_edge").and_then(Value::as_u64) {
                return Some(Resize::ShortestEdge(s as u32));
            }
            let h = m.get("height").and_then(Value::as_u64)?;
            let w = m.get("width").and_then(Value::as_u64)?;
            Some(Resize::Exact(w as u32, h as u32))
        }
        _ => None,
    }
}

impl ImageProcessor {
    pub fn from_config(c: &Value) -> Result<Self> {
        let flag = |k: &str, d: bool| c.get(k).and_then(Value::as_bool).unwrap_or(d);
        let resize = if flag("do_resize", true) { size(c.get("size")).unwrap_or(Resize::ShortestEdge(224)) } else { Resize::None };
        // PIL: 0 nearest, 2 bilinear, 3 bicubic (padrão), 1 lanczos.
        let filter = match c.get("resample").and_then(Value::as_u64).unwrap_or(3) {
            0 => FilterType::Nearest,
            1 => FilterType::Lanczos3,
            2 => FilterType::Triangle,
            _ => FilterType::CatmullRom,
        };
        let crop = if flag("do_center_crop", false) {
            match size(c.get("crop_size")) {
                Some(Resize::Exact(w, h)) => Some((w, h)),
                Some(Resize::ShortestEdge(s)) => Some((s, s)),
                _ => Some((224, 224)),
            }
        } else {
            None
        };
        let rescale = flag("do_rescale", true).then(|| c.get("rescale_factor").and_then(Value::as_f64).unwrap_or(1.0 / 255.0) as f32);
        Ok(Self {
            resize,
            filter,
            crop,
            rescale,
            mean: triple(c.get("image_mean"), [0.481_454_66, 0.457_827_5, 0.408_210_73]),
            std: triple(c.get("image_std"), [0.268_629_54, 0.261_302_6, 0.275_777_1]),
            normalize: flag("do_normalize", true),
        })
    }

    /// `(altura, largura, pixels CHW)`.
    pub fn process(&self, image: &DynamicImage) -> (u32, u32, Vec<f32>) {
        let mut rgb: RgbImage = image.to_rgb8();
        match self.resize {
            Resize::None => {}
            Resize::Exact(w, h) => rgb = image::imageops::resize(&rgb, w, h, self.filter),
            Resize::ShortestEdge(s) => {
                let (w, h) = rgb.dimensions();
                let (nw, nh) = if w <= h { (s, (s as u64 * h as u64 / w.max(1) as u64) as u32) } else { ((s as u64 * w as u64 / h.max(1) as u64) as u32, s) };
                rgb = image::imageops::resize(&rgb, nw.max(1), nh.max(1), self.filter);
            }
        }
        if let Some((cw, ch)) = self.crop {
            let (w, h) = rgb.dimensions();
            let left = w.saturating_sub(cw) / 2;
            let top = h.saturating_sub(ch) / 2;
            rgb = image::imageops::crop_imm(&rgb, left, top, cw.min(w), ch.min(h)).to_image();
        }
        let (w, h) = rgb.dimensions();
        let mut out = vec![0f32; 3 * (w * h) as usize];
        let plane = (w * h) as usize;
        for (i, p) in rgb.pixels().enumerate() {
            for c in 0..3 {
                let mut v = p[c] as f32;
                if let Some(r) = self.rescale {
                    v *= r;
                }
                if self.normalize {
                    v = (v - self.mean[c]) / self.std[c];
                }
                out[c * plane + i] = v;
            }
        }
        (h, w, out)
    }
}

pub fn load_image(input: &ImageInput) -> Result<DynamicImage> {
    match input {
        ImageInput::Path(p) => image::open(p).map_err(|e| EmbeddingError::input(format!("image '{}': {e}", p.display()))),
        ImageInput::Bytes(b) => image::load_from_memory(b).map_err(|e| EmbeddingError::input(format!("image: {e}"))),
        ImageInput::Image(i) => Ok(i.clone()),
    }
}

/// Texto + imagem em dois grafos ONNX.
pub struct DualEncoder {
    tokenizer: Tokenizer,
    text: OnnxModel,
    vision: OnnxModel,
    processor: ImageProcessor,
    text_output: String,
    vision_output: String,
    max_length: usize,
    pad_id: u32,
    /// SigLIP: textos completados até `max_length` (como no treino).
    pad_to_max_length: bool,
}

impl DualEncoder {
    pub fn load(files: &ModelFiles, device: &str, default_max_length: usize, pad_to_max_length: bool) -> Result<Self> {
        let config = files.json("config.json", true)?;
        let max_length = config
            .as_ref()
            .and_then(|c| c.get("text_config"))
            .and_then(|t| t.get("max_position_embeddings"))
            .and_then(Value::as_u64)
            .map_or(default_max_length, |n| n as usize);
        let tokenizer_config = files.json("tokenizer_config.json", true)?;
        let (tokenizer, embedded_pad) = load_tokenizer(&files.require("tokenizer.json")?, max_length, false)?;
        let pad_id = pad_id(&tokenizer, embedded_pad, tokenizer_config.as_ref());
        let processor_config = files
            .json("preprocessor_config.json", true)?
            .ok_or_else(|| EmbeddingError::model(format!("'{}' has no preprocessor_config.json", files.name())))?;
        let text = OnnxModel::load(&files.onnx(&["onnx/text_model.onnx", "text_model.onnx"], None)?, device)?;
        let vision = OnnxModel::load(&files.onnx(&["onnx/vision_model.onnx", "vision_model.onnx"], None)?, device)?;
        let text_output = text.pick_output(&["text_embeds", "pooler_output"], 2)?;
        let vision_output = vision.pick_output(&["image_embeds", "pooler_output"], 2)?;
        Ok(Self {
            tokenizer,
            text,
            vision,
            processor: ImageProcessor::from_config(&processor_config)?,
            text_output,
            vision_output,
            max_length,
            pad_id,
            pad_to_max_length,
        })
    }

    fn rows(out: &crate::core::onnx::Output) -> Result<Vec<Vec<f32>>> {
        let [n, d] = out.shape.as_slice() else { return Err(EmbeddingError::model(format!("unexpected embeddings shape {:?}", out.shape))) };
        Ok(out.data.chunks(*d).take(*n).map(<[f32]>::to_vec).collect())
    }

    pub fn embed_texts(&self, texts: &[&str], batch_size: usize) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for batch in texts.chunks(batch_size.max(1)) {
            let encodings = self.tokenizer.encode_batch(batch.to_vec(), true).map_err(EmbeddingError::model)?;
            let longest = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(1);
            let seq = if self.pad_to_max_length { self.max_length } else { longest };
            let (mut ids, mut mask) = (Vec::new(), Vec::new());
            for e in &encodings {
                let n = e.get_ids().len();
                ids.extend(e.get_ids().iter().map(|&x| x as i64));
                ids.extend(std::iter::repeat_n(self.pad_id as i64, seq - n));
                mask.extend(std::iter::repeat_n(1i64, n));
                mask.extend(std::iter::repeat_n(0i64, seq - n));
            }
            let shape = vec![batch.len(), seq];
            let mut feeds: Vec<(&str, Feed)> = vec![("input_ids", Feed::Int(shape.clone(), ids))];
            if self.text.has_input("attention_mask") {
                feeds.push(("attention_mask", Feed::Int(shape, mask)));
            }
            let res = self.text.run(feeds, &[self.text_output.as_str()])?;
            out.extend(Self::rows(&res[0])?);
        }
        Ok(out)
    }

    pub fn embed_images(&self, images: &[ImageInput], batch_size: usize) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(images.len());
        for batch in images.chunks(batch_size.max(1)) {
            let mut pixels = Vec::new();
            let mut dims = None;
            for img in batch {
                let (h, w, data) = self.processor.process(&load_image(img)?);
                if dims.is_some_and(|d| d != (h, w)) {
                    return Err(EmbeddingError::model("images of different sizes after preprocessing"));
                }
                dims = Some((h, w));
                pixels.extend(data);
            }
            let (h, w) = dims.unwrap_or((0, 0));
            let feeds = vec![("pixel_values", Feed::Float(vec![batch.len(), 3, h as usize, w as usize], pixels))];
            let res = self.vision.run(feeds, &[self.vision_output.as_str()])?;
            out.extend(Self::rows(&res[0])?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn clip_and_siglip_preprocessing() {
        let clip = ImageProcessor::from_config(&json!({
            "do_resize": true, "size": {"shortest_edge": 4}, "resample": 3,
            "do_center_crop": true, "crop_size": {"height": 4, "width": 4},
            "do_rescale": true, "rescale_factor": 1.0/255.0,
            "do_normalize": true, "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]
        }))
        .unwrap();
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(8, 6, image::Rgb([255, 0, 128])));
        let (h, w, data) = clip.process(&img);
        assert_eq!((h, w, data.len()), (4, 4, 48));
        assert!((data[0] - 1.0).abs() < 1e-6 && (data[16] + 1.0).abs() < 1e-6, "R=1, G=-1");
        let siglip = ImageProcessor::from_config(&json!({"size": {"height": 3, "width": 5}, "image_mean": [0.5,0.5,0.5], "image_std": [0.5,0.5,0.5]})).unwrap();
        let (h, w, _) = siglip.process(&img);
        assert_eq!((h, w), (3, 5));
    }
}
