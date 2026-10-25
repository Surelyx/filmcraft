//! Safe inspection and limited conversion of MOGRT ZIP packages.
//!
//! MOGRT's embedded After Effects project is commonly a proprietary binary payload. This module
//! does not decode it. It converts supported static SVG and definition metadata; the engine may
//! additionally import PDF-compatible Illustrator assets and bind declared choice controls.
//! Native AEP transforms, raster footage placement, effects, and animation remain unsupported
//! unless separately reported as converted features.

use std::collections::HashSet;
use std::io::{Cursor, Read};

use filmcraft_geom::Vec2;
use filmcraft_time::{TICKS_PER_SECOND, Tick};
use serde::Serialize;
use serde_json::Value;
use zip::ZipArchive;

use crate::graphic::{new_shape_layer, new_text_layer};
use crate::gtemplate::{ControlKind, GraphicsTemplate, TEMPLATE_FORMAT, TEMPLATE_VERSION, TemplateControl};
use crate::keyframe::{Param, ParamValue};
use crate::{LayerExtra, graphic};

const MAX_PACKAGE_BYTES: usize = 64 * 1024 * 1024;
const MAX_ENTRY_BYTES: u64 = 32 * 1024 * 1024;
const MAX_TOTAL_UNCOMPRESSED: u64 = 128 * 1024 * 1024;
const MAX_ENTRIES: usize = 4096;
const MAX_LAYERS: usize = 4096;

fn add_entry_size(size: u64, total: &mut u64) -> Result<(), MogrtError> {
    if size > MAX_ENTRY_BYTES {
        return Err(MogrtError::DecompressedTooLarge);
    }
    *total = total.checked_add(size).ok_or(MogrtError::DecompressedTooLarge)?;
    if *total > MAX_TOTAL_UNCOMPRESSED {
        return Err(MogrtError::DecompressedTooLarge);
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CompatibilityReport {
    pub name: String,
    pub author_app: Option<String>,
    pub description: Option<String>,
    pub license_id: Option<String>,
    pub canvas: Option<[u32; 2]>,
    pub duration_ticks: Option<i64>,
    pub frame_rate_fps: Option<f64>,
    pub layer_count: Option<usize>,
    pub converted_layer_count: Option<usize>,
    pub editable_text: Vec<String>,
    pub controls: Vec<DetectedControl>,
    pub manifests: Vec<String>,
    pub project_data: Vec<String>,
    pub assets: Vec<String>,
    pub supported_features: Vec<String>,
    pub unsupported_features: Vec<String>,
    pub convertible: bool,
    pub partial: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DetectedControl {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub value: Value,
    pub choices: Vec<String>,
    pub can_animate: bool,
    pub font_family: Option<String>,
    pub font_style: Option<String>,
    pub font_size: Option<f64>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Conversion {
    pub template: GraphicsTemplate,
    pub report: CompatibilityReport,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MogrtError {
    #[error("MOGRT package is too large")]
    PackageTooLarge,
    #[error("invalid MOGRT ZIP package: {0}")]
    InvalidArchive(String),
    #[error("MOGRT package has too many entries")]
    TooManyEntries,
    #[error("MOGRT package exceeds the decompressed size limit")]
    DecompressedTooLarge,
    #[error("unsafe path in MOGRT package: {0}")]
    UnsafePath(String),
    #[error("invalid MOGRT package: {0}")]
    InvalidPackage(String),
    #[error("unsupported MOGRT project: {0}")]
    Unsupported(String),
}

#[derive(Clone, Debug)]
struct EntryInfo {
    name: String,
    index: usize,
    size: u64,
    is_dir: bool,
}

struct Package<'a> {
    archive: ZipArchive<Cursor<&'a [u8]>>,
    entries: Vec<EntryInfo>,
}

impl<'a> Package<'a> {
    fn open(bytes: &'a [u8]) -> Result<Self, MogrtError> {
        if bytes.len() > MAX_PACKAGE_BYTES {
            return Err(MogrtError::PackageTooLarge);
        }
        let mut archive = ZipArchive::new(Cursor::new(bytes)).map_err(|e| MogrtError::InvalidArchive(e.to_string()))?;
        if archive.len() > MAX_ENTRIES {
            return Err(MogrtError::TooManyEntries);
        }
        let mut entries = Vec::with_capacity(archive.len());
        let mut seen = HashSet::new();
        let mut total = 0u64;
        for index in 0..archive.len() {
            let file = archive.by_index(index).map_err(|e| MogrtError::InvalidArchive(e.to_string()))?;
            let name = file.name().to_string();
            validate_entry_path(&name)?;
            if !seen.insert(name.clone()) {
                return Err(MogrtError::InvalidPackage(format!("duplicate entry `{name}`")));
            }
            add_entry_size(file.size(), &mut total)?;
            entries.push(EntryInfo { name, index, size: file.size(), is_dir: file.is_dir() });
        }
        Ok(Self { archive, entries })
    }

    fn read(&mut self, entry: &EntryInfo) -> Result<Vec<u8>, MogrtError> {
        if entry.size > MAX_ENTRY_BYTES {
            return Err(MogrtError::DecompressedTooLarge);
        }
        let mut file = self.archive.by_index(entry.index).map_err(|e| MogrtError::InvalidArchive(e.to_string()))?;
        let limit = MAX_ENTRY_BYTES.saturating_add(1);
        let mut bytes = Vec::with_capacity(entry.size.min(MAX_ENTRY_BYTES) as usize);
        file.by_ref().take(limit).read_to_end(&mut bytes).map_err(|e| MogrtError::InvalidArchive(e.to_string()))?;
        if bytes.len() as u64 > MAX_ENTRY_BYTES || bytes.len() as u64 != entry.size {
            return Err(MogrtError::DecompressedTooLarge);
        }
        Ok(bytes)
    }
}

fn validate_entry_path(name: &str) -> Result<(), MogrtError> {
    let unsafe_path = || MogrtError::UnsafePath(name.to_string());
    if name.is_empty() || name.starts_with('/') || name.starts_with('\\') || name.contains('\\') || name.contains('\0') {
        return Err(unsafe_path());
    }
    let mut parts = name.split('/');
    if parts.next().is_some_and(|p| p.len() >= 2 && p.as_bytes().get(1) == Some(&b':')) {
        return Err(unsafe_path());
    }
    if name.split('/').any(|part| part == ".." || part == ".") {
        return Err(unsafe_path());
    }
    Ok(())
}

/// Inspect package entries and supported/unsupported source features without creating a template.
pub fn inspect(bytes: &[u8]) -> Result<CompatibilityReport, MogrtError> {
    let mut package = Package::open(bytes)?;
    report_package(&mut package).map(|(report, _)| report)
}

/// Read standalone PDF-compatible Illustrator assets from a MOGRT package and its nested
/// `project.aegraphic` archive. Each entry is returned as `(archive path, bytes)` after the same
/// path and decompression checks as inspection. The AE composition itself is not decoded.
pub fn illustrator_assets(bytes: &[u8]) -> Result<Vec<(String, Vec<u8>)>, MogrtError> {
    let mut package = Package::open(bytes)?;
    let mut assets = Vec::new();
    let mut total = 0u64;
    for entry in package.entries.clone() {
        if entry.is_dir {
            continue;
        }
        let lower = entry.name.to_ascii_lowercase();
        if lower.ends_with(".ai") {
            add_entry_size(entry.size, &mut total)?;
            assets.push((entry.name.clone(), package.read(&entry)?));
        } else if lower.rsplit('/').next() == Some("project.aegraphic") {
            let nested_bytes = package.read(&entry)?;
            let mut nested = Package::open(&nested_bytes).map_err(|error| MogrtError::InvalidPackage(format!("invalid nested {}: {error}", entry.name)))?;
            for child in nested.entries.clone() {
                if child.is_dir || !child.name.to_ascii_lowercase().ends_with(".ai") {
                    continue;
                }
                add_entry_size(child.size, &mut total)?;
                assets.push((format!("{}/{}", entry.name, child.name), nested.read(&child)?));
            }
        }
        if assets.len() > 128 {
            return Err(MogrtError::TooManyEntries);
        }
    }
    Ok(assets)
}

/// Convert the package's static `project.svg` into FilmCraft's existing `.fcgt` model.
pub fn convert(bytes: &[u8]) -> Result<Conversion, MogrtError> {
    let mut package = Package::open(bytes)?;
    let (mut report, svg) = report_package(&mut package)?;
    let Some(svg) = svg else {
        return convert_definition_subset(report);
    };
    if !report.convertible {
        return Err(MogrtError::Unsupported(report.unsupported_features.join("; ")));
    }
    let (mut template, supported, unsupported) = convert_svg(&svg)?;
    report.supported_features.extend(supported);
    report.unsupported_features.extend(unsupported);
    report.supported_features.sort();
    report.supported_features.dedup();
    report.unsupported_features.sort();
    report.unsupported_features.dedup();
    report.convertible = !template.layers.is_empty();
    report.partial = !report.unsupported_features.is_empty();
    template.name = report.name.clone();
    template.id = format!("user:{}", crate::gtemplate::slug(&template.name));
    template.validate().map_err(|e| MogrtError::InvalidPackage(e.to_string()))?;
    Ok(Conversion { template, report })
}

fn report_package(package: &mut Package<'_>) -> Result<(CompatibilityReport, Option<Vec<u8>>), MogrtError> {
    let mut manifests = Vec::new();
    let mut project_data = Vec::new();
    let mut assets = Vec::new();
    let mut svg_entry = None;
    let mut name = String::from("Imported Motion Graphic");
    let mut canvas = None;
    let mut duration_ticks = None;
    let mut frame_rate_fps = None;
    let mut layer_count = None;
    let mut converted_layer_count = None;
    let mut editable_text = Vec::new();
    let mut controls = Vec::new();
    let mut author_app = None;
    let mut description = None;
    let mut license_id = None;
    let mut unsupported_features = Vec::new();
    let mut has_opaque_project = false;

    for entry in package.entries.clone() {
        if entry.is_dir {
            continue;
        }
        let lower = entry.name.to_ascii_lowercase();
        let base = lower.rsplit('/').next().unwrap_or(&lower);
        if matches!(base, "manifest.json" | "manifest.xml" | "metadata.json" | "definition.json") {
            manifests.push(entry.name.clone());
            let data = package.read(&entry)?;
            if let Ok(value) = serde_json::from_slice::<Value>(&data) {
                name = metadata_name(&value).unwrap_or(name);
                if base == "definition.json" {
                    author_app = value.get("authorApp").and_then(Value::as_str).map(str::to_string);
                    description = value.get("description").and_then(Value::as_str).map(str::to_string);
                    license_id = value.get("licenseID").and_then(Value::as_str).map(str::to_string);
                    canvas = definition_canvas(&value);
                    duration_ticks = definition_duration_ticks(&value);
                    frame_rate_fps = definition_frame_rate(&value);
                    controls = definition_controls(&value);
                    editable_text =
                        controls.iter().filter(|control| control.kind == "text").filter_map(|control| control.value.as_str().map(str::to_string)).collect();
                    let effects = value.get("usedEffects").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str);
                    unsupported_features.extend(effects.map(|effect| format!("After Effects effect `{effect}` is not converted")));
                    if value.get("usedCompRenderers").and_then(Value::as_array).is_some_and(|renderers| !renderers.is_empty()) {
                        unsupported_features.push("After Effects composition renderer settings are not converted".into());
                    }
                    if value.get("usedFontsLocalized").is_some() {
                        unsupported_features.push("declared fonts are reported but not embedded or substituted".into());
                    }
                    if controls.iter().any(|control| control.can_animate) {
                        unsupported_features.push("control animation capability is declared; animated values are not inspected or converted".into());
                    }
                }
            } else {
                unsupported_features.push(format!("manifest `{}` is not valid JSON; metadata was not read", entry.name));
            }
        } else if base == "project.aegraphic" {
            project_data.push(entry.name.clone());
            has_opaque_project = true;
            unsupported_features.push("opaque After Effects graphic project; layer structure, expressions, effects, and animation are not decoded".into());
            let nested_bytes = package.read(&entry)?;
            let mut nested = Package::open(&nested_bytes).map_err(|error| MogrtError::InvalidPackage(format!("invalid nested project.aegraphic: {error}")))?;
            inventory_nested_project(&entry.name, &mut nested, &mut project_data, &mut assets, &mut unsupported_features)?;
        } else if matches!(base, "project.mogrt" | "project.aep" | "project.aepx") || base.ends_with(".aep") || base.ends_with(".aepx") {
            project_data.push(entry.name.clone());
            has_opaque_project = true;
            unsupported_features.push(format!(
                "opaque After Effects project payload `{}`; expressions, effects, and animation inside it cannot be inspected or converted",
                entry.name
            ));
        } else if base == "project.svg" {
            project_data.push(entry.name.clone());
            if svg_entry.replace(entry.clone()).is_some() {
                return Err(MogrtError::InvalidPackage("multiple `project.svg` source documents are ambiguous".into()));
            }
        } else {
            assets.push(entry.name.clone());
        }
    }
    if manifests.is_empty() && project_data.is_empty() {
        return Err(MogrtError::InvalidPackage("no MOGRT manifest or project data was found".into()));
    }
    let svg = svg_entry.map(|entry| package.read(&entry)).transpose()?;
    let mut convertible = false;
    let mut partial = false;
    let mut supported_features = vec!["ZIP entries validated; decompressed data is size-limited".into()];
    if !manifests.is_empty() {
        supported_features.push("package manifest metadata detected".into());
    }
    if !assets.is_empty() {
        supported_features.push(format!("{} embedded asset(s) inventoried", assets.len()));
    }
    if assets.iter().any(|asset| {
        let lower = asset.to_ascii_lowercase();
        lower.starts_with("project.aegraphic/") && [".png", ".jpg", ".jpeg", ".tif", ".tiff"].iter().any(|extension| lower.ends_with(extension))
    }) {
        unsupported_features
            .push("embedded raster footage was inventoried but not placed; the opaque native AEP does not expose its composition transforms or timing".into());
    }
    if svg.is_none() {
        unsupported_features.push("no standalone `project.svg` source document; native After Effects project decoding is unavailable".into());
        if !controls.is_empty() && canvas.is_some() {
            convertible = true;
            partial = true;
            supported_features.push("definition.json control defaults can be represented as editable graphic layers".into());
            unsupported_features.push("composition artwork, source layer positions, shapes, and visual styling are not available in definition.json".into());
            unsupported_features.push("the MOGRT dropdown's choice list cannot be represented by FilmCraft template controls".into());
            unsupported_features.push("FilmCraft graphics templates do not store a source frame-rate field".into());
        }
    }
    if let Some(data) = svg.as_deref() {
        match convert_svg(data) {
            Ok((template, supported, unsupported)) => {
                convertible = !template.layers.is_empty() && !has_opaque_project;
                partial = !unsupported.is_empty() || has_opaque_project;
                canvas = Some(template.canvas);
                layer_count = Some(template.layers.len());
                converted_layer_count = Some(template.layers.len());
                editable_text = template
                    .layers
                    .iter()
                    .filter(|layer| layer.effect == graphic::TEXT_LAYER)
                    .filter_map(|layer| match layer.params.get("text").map(|param| &param.value) {
                        Some(ParamValue::Text(text)) => Some(text.clone()),
                        _ => None,
                    })
                    .collect();
                supported_features.extend(supported);
                unsupported_features.extend(unsupported);
                if template.layers.is_empty() {
                    unsupported_features.push("project.svg contains no supported static layers".into());
                } else if has_opaque_project {
                    unsupported_features.push("standalone SVG data is not assumed to represent the opaque After Effects project".into());
                }
            }
            Err(error) => unsupported_features.push(error.to_string()),
        }
    }
    supported_features.sort();
    supported_features.dedup();
    unsupported_features.sort();
    unsupported_features.dedup();
    Ok((
        CompatibilityReport {
            name,
            author_app,
            description,
            license_id,
            canvas,
            duration_ticks,
            frame_rate_fps,
            layer_count,
            converted_layer_count,
            editable_text,
            controls,
            manifests,
            project_data,
            assets,
            supported_features,
            unsupported_features,
            convertible,
            partial,
        },
        svg,
    ))
}

fn convert_definition_subset(mut report: CompatibilityReport) -> Result<Conversion, MogrtError> {
    let canvas = report.canvas.ok_or_else(|| MogrtError::Unsupported("definition.json does not provide a supported canvas size".into()))?;
    if report.controls.is_empty() {
        return Err(MogrtError::Unsupported("definition.json has no supported client controls".into()));
    }
    let mut template = GraphicsTemplate {
        name: report.name.clone(),
        id: format!("user:{}", crate::gtemplate::slug(&report.name)),
        canvas,
        duration: Tick(report.duration_ticks.unwrap_or(5 * TICKS_PER_SECOND)),
        description: "Partial MOGRT conversion. Editable definition controls use an approximate layout measured from the bundled preview; native artwork and animation are omitted.".into(),
        ..Default::default()
    };
    let preview_scale = (canvas[0] as f64 / 640.0).min(canvas[1] as f64 / 360.0);
    let preview_offset = Vec2::new((canvas[0] as f64 - 640.0 * preview_scale) / 2.0, (canvas[1] as f64 - 360.0 * preview_scale) / 2.0);
    let preview_position = |x: f64, y: f64| Vec2::new(preview_offset.x + x * preview_scale, preview_offset.y + y * preview_scale);
    let preview_size = |width: f64, height: f64| Vec2::new(width * preview_scale, height * preview_scale);
    if report.duration_ticks.is_none() {
        report.unsupported_features.push("source duration was missing or invalid; a five-second default was used".into());
    }

    let mut card = new_shape_layer(0, preview_position(235.0, 232.5), preview_size(147.0, 68.0), vec![]);
    set_param(&mut card, "name", ParamValue::Text("Approximate control panel".into()));
    set_param(&mut card, "fill", ParamValue::Bool(true));
    set_param(&mut card, "fill_color", ParamValue::Color([1.0, 1.0, 1.0, 1.0]));
    set_param(&mut card, "corner_radius", ParamValue::Float(11.0 * preview_scale));
    append_definition_layer(&mut template, card);

    let mut divider = new_shape_layer(0, preview_position(243.0, 215.5), preview_size(2.0, 11.0), vec![]);
    set_param(&mut divider, "name", ParamValue::Text("Approximate text divider".into()));
    set_param(&mut divider, "fill", ParamValue::Bool(true));
    set_param(&mut divider, "fill_color", ParamValue::Color([0.18, 0.18, 0.18, 1.0]));
    append_definition_layer(&mut template, divider);

    for control in report.controls.iter().filter(|control| control.kind == "text") {
        let value = control.value.as_str().unwrap_or_default();
        let (x, y) = match control.name.as_str() {
            "Color Number" => (202.0, 218.0),
            "Color Name" => (250.0, 218.0),
            "Color HEX Code" => (202.0, 239.0),
            "Prismacolor Premier" => (202.0, 259.0),
            _ => (202.0, 239.0),
        };
        let mut layer = new_text_layer(value, preview_position(x, y), control.font_size.unwrap_or(36.0));
        set_param(&mut layer, "name", ParamValue::Text(control.name.clone()));
        set_param(&mut layer, "fill_color", ParamValue::Color([0.12, 0.12, 0.12, 1.0]));
        if let Some(font) = &control.font_family {
            set_param(&mut layer, "font", ParamValue::Text(font.clone()));
        }
        if let Some(style) = &control.font_style {
            set_param(&mut layer, "font_style", ParamValue::Text(style.clone()));
        }
        let uid = append_definition_layer(&mut template, layer);
        add_definition_control(&mut template, control, uid, ControlKind::Text, "text");
    }
    for control in report.controls.iter().filter(|control| control.kind == "choice") {
        report.unsupported_features.push(format!(
            "choice control `{}` (current value {}) was not rendered as text because its six-option selection is dynamic",
            control.name, control.value
        ));
    }

    for control in report.controls.iter().filter(|control| control.kind == "color") {
        let Some(values) = control.value.as_array() else {
            report.unsupported_features.push(format!("color control `{}` has an unsupported value", control.name));
            continue;
        };
        let mut rgba = [0.0f32; 4];
        if values.len() != 4 {
            report.unsupported_features.push(format!("color control `{}` does not contain RGBA values", control.name));
            continue;
        }
        let mut valid = true;
        for (index, value) in values.iter().enumerate() {
            let Some(channel) = value.as_f64().filter(|value| value.is_finite() && (0.0..=1.0).contains(value)) else {
                valid = false;
                break;
            };
            rgba[index] = channel as f32;
        }
        if !valid {
            report.unsupported_features.push(format!("color control `{}` contains an out-of-range channel", control.name));
            continue;
        }
        let mut layer = new_shape_layer(1, preview_position(185.0, 233.0), preview_size(18.0, 18.0), vec![]);
        set_param(&mut layer, "name", ParamValue::Text(control.name.clone()));
        set_param(&mut layer, "fill_color", ParamValue::Color(rgba));
        let uid = append_definition_layer(&mut template, layer);
        add_definition_control(&mut template, control, uid, ControlKind::Color, "fill_color");
    }
    for control in report.controls.iter().filter(|control| control.kind == "other") {
        report.unsupported_features.push(format!("control `{}` has unsupported source type `{}`", control.name, control.kind));
    }
    if report.frame_rate_fps.is_some() {
        report.supported_features.push("source duration and frame rate metadata detected".into());
    }
    report.supported_features.push(format!("{} editable control-derived layers placed in an approximate preview-scaled layout", template.layers.len()));
    report.unsupported_features.push("native AEP artwork, layer geometry, placement, and keyframes were not decoded; output is not visually equivalent".into());
    report
        .unsupported_features
        .push("the bundled 640x360 preview was used only for approximate control-panel placement; it does not recover source layer coordinates".into());
    if report.assets.iter().any(|asset| asset.to_ascii_lowercase().ends_with(".ai")) {
        report.unsupported_features.push("butterfly Illustrator/PDF footage assets are present but omitted: FCGT has no image/vector-footage layer, and the native animated AEP composition is opaque".into());
    }
    report.converted_layer_count = Some(template.layers.len());
    report.convertible = !template.layers.is_empty();
    report.partial = report.convertible;
    report.supported_features.sort();
    report.supported_features.dedup();
    report.unsupported_features.sort();
    report.unsupported_features.dedup();
    template.validate().map_err(|error| MogrtError::InvalidPackage(error.to_string()))?;
    Ok(Conversion { template, report })
}

fn append_definition_layer(template: &mut GraphicsTemplate, mut layer: crate::EffectInstance) -> u64 {
    let uid = template.layers.len() as u64 + 1;
    layer.layer = Some(Box::new(LayerExtra { uid, ..Default::default() }));
    template.layers.push(layer);
    uid
}

fn add_definition_control(template: &mut GraphicsTemplate, control: &DetectedControl, uid: u64, kind: ControlKind, param: &str) {
    template.controls.push(TemplateControl {
        id: format!("mogrt-{}", crate::gtemplate::slug(&control.id)),
        name: control.name.clone(),
        kind,
        layer: uid,
        param: param.into(),
        min: None,
        max: None,
        choices: Vec::new(),
        choice_layers: Vec::new(),
    });
}

fn metadata_name(value: &serde_json::Value) -> Option<String> {
    let localized = value.get("capsuleNameLocalized").and_then(localized_text);
    localized
        .or_else(|| value.get("capsuleName").and_then(Value::as_str))
        .or_else(|| ["name", "title", "displayName"].into_iter().find_map(|key| value.get(key).and_then(Value::as_str)))
        .map(str::trim)
        .filter(|name| !name.is_empty() && name.len() <= 256)
        .map(str::to_string)
}

fn localized_text(value: &Value) -> Option<&str> {
    let entries = value.get("strDB")?.as_array()?;
    entries
        .iter()
        .find(|entry| entry.get("localeString").and_then(Value::as_str) == Some("en_US"))
        .or_else(|| entries.first())
        .and_then(|entry| entry.get("str"))
        .and_then(Value::as_str)
}

fn definition_canvas(value: &Value) -> Option<[u32; 2]> {
    let localized = definition_source_info(value)?;
    let size = localized.get("framesize")?.get("size")?;
    let dimension = |key: &str| {
        let value = size.get(key)?.as_u64()?;
        (value > 0 && value <= 16384).then_some(value as u32)
    };
    Some([dimension("x")?, dimension("y")?])
}

fn definition_source_info(value: &Value) -> Option<&Value> {
    let localized = value.get("sourceInfoLocalized")?.as_object()?;
    localized.get("en_US").or_else(|| localized.values().next())
}

fn definition_duration_ticks(value: &Value) -> Option<i64> {
    let duration = definition_source_info(value)?.get("duration")?;
    let value = duration.get("value")?.as_i64()?;
    let scale = duration.get("scale")?.as_i64()?;
    if value <= 0 || scale <= 0 {
        return None;
    }
    let ticks = value.checked_mul(TICKS_PER_SECOND)?.checked_div(scale)?;
    (ticks <= TICKS_PER_SECOND.checked_mul(24 * 60 * 60)?).then_some(ticks)
}

fn definition_frame_rate(value: &Value) -> Option<f64> {
    let ticks_per_frame = definition_source_info(value)?.get("framerate")?.get("ticksperframe")?.as_f64()?;
    if !ticks_per_frame.is_finite() || ticks_per_frame <= 0.0 {
        return None;
    }
    let fps = TICKS_PER_SECOND as f64 / ticks_per_frame;
    (fps.is_finite() && (1.0..=240.0).contains(&fps)).then_some(fps)
}

fn definition_controls(value: &Value) -> Vec<DetectedControl> {
    let Some(raw_controls) = value.get("clientControls").and_then(Value::as_array) else { return Vec::new() };
    raw_controls
        .iter()
        .take(MAX_LAYERS)
        .filter_map(|raw| {
            let id = raw.get("id")?.as_str()?.to_string();
            let name = raw.get("uiName").and_then(localized_text).unwrap_or("Unnamed control").to_string();
            let type_id = raw.get("type").and_then(Value::as_u64).unwrap_or(u64::MAX);
            let raw_value = raw.get("value").cloned().unwrap_or(Value::Null);
            let value = localized_text(&raw_value).map(|text| Value::String(text.to_string())).unwrap_or(raw_value);
            let kind = match type_id {
                6 => "text",
                4 => "color",
                13 => "choice",
                _ => "other",
            }
            .to_string();
            let choices = raw.get("menucontent").and_then(Value::as_array).into_iter().flatten().filter_map(localized_text).map(str::to_string).collect();
            let font = raw.get("fonteditinfo");
            let font_value = font.and_then(|font| font.get("fontEditValue"));
            let font_value = font_value.and_then(Value::as_str).or_else(|| font_value.and_then(Value::as_array).and_then(|values| values.first()?.as_str()));
            let (font_family, font_style) = font_value
                .map(|value| {
                    value.rsplit_once('-').map_or((value.to_string(), "Regular".to_string()), |(family, style)| (family.to_string(), style.to_string()))
                })
                .map_or((None, None), |(family, style)| (Some(family), Some(style)));
            let font_size =
                font.and_then(|font| font.get("fontSizeEditValue")).and_then(Value::as_f64).filter(|size| size.is_finite() && (1.0..=2000.0).contains(size));
            Some(DetectedControl {
                id,
                name,
                kind,
                value,
                choices,
                can_animate: raw.get("canAnimate").and_then(Value::as_bool).unwrap_or(false),
                font_family,
                font_style,
                font_size,
            })
        })
        .collect()
}

fn inventory_nested_project(
    prefix: &str,
    nested: &mut Package<'_>,
    project_data: &mut Vec<String>,
    assets: &mut Vec<String>,
    unsupported_features: &mut Vec<String>,
) -> Result<(), MogrtError> {
    for entry in nested.entries.clone() {
        if entry.is_dir {
            continue;
        }
        let full_name = format!("{prefix}/{}", entry.name);
        let base = entry.name.rsplit('/').next().unwrap_or(&entry.name).to_ascii_lowercase();
        if base.ends_with(".aep") || base.ends_with(".aepx") {
            project_data.push(full_name.clone());
            unsupported_features.push(format!("opaque native After Effects project `{full_name}`; layer count and timeline are unknown"));
            let contents = nested.read(&entry)?;
            if contents.starts_with(b"PK\x03\x04") || contents.starts_with(b"PK\x05\x06") {
                match Package::open(&contents) {
                    Ok(aep) => {
                        for child in aep.entries.iter().filter(|child| !child.is_dir) {
                            let child_name = format!("{full_name}/{}", child.name);
                            if child.name.ends_with(".aep") || child.name.ends_with(".aepx") {
                                project_data.push(child_name);
                            } else {
                                assets.push(child_name);
                            }
                        }
                    }
                    Err(error) => unsupported_features.push(format!("native project container `{full_name}` could not be inventoried: {error}")),
                }
            }
        } else {
            assets.push(full_name);
        }
    }
    Ok(())
}

fn convert_svg(bytes: &[u8]) -> Result<(GraphicsTemplate, Vec<String>, Vec<String>), MogrtError> {
    let text = std::str::from_utf8(bytes).map_err(|e| MogrtError::Unsupported(format!("project.svg is not UTF-8: {e}")))?;
    if text.contains("<!DOCTYPE") || text.contains("<!ENTITY") {
        return Err(MogrtError::Unsupported("SVG DTD/entity declarations are not supported".into()));
    }
    let document = roxmltree::Document::parse(text).map_err(|e| MogrtError::InvalidPackage(format!("invalid project.svg: {e}")))?;
    let root = document.root_element();
    if root.tag_name().name() != "svg" {
        return Err(MogrtError::Unsupported("project.svg root is not an SVG document".into()));
    }
    let root_attributes = ["id", "width", "height", "viewBox", "version", "xmlns", "preserveAspectRatio"];
    if let Some(attribute) = root.attributes().find(|attribute| !root_attributes.contains(&attribute.name())) {
        return Err(MogrtError::Unsupported(format!("unsupported SVG root attribute `{}`", attribute.name())));
    }
    if root.attribute("preserveAspectRatio").is_some_and(|value| !matches!(value, "xMidYMid meet" | "xMidYMid")) {
        return Err(MogrtError::Unsupported("non-default SVG preserveAspectRatio".into()));
    }
    if root.attributes().any(|attr| matches!(attr.name(), "transform" | "style" | "class" | "fill" | "stroke" | "opacity")) {
        return Err(MogrtError::Unsupported("inherited styling or transform on the SVG root".into()));
    }
    let canvas = svg_canvas(root)?;
    let mut template = GraphicsTemplate { canvas, duration: Tick(5 * TICKS_PER_SECOND), ..Default::default() };
    let mut supported = Vec::new();
    let mut unsupported = Vec::new();
    let mut uid = 1u64;
    let mut count = 0usize;

    for node in root.descendants().filter(roxmltree::Node::is_element) {
        if node == root {
            continue;
        }
        let tag = node.tag_name().name();
        if node.ancestors().any(|ancestor| {
            matches!(
                ancestor.tag_name().name(),
                "defs" | "filter" | "linearGradient" | "radialGradient" | "mask" | "clipPath" | "pattern" | "marker" | "symbol"
            )
        }) {
            if matches!(tag, "text" | "rect" | "circle" | "ellipse" | "path" | "polygon" | "polyline" | "line") {
                unsupported.push(format!("non-rendered SVG definition content `{tag}`"));
            }
            continue;
        }
        if matches!(tag, "animate" | "animateTransform" | "animateMotion" | "set") {
            unsupported.push(format!("SVG animation element `{tag}`"));
            continue;
        }
        if matches!(tag, "script" | "foreignObject") {
            unsupported.push(format!("SVG script/expression or foreign content `{tag}`"));
            continue;
        }
        if matches!(tag, "filter" | "linearGradient" | "radialGradient" | "mask" | "clipPath" | "pattern" | "style") {
            unsupported.push(format!("SVG effect or paint element `{tag}`"));
            continue;
        }
        if matches!(tag, "path" | "polygon" | "polyline" | "line" | "use" | "image") {
            unsupported.push(format!("SVG geometry element `{tag}`"));
            continue;
        }
        if !matches!(tag, "text" | "rect" | "circle" | "ellipse") {
            if !matches!(tag, "g" | "defs" | "title" | "desc" | "metadata") {
                unsupported.push(format!("unsupported SVG element `{tag}`"));
            }
            continue;
        }
        let known_attributes: &[&str] = match tag {
            "text" => &[
                "id",
                "x",
                "y",
                "font-size",
                "font-family",
                "font-weight",
                "font-style",
                "fill",
                "fill-opacity",
                "stroke",
                "stroke-opacity",
                "stroke-width",
                "text-anchor",
                "opacity",
            ],
            "rect" => &["id", "x", "y", "width", "height", "rx", "ry", "fill", "fill-opacity", "stroke", "stroke-opacity", "stroke-width", "opacity"],
            "circle" => &["id", "cx", "cy", "r", "fill", "fill-opacity", "stroke", "stroke-opacity", "stroke-width", "opacity"],
            "ellipse" => &["id", "cx", "cy", "rx", "ry", "fill", "fill-opacity", "stroke", "stroke-opacity", "stroke-width", "opacity"],
            _ => &[],
        };
        let unknown_attribute = node.attributes().find(|attribute| !known_attributes.contains(&attribute.name()));
        if let Some(attribute) = unknown_attribute {
            unsupported.push(format!("unsupported SVG `{tag}` attribute `{}`", attribute.name()));
            continue;
        }
        if node.attribute("transform").is_some() || node.attribute("style").is_some() || node.attribute("class").is_some() {
            unsupported.push(format!("SVG styling or transform on `{tag}`"));
            continue;
        }
        if node.ancestors().any(|ancestor| {
            ancestor != root
                && ancestor.tag_name().name() == "g"
                && (ancestor.attributes().any(|attr| attr.name() != "id") || ancestor.attribute("transform").is_some())
        }) {
            unsupported.push(format!("SVG group styling or transform containing `{tag}"));
            continue;
        }
        let layer = match tag {
            "text" => svg_text_layer(node)?,
            "rect" => svg_rect_layer(node)?,
            "circle" => svg_circle_layer(node)?,
            "ellipse" => svg_ellipse_layer(node)?,
            _ => continue,
        };
        if let Some(mut layer) = layer {
            count += 1;
            if count > MAX_LAYERS {
                return Err(MogrtError::InvalidPackage("SVG contains too many graphic layers".into()));
            }
            layer.layer = Some(Box::new(LayerExtra { uid, ..Default::default() }));
            if layer.effect == graphic::TEXT_LAYER {
                if !supported.iter().any(|feature| feature == "static SVG text") {
                    supported.push("static SVG text".into());
                }
                let text = match layer.params.get("text").map(|p| &p.value) {
                    Some(ParamValue::Text(text)) => text.as_str(),
                    _ => "",
                };
                let label: String = text.chars().take(40).collect();
                template.controls.push(TemplateControl {
                    id: format!("text-{uid}"),
                    name: if label.is_empty() { format!("Text {uid}") } else { label },
                    kind: ControlKind::Text,
                    layer: uid,
                    param: "text".into(),
                    min: None,
                    max: None,
                    choices: Vec::new(),
                    choice_layers: Vec::new(),
                });
            }
            if layer.effect == graphic::SHAPE_LAYER && !supported.iter().any(|feature| feature == "static SVG rectangles and ellipses") {
                supported.push("static SVG rectangles and ellipses".into());
            }
            if node.attribute("stroke").is_some_and(|stroke| stroke != "none") && !supported.iter().any(|feature| feature == "static SVG solid strokes") {
                supported.push("static SVG solid strokes".into());
            }
            template.layers.push(layer);
            uid = uid.checked_add(1).ok_or_else(|| MogrtError::InvalidPackage("too many layers".into()))?;
        }
    }
    if count == 0 {
        return Err(MogrtError::Unsupported("project.svg contains no supported static text or geometry".into()));
    }
    template.name = "Imported Motion Graphic".into();
    template.format = TEMPLATE_FORMAT.into();
    template.version = TEMPLATE_VERSION;
    Ok((template, supported, unsupported))
}

fn svg_canvas(root: roxmltree::Node<'_, '_>) -> Result<[u32; 2], MogrtError> {
    let view_box = root
        .attribute("viewBox")
        .map(|value| {
            let normalized = value.replace(',', " ");
            let values: Vec<f64> = normalized.split_ascii_whitespace().filter_map(|s| s.parse().ok()).collect();
            if values.len() != 4 || values.iter().any(|number| !number.is_finite()) || values[2] <= 0.0 || values[3] <= 0.0 {
                return Err(MogrtError::Unsupported("invalid SVG viewBox".into()));
            }
            Ok([values[0], values[1], values[2], values[3]])
        })
        .transpose()?;
    if view_box.is_some_and(|v| v[0] != 0.0 || v[1] != 0.0) || root.attribute("transform").is_some() {
        return Err(MogrtError::Unsupported("non-zero SVG viewBox origin or root transform".into()));
    }
    let dimension = |key: &str| root.attribute(key).and_then(parse_svg_number);
    let width = dimension("width").or_else(|| view_box.map(|v| v[2])).ok_or_else(|| MogrtError::Unsupported("SVG has no valid width or viewBox".into()))?;
    let height = dimension("height").or_else(|| view_box.map(|v| v[3])).ok_or_else(|| MogrtError::Unsupported("SVG has no valid height or viewBox".into()))?;
    if !width.is_finite() || !height.is_finite() || width < 1.0 || height < 1.0 || width > 16384.0 || height > 16384.0 {
        return Err(MogrtError::InvalidPackage("SVG canvas dimensions are outside supported bounds".into()));
    }
    Ok([width.round() as u32, height.round() as u32])
}

fn parse_svg_number(value: &str) -> Option<f64> {
    let value = value.trim().strip_suffix("px").unwrap_or(value.trim());
    let number = value.parse::<f64>().ok()?;
    number.is_finite().then_some(number)
}

fn attr_number(node: roxmltree::Node<'_, '_>, key: &str, default: f64) -> Result<f64, MogrtError> {
    let Some(value) = node.attribute(key) else { return Ok(default) };
    let number = parse_svg_number(value).ok_or_else(|| MogrtError::Unsupported(format!("unsupported SVG `{key}` value `{value}`")))?;
    if number.abs() > 1_000_000.0 {
        return Err(MogrtError::Unsupported(format!("SVG `{key}` value outside supported bounds")));
    }
    Ok(number)
}

fn set_param(layer: &mut crate::EffectInstance, key: &str, value: ParamValue) {
    layer.params.insert(key.to_string(), Param::new(value));
}

fn svg_color(value: Option<&str>) -> Result<Option<[f32; 4]>, MogrtError> {
    let Some(value) = value else { return Ok(Some([0.0, 0.0, 0.0, 1.0])) };
    if value == "none" {
        return Ok(None);
    }
    let Some(hex) = value.strip_prefix('#') else {
        return Err(MogrtError::Unsupported(format!("unsupported SVG fill `{value}`")));
    };
    let color = match hex.len() {
        3 => {
            let chars: Vec<char> = hex.chars().collect();
            let mut rgba = [0.0; 4];
            for (index, channel) in rgba.iter_mut().take(3).enumerate() {
                let digit = chars.get(index).and_then(|c| c.to_digit(16)).ok_or_else(|| MogrtError::Unsupported(format!("invalid SVG color `{value}`")))?;
                *channel = (digit * 17) as f32 / 255.0;
            }
            rgba[3] = 1.0;
            rgba
        }
        6 => {
            let mut rgba = [0.0; 4];
            for (index, channel) in rgba.iter_mut().take(3).enumerate() {
                let start = index * 2;
                let component = u8::from_str_radix(hex.get(start..start + 2).unwrap_or(""), 16)
                    .map_err(|_| MogrtError::Unsupported(format!("invalid SVG color `{value}`")))?;
                *channel = component as f32 / 255.0;
            }
            rgba[3] = 1.0;
            rgba
        }
        _ => return Err(MogrtError::Unsupported(format!("unsupported SVG fill `{value}`"))),
    };
    Ok(Some(color))
}

fn apply_fill(layer: &mut crate::EffectInstance, node: roxmltree::Node<'_, '_>) -> Result<(), MogrtError> {
    let fill_opacity = attr_number(node, "fill-opacity", 1.0)?;
    if !(0.0..=1.0).contains(&fill_opacity) {
        return Err(MogrtError::Unsupported("SVG fill-opacity outside 0..1".into()));
    }
    if let Some(color) = svg_color(node.attribute("fill"))? {
        let mut color = color;
        color[3] *= fill_opacity as f32;
        set_param(layer, "fill", ParamValue::Bool(true));
        set_param(layer, "fill_color", ParamValue::Color(color));
    } else {
        set_param(layer, "fill", ParamValue::Bool(false));
    }
    let stroke_width = attr_number(node, "stroke-width", 1.0)?;
    if !(0.0..=1000.0).contains(&stroke_width) {
        return Err(MogrtError::Unsupported("SVG stroke-width outside 0..1000".into()));
    }
    match node.attribute("stroke") {
        Some(stroke) => {
            if let Some(mut color) = svg_color(Some(stroke))? {
                let stroke_opacity = attr_number(node, "stroke-opacity", 1.0)?;
                if !(0.0..=1.0).contains(&stroke_opacity) {
                    return Err(MogrtError::Unsupported("SVG stroke-opacity outside 0..1".into()));
                }
                color[3] *= stroke_opacity as f32;
                set_param(layer, "stroke", ParamValue::Bool(true));
                set_param(layer, "stroke_color", ParamValue::Color(color));
                set_param(layer, "stroke_width", ParamValue::Float(stroke_width));
                set_param(layer, "stroke_type", ParamValue::Choice(1));
            } else {
                set_param(layer, "stroke", ParamValue::Bool(false));
            }
        }
        None => set_param(layer, "stroke", ParamValue::Bool(false)),
    }
    let opacity = attr_number(node, "opacity", 1.0)?;
    if !(0.0..=1.0).contains(&opacity) {
        return Err(MogrtError::Unsupported("SVG opacity outside 0..1".into()));
    }
    set_param(layer, "opacity", ParamValue::Float(opacity * 100.0));
    Ok(())
}

fn svg_text_layer(node: roxmltree::Node<'_, '_>) -> Result<Option<crate::EffectInstance>, MogrtError> {
    if node.children().any(|child| child.is_element()) {
        return Err(MogrtError::Unsupported("nested SVG text spans are not supported".into()));
    }
    let value = node.text().unwrap_or_default();
    if value.is_empty() {
        return Ok(None);
    }
    let x = attr_number(node, "x", 0.0)?;
    let y = attr_number(node, "y", 0.0)?;
    let size = attr_number(node, "font-size", 16.0)?;
    if size <= 0.0 || size > 2000.0 {
        return Err(MogrtError::Unsupported("SVG font-size outside supported bounds".into()));
    }
    let mut layer = new_text_layer(value, Vec2::new(x, y), size);
    if let Some(font) = node.attribute("font-family") {
        set_param(&mut layer, "font", ParamValue::Text(font.to_string()));
    }
    let weight = node.attribute("font-weight").unwrap_or("normal");
    if !matches!(weight, "normal" | "400" | "bold" | "700") {
        return Err(MogrtError::Unsupported(format!("unsupported SVG font-weight `{weight}`")));
    }
    let italic = match node.attribute("font-style").unwrap_or("normal") {
        "normal" => false,
        "italic" => true,
        value => return Err(MogrtError::Unsupported(format!("unsupported SVG font-style `{value}`"))),
    };
    let style = match (matches!(weight, "bold" | "700"), italic) {
        (true, true) => "Bold Italic",
        (true, false) => "Bold",
        (false, true) => "Italic",
        (false, false) => "Regular",
    };
    set_param(&mut layer, "font_style", ParamValue::Text(style.into()));
    apply_fill(&mut layer, node)?;
    let align = match node.attribute("text-anchor").unwrap_or("start") {
        "start" => 0,
        "middle" => 1,
        "end" => 2,
        value => return Err(MogrtError::Unsupported(format!("unsupported SVG text-anchor `{value}`"))),
    };
    set_param(&mut layer, "align", ParamValue::Choice(align));
    Ok(Some(layer))
}

fn svg_rect_layer(node: roxmltree::Node<'_, '_>) -> Result<Option<crate::EffectInstance>, MogrtError> {
    let x = attr_number(node, "x", 0.0)?;
    let y = attr_number(node, "y", 0.0)?;
    let width = attr_number(node, "width", 0.0)?;
    let height = attr_number(node, "height", 0.0)?;
    if width <= 0.0 || height <= 0.0 {
        return Ok(None);
    }
    if attr_number(node, "rx", 0.0)? != 0.0 || attr_number(node, "ry", 0.0)? != 0.0 {
        return Err(MogrtError::Unsupported("rounded SVG rectangles are not supported".into()));
    }
    let mut layer = new_shape_layer(0, Vec2::new(x + width / 2.0, y + height / 2.0), Vec2::new(width, height), vec![]);
    apply_fill(&mut layer, node)?;
    Ok(Some(layer))
}

fn svg_circle_layer(node: roxmltree::Node<'_, '_>) -> Result<Option<crate::EffectInstance>, MogrtError> {
    let r = attr_number(node, "r", 0.0)?;
    if r <= 0.0 {
        return Ok(None);
    }
    let mut layer = new_shape_layer(1, Vec2::new(attr_number(node, "cx", 0.0)?, attr_number(node, "cy", 0.0)?), Vec2::new(r * 2.0, r * 2.0), vec![]);
    apply_fill(&mut layer, node)?;
    Ok(Some(layer))
}

fn svg_ellipse_layer(node: roxmltree::Node<'_, '_>) -> Result<Option<crate::EffectInstance>, MogrtError> {
    let rx = attr_number(node, "rx", 0.0)?;
    let ry = attr_number(node, "ry", 0.0)?;
    if rx <= 0.0 || ry <= 0.0 {
        return Ok(None);
    }
    let mut layer = new_shape_layer(1, Vec2::new(attr_number(node, "cx", 0.0)?, attr_number(node, "cy", 0.0)?), Vec2::new(rx * 2.0, ry * 2.0), vec![]);
    apply_fill(&mut layer, node)?;
    Ok(Some(layer))
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use zip::write::SimpleFileOptions;

    use super::*;

    fn package(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let cursor = Cursor::new(Vec::new());
        let mut writer = zip::ZipWriter::new(cursor);
        for (name, data) in entries {
            writer.start_file(*name, SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated)).expect("fixture path accepted");
            writer.write_all(data).expect("fixture data written");
        }
        writer.finish().expect("fixture ZIP finished").into_inner()
    }

    #[test]
    fn rejects_malformed_archive() {
        assert!(matches!(inspect(b"not a ZIP"), Err(MogrtError::InvalidArchive(_))));
    }

    #[test]
    fn rejects_invalid_package() {
        let bytes = package(&[("thumbnail.png", b"png")]);
        assert!(matches!(inspect(&bytes), Err(MogrtError::InvalidPackage(_))));
    }

    #[test]
    fn rejects_oversized_entry_from_metadata() {
        let mut total = 0;
        assert_eq!(add_entry_size(MAX_ENTRY_BYTES + 1, &mut total), Err(MogrtError::DecompressedTooLarge));
        assert_eq!(add_entry_size(MAX_ENTRY_BYTES, &mut total), Ok(()));
        assert_eq!(add_entry_size(MAX_TOTAL_UNCOMPRESSED - MAX_ENTRY_BYTES + 1, &mut total), Err(MogrtError::DecompressedTooLarge));
    }

    #[test]
    fn rejects_path_traversal() {
        for path in ["../escape", "dir/../../escape", "/absolute", "C:/absolute", "dir\\escape"] {
            assert!(matches!(validate_entry_path(path), Err(MogrtError::UnsafePath(_))), "{path}");
        }
    }

    #[test]
    fn rejects_traversal_entry_in_package() {
        let bytes = package(&[("../escape", b"data"), ("manifest.json", b"{}")]);
        assert!(matches!(Package::open(&bytes), Err(MogrtError::UnsafePath(_))));
    }

    #[test]
    fn reports_animation_and_opaque_project_without_claiming_conversion() {
        let svg = br#"<svg xmlns="http://www.w3.org/2000/svg" width="320" height="180"><text x="10" y="20">Hello</text><animate attributeName="x"/></svg>"#;
        let bytes = package(&[("manifest.json", br#"{"name":"Animated title"}"#), ("project.svg", svg), ("project.mogrt", b"opaque")]);
        let report = inspect(&bytes).expect("package inspected");
        assert_eq!(report.name, "Animated title");
        assert_eq!(report.convertible, false);
        assert_eq!(report.editable_text, ["Hello"]);
        assert!(report.unsupported_features.iter().any(|f| f.contains("opaque After Effects project")));
        assert!(report.unsupported_features.iter().any(|f| f.contains("SVG animation")));
        assert!(convert(&bytes).is_err());
    }

    #[test]
    fn reports_animation_in_svg_only_package() {
        let svg = br#"<svg xmlns="http://www.w3.org/2000/svg" width="320" height="180"><text x="10" y="20">Hello</text><animate attributeName="x"/></svg>"#;
        let bytes = package(&[("manifest.json", br#"{"name":"Animated title"}"#), ("project.svg", svg)]);
        let converted = convert(&bytes).expect("static elements convert with warning");
        assert!(converted.report.unsupported_features.iter().any(|f| f.contains("SVG animation")));
    }

    #[test]
    fn converts_minimal_supported_fixture_to_fcgt_schema() {
        let svg = br##"<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360"><rect x="10" y="20" width="200" height="60" fill="#336699" fill-opacity="0.5" stroke="#112233" stroke-width="3"/><text x="20" y="55" font-family="Inter" font-size="24" fill="#ffffff" stroke="#000000" stroke-width="2">Headline</text></svg>"##;
        let bytes = package(&[("manifest.json", br#"{"name":"Minimal card"}"#), ("project.svg", svg)]);
        let converted = convert(&bytes).expect("supported package converted");
        assert_eq!(converted.template.format, TEMPLATE_FORMAT);
        assert_eq!(converted.template.version, TEMPLATE_VERSION);
        assert_eq!(converted.template.canvas, [640, 360]);
        assert_eq!(converted.report.canvas, Some([640, 360]));
        assert_eq!(converted.report.layer_count, Some(2));
        assert_eq!(converted.report.editable_text, ["Headline"]);
        assert_eq!(converted.template.layers.len(), 2);
        assert_eq!(converted.template.controls.len(), 1);
        assert_eq!(converted.template.controls[0].param, "text");
        assert_eq!(
            converted.template.layers[0].params["fill_color"].value,
            ParamValue::Color([0x33 as f32 / 255.0, 0x66 as f32 / 255.0, 0x99 as f32 / 255.0, 0.5])
        );
        assert_eq!(converted.template.layers[0].params["stroke"].value, ParamValue::Bool(true));
        assert_eq!(converted.template.layers[0].params["stroke_width"].value, ParamValue::Float(3.0));
        assert_eq!(converted.template.layers[0].params["stroke_type"].value, ParamValue::Choice(1));
        assert_eq!(converted.template.layers[1].params["stroke"].value, ParamValue::Bool(true));
        assert!(converted.report.supported_features.iter().any(|feature| feature == "static SVG solid strokes"));
        let serialized = converted.template.to_json();
        let parsed = GraphicsTemplate::from_bytes(serialized.as_bytes()).expect("serialized FCGT parses");
        assert_eq!(parsed, converted.template);
    }

    #[test]
    fn inspects_nested_aegraphic_definition_metadata() {
        let aep = package(&[("composition.bin", b"opaque composition"), ("Footage/card.png", b"image bytes")]);
        let aegraphic = package(&[
            ("Project/Graphic.aep", &aep),
            ("Footage/texture.png", b"texture"),
            ("Footage/Butterfly/butterfly-only-violet.ai", b"%PDF-1.6 vector footage"),
        ]);
        let definition = serde_json::json!({
            "authorApp": "aefx",
            "capsuleName": "Color Arrives System",
            "capsuleNameLocalized": {"strDB": [{"localeString": "en_US", "str": "Color Arrives System"}]},
            "description": "Sample template",
            "licenseID": "",
            "clientControls": [
                {"id": "color", "type": 4, "uiName": {"strDB": [{"localeString": "en_US", "str": "Circle Color"}]}, "value": [0.5, 0.25, 0.75, 1.0], "canAnimate": true},
                {"id": "name", "type": 6, "uiName": {"strDB": [{"localeString": "en_US", "str": "Color Name"}]}, "value": {"strDB": [{"localeString": "en_US", "str": "Violet"}]}, "canAnimate": true},
                {"id": "select", "type": 13, "uiName": {"strDB": [{"localeString": "en_US", "str": "Selector"}]}, "value": 1, "menucontent": [{"strDB": [{"localeString": "en_US", "str": "Violet"}]}]}
            ],
            "sourceInfoLocalized": {"en_US": {"framesize": {"size": {"x": 1920, "y": 1080}}}},
            "usedEffects": ["ADBE Fill", "CC Particle Systems II"],
            "usedFontsLocalized": {"en_US": ["Inter-Regular"]},
            "usedCompRenderers": ["ADBE Escher"]
        });
        let definition = serde_json::to_vec(&definition).expect("definition serialized");
        let bytes = package(&[("definition.json", &definition), ("project.aegraphic", &aegraphic), ("thumb.png", b"preview")]);

        let report = inspect(&bytes).expect("nested Adobe-style package inspected");
        assert_eq!(report.name, "Color Arrives System");
        assert_eq!(report.author_app.as_deref(), Some("aefx"));
        assert_eq!(report.description.as_deref(), Some("Sample template"));
        assert_eq!(report.canvas, Some([1920, 1080]));
        assert_eq!(report.layer_count, None, "opaque native project layers must not be guessed");
        assert_eq!(report.editable_text, ["Violet"]);
        assert_eq!(report.controls.len(), 3);
        assert!(report.controls.iter().any(|control| control.name == "Selector" && control.choices == ["Violet"]));
        assert!(report.project_data.iter().any(|name| name == "project.aegraphic/Project/Graphic.aep"));
        assert!(report.assets.iter().any(|name| name == "project.aegraphic/Footage/texture.png"));
        assert!(report.assets.iter().any(|name| name == "thumb.png"));
        assert!(report.unsupported_features.iter().any(|feature| feature.contains("embedded raster footage was inventoried but not placed")));
        assert!(report.convertible);
        assert!(report.partial);
        assert!(report.unsupported_features.iter().any(|feature| feature.contains("CC Particle Systems II")));
        assert!(report.unsupported_features.iter().any(|feature| feature.contains("layer count and timeline are unknown")));
        let converted = convert(&bytes).expect("supported definition controls convert as a partial template");
        assert!(converted.template.validate().is_ok());
        assert_eq!(converted.template.duration, Tick(5 * TICKS_PER_SECOND));
        assert_eq!(converted.report.converted_layer_count, Some(4));
        assert_eq!(converted.template.controls.len(), 2);
        let text_position = converted
            .template
            .layers
            .iter()
            .find(|layer| layer.effect == graphic::TEXT_LAYER)
            .and_then(|layer| layer.params.get("position"))
            .and_then(|param| param.value.as_vec2());
        assert_eq!(text_position, Some(Vec2::new(750.0, 654.0)));
        let card_position = converted
            .template
            .layers
            .iter()
            .find(|layer| matches!(layer.params.get("name").map(|param| &param.value), Some(ParamValue::Text(name)) if name == "Approximate control panel"))
            .and_then(|layer| layer.params.get("position"))
            .and_then(|param| param.value.as_vec2());
        assert_eq!(card_position, Some(Vec2::new(705.0, 697.5)));
        let swatch_position = converted
            .template
            .layers
            .iter()
            .find(|layer| matches!(layer.params.get("name").map(|param| &param.value), Some(ParamValue::Text(name)) if name == "Circle Color"))
            .and_then(|layer| layer.params.get("position"))
            .and_then(|param| param.value.as_vec2());
        assert_eq!(swatch_position, Some(Vec2::new(555.0, 699.0)));
        assert!(converted.report.partial);
        assert!(
            converted.report.unsupported_features.iter().any(|feature| feature.contains("butterfly Illustrator/PDF footage assets are present but omitted"))
        );
    }
}
