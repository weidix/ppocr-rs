//! Pinned PP-OCR model metadata and local model storage.

use anyhow::{Context, Result, bail};
use fs2::FileExt;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    fmt,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    str::FromStr,
    time::Duration,
};

const CATALOG: &str = include_str!("../models.json");
const CACHE_MARKER: &str = ".ppocr-rs.complete";
const LOCK_FILE: &str = ".ppocr-rs.lock";
const DOWNLOAD_ATTEMPTS: usize = 3;

/// A PP-OCR model role.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ModelKind {
    /// Text-region detector.
    Detector,
    /// Text recognizer.
    Recognizer,
}

impl ModelKind {
    /// Returns the stable model-directory suffix.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Detector => "det",
            Self::Recognizer => "rec",
        }
    }
}

impl fmt::Display for ModelKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for ModelKind {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "det" | "detector" => Ok(Self::Detector),
            "rec" | "recognizer" => Ok(Self::Recognizer),
            _ => Err(format!(
                "unsupported model kind {value:?}; expected det or rec"
            )),
        }
    }
}

/// A released PP-OCRv6 model tier.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ModelSize {
    /// Highest-accuracy released tier.
    Medium,
    /// Balanced released tier.
    Small,
    /// Smallest released tier.
    Tiny,
}

impl ModelSize {
    /// Returns the stable model-directory prefix.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Medium => "medium",
            Self::Small => "small",
            Self::Tiny => "tiny",
        }
    }

    /// Returns the recognizer output-class count for this tier.
    pub const fn recognizer_classes(self) -> usize {
        match self {
            Self::Medium | Self::Small => 18_710,
            Self::Tiny => 6_906,
        }
    }
}

impl fmt::Display for ModelSize {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for ModelSize {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "medium" => Ok(Self::Medium),
            "small" => Ok(Self::Small),
            "tiny" => Ok(Self::Tiny),
            _ => Err(format!(
                "unsupported model size {value:?}; expected medium, small, or tiny"
            )),
        }
    }
}

/// Paths to one complete model package.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelPaths {
    /// Directory containing the model package.
    pub directory: PathBuf,
    /// Safetensors weights.
    pub weights: PathBuf,
    /// Model architecture metadata.
    pub config: PathBuf,
    /// Paddle inference configuration, including recognizer dictionaries.
    pub inference: PathBuf,
    /// Image preprocessing metadata.
    pub preprocessor_config: PathBuf,
}

/// Paths to the detector and recognizer required for end-to-end OCR.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OcrModelPaths {
    /// Detector package.
    pub detector: ModelPaths,
    /// Recognizer package.
    pub recognizer: ModelPaths,
}

/// Cache-validation and download policy used while resolving pinned models.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelAccess {
    /// Use a complete cache when available and download missing or invalid files.
    Online,
    /// Require a complete cache and never use the network.
    Offline,
    /// Require a complete cache and recompute every asset digest.
    Verify,
}

/// A model cache backed by the checked-in, pinned model catalog.
#[derive(Clone, Debug)]
pub struct ModelStore {
    root: PathBuf,
}

impl Default for ModelStore {
    fn default() -> Self {
        let root = std::env::var_os("PPOCR_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("models"));
        Self::new(root)
    }
}

impl ModelStore {
    /// Creates a store rooted at `root`.
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
        }
    }

    /// Returns the model-cache root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Returns the expected paths without touching the filesystem.
    pub fn paths(&self, kind: ModelKind, size: ModelSize) -> Result<ModelPaths> {
        let manifest = model_manifest(kind, size)?;
        model_paths(&self.root, &manifest)
    }

    /// Ensures one model package is present, verified, and ready to load.
    ///
    /// Missing or invalid files are downloaded from the model's pinned
    /// Hugging Face revision. Downloads are serialized across processes and
    /// are verified before becoming visible in the cache.
    pub fn ensure(&self, kind: ModelKind, size: ModelSize) -> Result<ModelPaths> {
        self.resolve(kind, size, ModelAccess::Online)
    }

    /// Ensures a model package is already available without using the network.
    pub fn ensure_offline(&self, kind: ModelKind, size: ModelSize) -> Result<ModelPaths> {
        self.resolve(kind, size, ModelAccess::Offline)
    }

    /// Fully revalidates every file in a cached model package.
    pub fn verify(&self, kind: ModelKind, size: ModelSize) -> Result<ModelPaths> {
        self.resolve(kind, size, ModelAccess::Verify)
    }

    /// Resolves one pinned model package using one explicit cache policy.
    pub fn resolve(
        &self,
        kind: ModelKind,
        size: ModelSize,
        access: ModelAccess,
    ) -> Result<ModelPaths> {
        if access != ModelAccess::Verify {
            return self.ensure_resolved(kind, size, access == ModelAccess::Offline);
        }
        let manifest = model_manifest(kind, size)?;
        let paths = model_paths(&self.root, &manifest)?;
        let _lock = self.lock()?;
        if !all_files_are_valid(&paths, &manifest)? {
            bail!(
                "{} is missing or invalid in {}; run without --offline to repair it",
                manifest.name,
                paths.directory.display()
            );
        }
        write_cache_marker(&paths, &manifest)?;
        Ok(paths)
    }

    /// Ensures the detector and recognizer packages for one OCR configuration.
    pub fn ensure_pair(
        &self,
        detector_size: ModelSize,
        recognizer_size: ModelSize,
    ) -> Result<OcrModelPaths> {
        self.resolve_pair(detector_size, recognizer_size, ModelAccess::Online)
    }

    /// Resolves the detector and recognizer packages using one cache policy.
    pub fn resolve_pair(
        &self,
        detector_size: ModelSize,
        recognizer_size: ModelSize,
        access: ModelAccess,
    ) -> Result<OcrModelPaths> {
        Ok(OcrModelPaths {
            detector: self.resolve(ModelKind::Detector, detector_size, access)?,
            recognizer: self.resolve(ModelKind::Recognizer, recognizer_size, access)?,
        })
    }

    fn ensure_resolved(
        &self,
        kind: ModelKind,
        size: ModelSize,
        offline: bool,
    ) -> Result<ModelPaths> {
        let manifest = model_manifest(kind, size)?;
        let paths = model_paths(&self.root, &manifest)?;
        if cache_is_complete(&paths, &manifest)? {
            return Ok(paths);
        }

        let _lock = self.lock()?;
        if cache_is_complete(&paths, &manifest)? {
            return Ok(paths);
        }
        if all_files_are_valid(&paths, &manifest)? {
            write_cache_marker(&paths, &manifest)?;
            return Ok(paths);
        }
        if offline {
            bail!(
                "{} is not available in {}; rerun without --offline to download it",
                manifest.name,
                paths.directory.display()
            );
        }

        fs::create_dir_all(&paths.directory)
            .with_context(|| format!("create model directory {}", paths.directory.display()))?;
        for file in &manifest.files {
            let destination = asset_path(&paths.directory, &file.name)?;
            if !file_is_valid(&destination, file)? {
                download_with_retry(&manifest, file, &destination)?;
            }
        }
        if !all_files_are_valid(&paths, &manifest)? {
            bail!("downloaded {} did not pass validation", manifest.name);
        }
        write_cache_marker(&paths, &manifest)?;
        Ok(paths)
    }

    fn lock(&self) -> Result<File> {
        fs::create_dir_all(&self.root)
            .with_context(|| format!("create model cache {}", self.root.display()))?;
        let lock_path = self.root.join(LOCK_FILE);
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)
            .with_context(|| format!("open model cache lock {}", lock_path.display()))?;
        FileExt::lock_exclusive(&file)
            .with_context(|| format!("lock model cache {}", self.root.display()))?;
        Ok(file)
    }
}

#[derive(Clone, Debug, Deserialize)]
struct Catalog {
    models: Vec<ModelManifest>,
}

#[derive(Clone, Debug, Deserialize)]
struct ModelManifest {
    name: String,
    repository: String,
    revision: String,
    files: Vec<ModelFile>,
}

#[derive(Clone, Debug, Deserialize)]
struct ModelFile {
    name: String,
    bytes: u64,
    sha256: String,
}

fn model_manifest(kind: ModelKind, size: ModelSize) -> Result<ModelManifest> {
    let catalog: Catalog = serde_json::from_str(CATALOG).context("parse embedded model catalog")?;
    let name = format!("{}-{}", size.as_str(), kind.as_str());
    catalog
        .models
        .into_iter()
        .find(|model| model.name == name)
        .with_context(|| format!("model catalog does not contain {name}"))
}

fn model_paths(root: &Path, manifest: &ModelManifest) -> Result<ModelPaths> {
    let directory = root.join(&manifest.name);
    Ok(ModelPaths {
        weights: asset_path(&directory, "model.safetensors")?,
        config: asset_path(&directory, "config.json")?,
        inference: asset_path(&directory, "inference.yml")?,
        preprocessor_config: asset_path(&directory, "preprocessor_config.json")?,
        directory,
    })
}

fn asset_path(directory: &Path, name: &str) -> Result<PathBuf> {
    let path = Path::new(name);
    if path.components().count() != 1
        || !matches!(path.components().next(), Some(Component::Normal(_)))
    {
        bail!("model catalog contains invalid asset path {name:?}");
    }
    Ok(directory.join(path))
}

fn cache_is_complete(paths: &ModelPaths, manifest: &ModelManifest) -> Result<bool> {
    let marker = paths.directory.join(CACHE_MARKER);
    let found = match fs::read_to_string(&marker) {
        Ok(found) => found,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error).with_context(|| format!("read {}", marker.display())),
    };
    if found.trim() != manifest_fingerprint(manifest) {
        return Ok(false);
    }
    manifest.files.iter().try_fold(true, |complete, file| {
        Ok(complete && file_has_expected_length(&asset_path(&paths.directory, &file.name)?, file)?)
    })
}

fn all_files_are_valid(paths: &ModelPaths, manifest: &ModelManifest) -> Result<bool> {
    manifest.files.iter().try_fold(true, |valid, file| {
        Ok(valid && file_is_valid(&asset_path(&paths.directory, &file.name)?, file)?)
    })
}

fn file_has_expected_length(path: &Path, file: &ModelFile) -> Result<bool> {
    match fs::metadata(path) {
        Ok(metadata) => Ok(metadata.is_file() && metadata.len() == file.bytes),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error).with_context(|| format!("inspect model file {}", path.display())),
    }
}

fn file_is_valid(path: &Path, file: &ModelFile) -> Result<bool> {
    if !file_has_expected_length(path, file)? {
        return Ok(false);
    }
    let actual = sha256_file(path)?;
    Ok(actual == file.sha256)
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut reader =
        File::open(path).with_context(|| format!("open model file {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("read model file {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn write_cache_marker(paths: &ModelPaths, manifest: &ModelManifest) -> Result<()> {
    fs::create_dir_all(&paths.directory)
        .with_context(|| format!("create model directory {}", paths.directory.display()))?;
    let marker = paths.directory.join(CACHE_MARKER);
    let temporary = paths.directory.join(format!("{CACHE_MARKER}.part"));
    fs::write(&temporary, format!("{}\n", manifest_fingerprint(manifest)))
        .with_context(|| format!("write model cache marker {}", temporary.display()))?;
    replace_file(&temporary, &marker)
}

fn manifest_fingerprint(manifest: &ModelManifest) -> String {
    let mut hasher = Sha256::new();
    hasher.update(manifest.name.as_bytes());
    hasher.update([0]);
    hasher.update(manifest.repository.as_bytes());
    hasher.update([0]);
    hasher.update(manifest.revision.as_bytes());
    for file in &manifest.files {
        hasher.update([0]);
        hasher.update(file.name.as_bytes());
        hasher.update([0]);
        hasher.update(file.bytes.to_le_bytes());
        hasher.update(file.sha256.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn download_with_retry(
    manifest: &ModelManifest,
    file: &ModelFile,
    destination: &Path,
) -> Result<()> {
    let temporary = destination.with_file_name(format!(".{}.part", file.name));
    let mut last_error = None;
    for attempt in 1..=DOWNLOAD_ATTEMPTS {
        remove_file_if_exists(&temporary)?;
        eprintln!(
            "downloading {} ({}/{DOWNLOAD_ATTEMPTS})",
            destination.display(),
            attempt
        );
        match download_once(manifest, file, &temporary) {
            Ok(()) => return replace_file(&temporary, destination),
            Err(error) => {
                let _ = remove_file_if_exists(&temporary);
                last_error = Some(error);
                if attempt < DOWNLOAD_ATTEMPTS {
                    std::thread::sleep(Duration::from_millis(250 * attempt as u64));
                }
            }
        }
    }
    Err(last_error
        .expect("download attempts are positive")
        .context(format!(
            "download {} from pinned revision {}",
            destination.display(),
            manifest.revision
        )))
}

fn download_once(manifest: &ModelManifest, file: &ModelFile, temporary: &Path) -> Result<()> {
    let url = format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        manifest.repository, manifest.revision, file.name
    );
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(20))
        .timeout_read(Duration::from_secs(120))
        .timeout_write(Duration::from_secs(120))
        .build();
    let response = agent
        .get(&url)
        .set(
            "User-Agent",
            concat!("ppocr-rs/", env!("CARGO_PKG_VERSION")),
        )
        .call()
        .with_context(|| format!("request {url}"))?;
    let mut reader = response.into_reader();
    let mut writer = File::create(temporary)
        .with_context(|| format!("create temporary model file {}", temporary.display()))?;
    let mut hasher = Sha256::new();
    let mut bytes = 0u64;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("read {url}"))?;
        if read == 0 {
            break;
        }
        writer
            .write_all(&buffer[..read])
            .with_context(|| format!("write temporary model file {}", temporary.display()))?;
        hasher.update(&buffer[..read]);
        bytes = bytes
            .checked_add(read as u64)
            .context("downloaded model file size overflow")?;
    }
    writer
        .sync_all()
        .with_context(|| format!("sync temporary model file {}", temporary.display()))?;
    if bytes != file.bytes {
        bail!(
            "downloaded {} bytes for {}; expected {}",
            bytes,
            file.name,
            file.bytes
        );
    }
    let actual = format!("{:x}", hasher.finalize());
    if actual != file.sha256 {
        bail!(
            "checksum mismatch for {}; expected {}, found {actual}",
            file.name,
            file.sha256
        );
    }
    Ok(())
}

fn replace_file(source: &Path, destination: &Path) -> Result<()> {
    remove_file_if_exists(destination)?;
    fs::rename(source, destination).with_context(|| {
        format!(
            "move downloaded model file {} to {}",
            source.display(),
            destination.display()
        )
    })
}

fn remove_file_if_exists(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("remove {}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn parses_all_pinned_model_packages() {
        for size in [ModelSize::Medium, ModelSize::Small, ModelSize::Tiny] {
            for kind in [ModelKind::Detector, ModelKind::Recognizer] {
                let manifest = model_manifest(kind, size).expect("model manifest");
                assert_eq!(manifest.files.len(), 4);
                for name in [
                    "model.safetensors",
                    "config.json",
                    "inference.yml",
                    "preprocessor_config.json",
                ] {
                    assert!(manifest.files.iter().any(|file| file.name == name));
                }
            }
        }
    }

    #[test]
    fn cache_marker_avoids_rehashing_unchanged_files() {
        let root = temporary_root();
        let digest = format!("{:x}", Sha256::digest(b"abc"));
        let manifest = ModelManifest {
            name: "test-det".to_owned(),
            repository: "owner/repository".to_owned(),
            revision: "revision".to_owned(),
            files: [
                "model.safetensors",
                "config.json",
                "inference.yml",
                "preprocessor_config.json",
            ]
            .into_iter()
            .map(|name| ModelFile {
                name: name.to_owned(),
                bytes: 3,
                sha256: digest.clone(),
            })
            .collect(),
        };
        let paths = model_paths(&root, &manifest).expect("model paths");
        fs::create_dir_all(&paths.directory).expect("create model directory");
        for file in &manifest.files {
            fs::write(paths.directory.join(&file.name), b"abc").expect("write model file");
        }

        assert!(all_files_are_valid(&paths, &manifest).expect("validate model files"));
        assert!(!cache_is_complete(&paths, &manifest).expect("check cache marker"));
        write_cache_marker(&paths, &manifest).expect("write cache marker");
        assert!(cache_is_complete(&paths, &manifest).expect("check cache marker"));

        fs::write(&paths.weights, b"bad").expect("modify model file");
        assert!(!all_files_are_valid(&paths, &manifest).expect("verify model files"));
        fs::remove_dir_all(root).expect("remove temporary model directory");
    }

    #[test]
    fn parses_model_identifiers() {
        assert_eq!("detector".parse(), Ok(ModelKind::Detector));
        assert_eq!("tiny".parse(), Ok(ModelSize::Tiny));
        assert!("large".parse::<ModelSize>().is_err());
    }

    fn temporary_root() -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "ppocr-rs-model-test-{}-{stamp}",
            std::process::id()
        ))
    }
}
