use clap::Args;
use ppocr_rs::ModelAccess;
use std::path::PathBuf;

/// Shared pinned-model cache options used by every command-line entry point.
#[derive(Debug, Args)]
pub struct ModelStoreArgs {
    /// Directory where pinned model packages are stored.
    #[arg(long, env = "PPOCR_MODEL_DIR", default_value = "models")]
    pub model_dir: PathBuf,

    /// Fail when the pinned model is not already cached.
    #[arg(long, conflicts_with = "verify_models")]
    pub offline: bool,

    /// Recompute hashes for the pinned model package before loading it.
    #[arg(long, conflicts_with = "offline")]
    pub verify_models: bool,
}

impl ModelStoreArgs {
    pub const fn access(&self) -> ModelAccess {
        if self.verify_models {
            ModelAccess::Verify
        } else if self.offline {
            ModelAccess::Offline
        } else {
            ModelAccess::Online
        }
    }
}
