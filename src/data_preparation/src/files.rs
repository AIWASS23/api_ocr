use std::path::{Path, PathBuf};

use walkdir::WalkDir;

/// Extensões aceitas (espelha `SUPPORTED_EXTENSIONS` do código Python).
pub const SUPPORTED_EXTENSIONS: &[&str] = &[
    "pdf", "docx", "pptx", "xlsx", "csv", "md", "html", "htm", "json", "asciidoc", "adoc", "png",
    "jpg", "jpeg", "tif", "tiff", "webp", "bmp", "mp3", "vtt", "xml",
];

pub fn is_supported(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| SUPPORTED_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Se `path` é um arquivo, devolve só ele; se é uma pasta, lista
/// recursivamente os arquivos de extensão suportada (ordenados).
pub fn detects_folder_or_file(path: impl AsRef<Path>) -> Vec<PathBuf> {
    let path = path.as_ref();
    if path.is_file() {
        return vec![path.to_path_buf()];
    }

    let mut files: Vec<PathBuf> = WalkDir::new(path)
        .follow_links(true)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .filter(|p| is_supported(p))
        .collect();
    files.sort();
    files
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_supported_files_recursively_including_symlinks() {
        let dir = std::env::temp_dir().join(format!("dp_files_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("a.PDF"), b"").unwrap();
        std::fs::write(dir.join("sub/b.docx"), b"").unwrap();
        std::fs::write(dir.join("ignorado.txt"), b"").unwrap();
        std::os::unix::fs::symlink(dir.join("a.PDF"), dir.join("link.pdf")).unwrap();

        let found: Vec<_> = detects_folder_or_file(&dir)
            .into_iter()
            .map(|p| p.strip_prefix(&dir).unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(found, ["a.PDF", "link.pdf", "sub/b.docx"]);

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
