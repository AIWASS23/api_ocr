use data_preparation::{DataPreparation, DataPreparationConfig, VlmProvider};

/// uso: process <arquivo-ou-pasta>
/// PARALLEL_FILES=n arquivos leves simultâneos (0 = automático, 1 = sequencial)
/// VLM opcional: VLM_MODEL=gemma3:4b [VLM_URL=http://localhost:11434/v1/chat/completions]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("uso: process <arquivo-ou-pasta>");

    let mut config = DataPreparationConfig::default();
    if let Ok(model) = std::env::var("VLM_MODEL") {
        config.provider = VlmProvider::Local {
            base_url: std::env::var("VLM_URL")
                .unwrap_or_else(|_| "http://localhost:11434/v1/chat/completions".into()),
            model,
        };
    }

    if let Some(n) = std::env::var("PARALLEL_FILES").ok().and_then(|v| v.parse().ok()) {
        config.parallel.max_light_files = n;
    }

    let prep = DataPreparation::new(config)?;
    let (docs, errors) = prep.process_files(&path, "txt")?;

    for d in &docs {
        println!("--- {:?}\n{}\n", d.metadata, d.page_content);
    }
    for e in &errors {
        println!("ERRO {}: {}", e.file, e.error);
    }
    println!("{} documentos, {} erros", docs.len(), errors.len());
    Ok(())
}
