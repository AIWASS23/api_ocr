//! `self_query` — o `SelfQueryRetriever`: a consulta em linguagem natural vira texto de busca +
//! filtro de metadados ("leis de 2020 sobre saúde" → busca "saúde", filtro `ano = 2020`).
//!
//! Com um construtor injetado (`with_query_constructor`), ele decide. Sem ele, com
//! `options.metadata_fields` (`[{"name", "description", "type"}]`) e um LLM, o LLM devolve
//! `{"query", "filter"}` em JSON (só os campos declarados entram no filtro). Sem nada disso, a
//! consulta vai direto para a estratégia base.

use chunking::Document;
use serde_json::{Map, Value};

use crate::context::{Component, Ctx};
use crate::error::Result;

fn fields(ctx: &Ctx) -> Vec<(String, String, String)> {
    match ctx.cfg.option("metadata_fields") {
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(|f| {
                let s = |k: &str, d: &str| f.get(k).and_then(Value::as_str).unwrap_or(d).to_string();
                f.get("name").and_then(Value::as_str).map(|_| (s("name", ""), s("description", ""), s("type", "string")))
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// Primeiro objeto JSON da resposta do LLM.
fn first_json_object(text: &str) -> Option<Map<String, Value>> {
    let start = text.find('{')?;
    let mut depth = 0i32;
    let (mut in_str, mut esc) = (false, false);
    for (i, c) in text[start..].char_indices() {
        match c {
            _ if esc => esc = false,
            '\\' if in_str => esc = true,
            '"' => in_str = !in_str,
            '{' if !in_str => depth += 1,
            '}' if !in_str => {
                depth -= 1;
                if depth == 0 {
                    return serde_json::from_str(&text[start..start + i + 1]).ok();
                }
            }
            _ => {}
        }
    }
    None
}

/// Consulta e filtro escritos pelo LLM (`None` se a resposta não for utilizável).
fn llm_structured(ctx: &Ctx, query: &str, fields: &[(String, String, String)]) -> Result<Option<(String, Map<String, Value>)>> {
    let description = ctx.cfg.option_str("document_content_description").unwrap_or("Documents available for retrieval");
    let listed: Vec<String> = fields.iter().map(|(n, d, t)| format!("- {n} ({t}): {d}")).collect();
    let prompt = format!(
        "Converta a pergunta numa consulta estruturada. Responda só com JSON no formato \
         {{\"query\": \"texto para a busca\", \"filter\": {{\"campo\": valor}}}}.\n\
         Conteúdo dos documentos: {description}\nCampos de metadado:\n{}\n\
         Use só esses campos no filtro; sem filtro, use {{}}.\nPergunta: {query}",
        listed.join("\n")
    );
    let Some(obj) = first_json_object(&ctx.call_llm(&prompt)?) else { return Ok(None) };
    let text = obj.get("query").and_then(Value::as_str).filter(|s| !s.trim().is_empty()).unwrap_or(query).to_string();
    let allowed: Vec<&str> = fields.iter().map(|(n, ..)| n.as_str()).collect();
    let filter = obj
        .get("filter")
        .and_then(Value::as_object)
        .map(|m| m.iter().filter(|(k, v)| allowed.contains(&k.as_str()) && !v.is_object()).map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default();
    Ok(Some((text, filter)))
}

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    let base = Component::Strategy(ctx.base_strategy().into());
    let constructed = if let Some(c) = &ctx.env.deps.query_constructor {
        Some(c(query)?)
    } else {
        let f = fields(ctx);
        if !f.is_empty() && (ctx.env.deps.llm.is_some() || ctx.cfg.llm.is_some()) { llm_structured(ctx, query, &f)? } else { None }
    };
    match constructed {
        None => ctx.component(query, &base),
        Some((text, filters)) => {
            let child = ctx.child(|c| c.filters.extend(filters));
            child.component(&text, &base)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::first_json_object;

    #[test]
    fn json_in_llm_text() {
        let o = first_json_object("Claro: {\"query\": \"saúde {x}\", \"filter\": {\"ano\": 2020}} fim").unwrap();
        assert_eq!(o["filter"]["ano"], 2020);
        assert!(first_json_object("sem json").is_none());
    }
}
