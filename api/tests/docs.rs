mod common;
use std::collections::BTreeSet;

use common::*;
use serde_json::Value;

/// Rotas REAIS registradas em `router()`, como (método, caminho). Se alguém adicionar uma rota e
/// esquecer de documentar (ou o contrário), `documented_routes_match_real_routes` falha.
const REAL_ROUTES: &[(&str, &str)] = &[
    ("get", "/healthz"),
    ("get", "/readyz"),
    ("get", "/.well-known/oauth-authorization-server"),
    ("get", "/oauth/authorize"),
    ("post", "/oauth/authorize"),
    ("post", "/oauth/token"),
    ("post", "/oauth/revoke"),
    ("post", "/auth/register"),
    ("get", "/v1/me"),
    ("post", "/v1/ocr"),
];

async fn docs_app() -> TestApp {
    TestApp::start_with(|c| c.enable_docs = true).await
}

async fn spec(app: &TestApp) -> Value {
    let r = app.http.get(app.url("/openapi.json")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    r.json().await.unwrap()
}

#[tokio::test]
async fn docs_are_off_by_default() {
    let app = TestApp::start().await;
    for p in ["/docs", "/docs/", "/openapi.json", "/docs/swagger-ui-bundle.js", "/docs/oauth2-redirect.html"] {
        assert_eq!(app.http.get(app.url(p)).send().await.unwrap().status(), 404, "{p} deveria sumir");
    }
}

#[tokio::test]
async fn documented_routes_match_real_routes() {
    let app = docs_app().await;
    let doc = spec(&app).await;
    assert!(doc["openapi"].as_str().unwrap().starts_with("3."));

    let mut documented = BTreeSet::new();
    for (path, item) in doc["paths"].as_object().unwrap() {
        for method in item.as_object().unwrap().keys() {
            documented.insert((method.clone(), path.clone()));
        }
    }
    let real: BTreeSet<(String, String)> = REAL_ROUTES.iter().map(|(m, p)| (m.to_string(), p.to_string())).collect();
    assert_eq!(documented, real, "documentação e rotas reais divergem");

    // Cada operação documentada EXISTE de fato (nem 404 nem 405).
    for (method, path) in &documented {
        let req = match method.as_str() {
            "get" => app.http.get(app.url(path)),
            "post" => app.http.post(app.url(path)),
            other => panic!("método inesperado {other}"),
        };
        let status = req.send().await.unwrap().status().as_u16();
        assert!(status != 404 && status != 405, "{method} {path} -> {status}");
    }
}

#[tokio::test]
async fn every_ref_resolves_and_operations_are_complete() {
    let app = docs_app().await;
    let doc = spec(&app).await;
    let schemas = doc["components"]["schemas"].as_object().unwrap();

    fn refs(v: &Value, out: &mut Vec<String>) {
        match v {
            Value::Object(m) => {
                for (k, x) in m {
                    if k == "$ref" {
                        out.push(x.as_str().unwrap().to_string());
                    } else {
                        refs(x, out);
                    }
                }
            }
            Value::Array(a) => a.iter().for_each(|x| refs(x, out)),
            _ => {}
        }
    }
    let mut all = vec![];
    refs(&doc, &mut all);
    assert!(!all.is_empty());
    for r in all {
        let name = r.strip_prefix("#/components/schemas/").unwrap_or_else(|| panic!("$ref estranho: {r}"));
        assert!(schemas.contains_key(name), "$ref quebrado: {r}");
    }

    for (path, item) in doc["paths"].as_object().unwrap() {
        for (method, op) in item.as_object().unwrap() {
            assert!(op["tags"].as_array().is_some_and(|t| !t.is_empty()), "{method} {path} sem tag");
            assert!(op["responses"].as_object().is_some_and(|r| !r.is_empty()), "{method} {path} sem respostas");
            assert!(op["description"].is_string() || op["summary"].is_string() || path.starts_with("/healthz") || path.starts_with("/readyz"), "{method} {path} sem descrição");
        }
    }
}

#[tokio::test]
async fn security_schemes_use_the_real_issuer_and_protected_routes_require_them() {
    let app = docs_app().await;
    let doc = spec(&app).await;
    let flow = &doc["components"]["securitySchemes"]["oauth2"]["flows"]["authorizationCode"];
    assert_eq!(flow["authorizationUrl"], format!("{ISSUER}/oauth/authorize"));
    assert_eq!(flow["tokenUrl"], format!("{ISSUER}/oauth/token"));
    assert!(flow["scopes"]["ocr"].is_string() && flow["scopes"]["profile"].is_string());
    assert_eq!(doc["components"]["securitySchemes"]["bearer"]["scheme"], "bearer");
    assert_eq!(doc["servers"][0]["url"], ISSUER);

    // Rotas protegidas declaram segurança; as públicas não.
    for (path, method, protected) in [("/v1/me", "get", true), ("/v1/ocr", "post", true), ("/oauth/token", "post", false), ("/auth/register", "post", false)] {
        let sec = &doc["paths"][path][method]["security"];
        assert_eq!(sec.is_array() && !sec.as_array().unwrap().is_empty(), protected, "{method} {path}");
    }
}

/// A documentação não pode mentir: os campos reais das respostas têm que existir no schema (e vice-versa).
#[tokio::test]
async fn documented_schemas_match_real_responses() {
    let app = docs_app().await;
    let doc = spec(&app).await;
    let props = |name: &str| -> BTreeSet<String> {
        doc["components"]["schemas"][name]["properties"].as_object().unwrap_or_else(|| panic!("schema {name}")).keys().cloned().collect()
    };
    let keys = |v: &Value| -> BTreeSet<String> { v.as_object().unwrap().keys().cloned().collect() };

    // Token
    let (verifier, challenge) = pkce();
    let code = app.login_for_code(&challenge, "s").await;
    let token: Value = app.exchange(&code, &verifier).await.json().await.unwrap();
    assert_eq!(keys(&token), props("TokenResponse"));
    let access = token["access_token"].as_str().unwrap();

    // /v1/me
    let me: Value = app.http.get(app.url("/v1/me")).bearer_auth(access).send().await.unwrap().json().await.unwrap();
    assert_eq!(keys(&me), props("MeResponse"));

    // /v1/ocr
    let form = reqwest::multipart::Form::new().part("file", reqwest::multipart::Part::bytes(minimal_docx("doc")).file_name("a.docx"));
    let ocr: Value = app.http.post(app.url("/v1/ocr")).bearer_auth(access).multipart(form).send().await.unwrap().json().await.unwrap();
    assert_eq!(keys(&ocr), props("OcrResponse"));
    assert_eq!(keys(&ocr["documents"][0]), props("OcrDocument"));

    // Erro e metadados
    let err: Value = app.http.get(app.url("/v1/me")).send().await.unwrap().json().await.unwrap();
    assert_eq!(keys(&err), props("ErrorBody"));
    let meta: Value = app.http.get(app.url("/.well-known/oauth-authorization-server")).send().await.unwrap().json().await.unwrap();
    assert_eq!(keys(&meta), props("AuthServerMetadata"));
}

#[tokio::test]
async fn swagger_ui_is_served_from_the_binary_with_a_scoped_csp() {
    let app = docs_app().await;
    let page = app.http.get(app.url("/docs/")).send().await.unwrap();
    assert_eq!(page.status(), 200);
    let csp = page.headers()["content-security-policy"].to_str().unwrap().to_string();
    // Scripts só do próprio servidor: nada de CDN, nada de inline.
    assert!(csp.contains("script-src 'self'") && !csp.contains("script-src 'self' 'unsafe"), "{csp}");
    assert!(csp.contains("frame-ancestors 'none'") && csp.contains("default-src 'none'"));
    assert_eq!(page.headers()["x-frame-options"], "DENY");
    let html = page.text().await.unwrap();
    assert!(!html.contains("https://") && !html.contains("unpkg") && !html.contains("cdn"), "o HTML não pode depender de CDN");
    assert!(html.contains("swagger-ui"));

    for asset in ["/docs/swagger-ui-bundle.js", "/docs/swagger-ui.css", "/docs/oauth2-redirect.html", "/docs/swagger-initializer.js"] {
        let r = app.http.get(app.url(asset)).send().await.unwrap();
        assert_eq!(r.status(), 200, "{asset}");
    }
    // O restante da API mantém a CSP restrita.
    let health = app.http.get(app.url("/healthz")).send().await.unwrap();
    assert_eq!(health.headers()["content-security-policy"], "default-src 'none'; frame-ancestors 'none'");
}

#[tokio::test]
async fn swagger_authorize_button_works_end_to_end() {
    let app = docs_app().await;
    // O cliente do Swagger UI foi registrado sozinho, só com a própria origem como redirect.
    let uris: Vec<String> = sqlx::query_scalar("SELECT unnest(redirect_uris) FROM oauth_clients WHERE client_id = 'swagger-ui'").fetch_all(&app.db).await.unwrap();
    assert_eq!(uris, vec![format!("{ISSUER}/docs/oauth2-redirect.html")]);

    let (verifier, challenge) = pkce();
    let redirect = format!("{ISSUER}/docs/oauth2-redirect.html");
    let q = [
        ("response_type", "code"),
        ("client_id", "swagger-ui"),
        ("redirect_uri", redirect.as_str()),
        ("scope", "ocr profile"),
        ("state", "abc"),
        ("code_challenge", challenge.as_str()),
        ("code_challenge_method", "S256"),
    ];
    let page = app.http.get(app.url("/oauth/authorize")).query(&q).send().await.unwrap();
    assert_eq!(page.status(), 200);
    let cookie = page.headers()["set-cookie"].to_str().unwrap().to_string();
    let csrf = cookie.split(';').next().unwrap().split_once('=').unwrap().1.to_string();

    let mut form: Vec<(&str, String)> = q.iter().map(|(k, v)| (*k, v.to_string())).collect();
    form.extend([("email", EMAIL.to_string()), ("password", PASSWORD.to_string()), ("csrf", csrf.clone())]);
    let r = app.http.post(app.url("/oauth/authorize")).header("cookie", format!("csrf={csrf}")).form(&form).send().await.unwrap();
    assert_eq!(r.status(), 303);
    let loc = url::Url::parse(r.headers()["location"].to_str().unwrap()).unwrap();
    assert_eq!(loc.path(), "/docs/oauth2-redirect.html");
    let code = loc.query_pairs().find(|(k, _)| k == "code").unwrap().1.to_string();

    let t = app
        .http
        .post(app.url("/oauth/token"))
        .form(&[("grant_type", "authorization_code"), ("code", code.as_str()), ("redirect_uri", redirect.as_str()), ("client_id", "swagger-ui"), ("code_verifier", verifier.as_str())])
        .send()
        .await
        .unwrap();
    assert_eq!(t.status(), 200);
}

#[tokio::test]
async fn docs_respect_rate_limits_and_body_limits() {
    let app = TestApp::start_with(|c| {
        c.enable_docs = true;
        c.rate_general_per_min = 30;
    })
    .await;
    let mut limited = false;
    for _ in 0..40 {
        if app.http.get(app.url("/openapi.json")).send().await.unwrap().status() == 429 {
            limited = true;
            break;
        }
    }
    assert!(limited, "/openapi.json também passa pelo limite geral por IP");
}
