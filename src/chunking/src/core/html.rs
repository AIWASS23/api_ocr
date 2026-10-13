//! Árvore HTML simples (elementos, textos e comentários) usada pelas estratégias HTML.
//!
//! O HTML é lido pelo `scraper` (html5ever, parser HTML5 em Rust) e convertido para esta árvore,
//! que é fácil de percorrer e de modificar. Reproduz o que importa do BeautifulSoup
//! (`html.parser`) para os splitters: comentários contam como texto em `find_all(string=True)`,
//! `get_text()` ignora comentários e o conteúdo de `<script>`/`<style>`, e um documento sem
//! `<body>` é percorrido inteiro (`soup.body or soup`).

#[derive(Debug, Clone)]
pub enum HNode {
    Element(HElement),
    Text(String),
    Comment(String),
}

#[derive(Debug, Clone)]
pub struct HElement {
    pub name: String,
    pub attrs: Vec<(String, String)>,
    pub children: Vec<HNode>,
}

impl HElement {
    pub fn attr(&self, name: &str) -> Option<&str> {
        self.attrs.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }

    /// Filhos que são elementos (`find_all(recursive=False)`).
    pub fn child_elements(&self) -> impl Iterator<Item = &HElement> {
        self.children.iter().filter_map(|c| match c {
            HNode::Element(e) => Some(e),
            _ => None,
        })
    }

    /// Textos diretos, incluindo comentários (`find_all(string=True, recursive=False)`).
    pub fn direct_strings(&self) -> Vec<&str> {
        self.children
            .iter()
            .filter_map(|c| match c {
                HNode::Text(t) | HNode::Comment(t) => Some(t.as_str()),
                HNode::Element(_) => None,
            })
            .collect()
    }

    /// `get_text(strip=True)`: textos (sem comentários, sem script/style) sem espaço nas pontas,
    /// concatenados sem separador.
    pub fn text_stripped(&self) -> String {
        let mut out = String::new();
        self.walk_text(&mut |t| out.push_str(crate::core::pystr::strip(t)));
        out
    }

    fn walk_text(&self, f: &mut dyn FnMut(&str)) {
        if matches!(self.name.as_str(), "script" | "style" | "template") {
            return;
        }
        for c in &self.children {
            match c {
                HNode::Text(t) => f(t),
                HNode::Element(e) => e.walk_text(f),
                HNode::Comment(_) => {}
            }
        }
    }

    /// `.text`/`get_text()` sem strip: textos concatenados (sem comentários, sem script/style).
    pub fn text_content(&self) -> String {
        let mut out = String::new();
        self.walk_text(&mut |t| out.push_str(t));
        out
    }

    /// Todos os textos e comentários em ordem de documento (`next_elements` filtrado para str).
    pub fn all_strings<'a>(&'a self, out: &mut Vec<&'a str>) {
        for c in &self.children {
            match c {
                HNode::Text(t) | HNode::Comment(t) => out.push(t),
                HNode::Element(e) => e.all_strings(out),
            }
        }
    }

    /// Primeiro elemento com o nome dado (busca em profundidade).
    pub fn find(&self, name: &str) -> Option<&HElement> {
        for e in self.child_elements() {
            if e.name == name {
                return Some(e);
            }
            if let Some(f) = e.find(name) {
                return Some(f);
            }
        }
        None
    }
}

#[cfg(feature = "html")]
pub fn parse(html: &str) -> HElement {
    use ego_tree::NodeRef;
    use scraper::{Html, Node};

    fn convert(node: NodeRef<'_, Node>) -> Option<HNode> {
        match node.value() {
            Node::Text(t) => Some(HNode::Text(t.text.to_string())),
            Node::Comment(c) => Some(HNode::Comment(c.comment.to_string())),
            Node::Element(e) => Some(HNode::Element(HElement {
                name: e.name().to_ascii_lowercase(),
                attrs: e.attrs().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
                children: node.children().filter_map(convert).collect(),
            })),
            _ => None,
        }
    }

    let doc = Html::parse_document(html);
    HElement {
        name: "[document]".into(),
        attrs: Vec::new(),
        children: doc.tree.root().children().filter_map(|n| convert(n)).collect(),
    }
}

/// Raiz para percorrer: o `<body>` se o HTML de entrada tiver um, senão o documento inteiro.
pub fn body_or_document<'a>(source: &str, doc: &'a HElement) -> &'a HElement {
    if source.to_ascii_lowercase().contains("<body") {
        doc.find("body").unwrap_or(doc)
    } else {
        doc
    }
}
