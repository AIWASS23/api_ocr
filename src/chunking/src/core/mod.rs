//! Peças compartilhadas pelas estratégias (não são estratégias).

#[cfg_attr(not(feature = "html"), allow(dead_code))]
pub mod html;
pub mod languages;
pub mod numeric;
pub mod options;
pub mod pyre;
pub mod pyjson;
pub mod pystr;
pub mod section;
pub mod splitter;
pub mod tokens;
