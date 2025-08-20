use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemStruct};

/// Attribute macro that augments a struct with serde::Deserialize and schemars::JsonSchema.
#[proc_macro_attribute]
pub fn tool_args(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemStruct);
    let vis = &input.vis;
    let ident = &input.ident;
    let generics = &input.generics;
    let fields = &input.fields;
    let attrs = &input.attrs;

    let expanded = quote! {
        #[derive(serde::Deserialize, schemars::JsonSchema)]
        #(#attrs)*
        #vis struct #ident #generics #fields
    };
    expanded.into()
}

/// Attribute macro that augments a struct with serde::Serialize and schemars::JsonSchema.
#[proc_macro_attribute]
pub fn tool_output(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemStruct);
    let vis = &input.vis;
    let ident = &input.ident;
    let generics = &input.generics;
    let fields = &input.fields;
    let attrs = &input.attrs;

    let expanded = quote! {
        #[derive(serde::Serialize, schemars::JsonSchema)]
        #(#attrs)*
        #vis struct #ident #generics #fields
    };
    expanded.into()
}
