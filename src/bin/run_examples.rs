use std::env;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::Command;

fn is_interactive_example(source: &str) -> bool {
    let lowered = source.to_lowercase();
    lowered.contains("read_line(")
        || lowered.contains("stdin()")
        || lowered.contains("requires_approval(")
        || lowered.contains("prompt(")
}

fn find_expected_output(source: &str) -> Option<String> {
    // Convention: lines starting with `//! Expected:` or `// Expected:` form the expected output block
    let mut lines = source.lines().peekable();
    let mut buf = String::new();
    let mut in_expected = false;

    while let Some(line) = lines.next() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("//! Expected:") || trimmed.starts_with("// Expected:") {
            in_expected = true;
            // capture this line's content after the marker
            if let Some(idx) = trimmed.find(":") {
                buf.push_str(trimmed[idx + 1..].trim_start());
                buf.push('\n');
            }
            // continue capturing subsequent comment lines until a blank line or non-comment
            while let Some(peek) = lines.peek() {
                let t = peek.trim_start();
                if t.starts_with("//! ") || t.starts_with("// ") {
                    let content = t.trim_start_matches("//! ").trim_start_matches("// ");
                    buf.push_str(content);
                    buf.push('\n');
                    lines.next();
                } else if t.is_empty() {
                    lines.next();
                    break;
                } else {
                    break;
                }
            }
            break;
        }
    }

    if in_expected && !buf.trim().is_empty() {
        Some(buf)
    } else {
        None
    }
}

fn list_example_files(root: &Path) -> io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let examples_dir = root.join("examples");
    if !examples_dir.exists() {
        return Ok(out);
    }
    for entry in fs::read_dir(examples_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

fn example_name_from_path(path: &Path) -> Option<String> {
    path.file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
}

fn main() -> io::Result<()> {
    let crate_root = env::current_dir()?;
    let example_paths = list_example_files(&crate_root)?;

    println!("Running {} examples...", example_paths.len());
    println!("============================================================");

    for path in example_paths {
        let name = match example_name_from_path(&path) {
            Some(n) => n,
            None => continue,
        };

        // Read the example source to decide skip and expected text
        let mut source = String::new();
        fs::File::open(&path)?.read_to_string(&mut source)?;

        if is_interactive_example(&source) {
            println!(
                "[SKIP] example '{}' at {} (appears interactive)",
                name,
                path.display()
            );
            println!("------------------------------------------------------------");
            continue;
        }

        let expected =
            find_expected_output(&source).unwrap_or_else(|| "(not specified)".to_string());

        println!("[BEGIN] example: '{}'\n[PATH] {}", name, path.display());
        println!("[EXPECTED]\n{}", expected.trim());

        let mut cmd = Command::new("cargo");
        cmd.arg("run")
            .arg("--example")
            .arg(&name)
            .arg("--quiet")
            .current_dir(&crate_root)
            .envs(env::vars());

        let output = cmd.output()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        println!("[OUTPUT]\n{}", stdout.trim());
        if !stderr.trim().is_empty() {
            println!("[STDERR]\n{}", stderr.trim());
        }
        println!(
            "[STATUS] {}",
            if output.status.success() {
                "ok"
            } else {
                "error"
            }
        );
        println!("[END] example: '{}'", name);
        println!("============================================================");
    }

    Ok(())
}
