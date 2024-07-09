use std::num::ParseIntError;

#[derive(Debug)]
pub struct ParsedArgs {
    pub k: usize,
    pub min_k: usize,
    pub delim: u8,
    pub file_path: String,
}

#[derive(Debug)]
pub enum ParseArgsError {
    Missing(&'static str),
    Bad(String),
    ParseIntError(ParseIntError),
}

type ParseArgsResult = Result<ParsedArgs, ParseArgsError>;

/// Build a ParsedArgs struct from given command line arguments
pub fn parse_args(args: impl IntoIterator<Item = String>) -> ParseArgsResult {
    let mut fin = ParsedArgs {
        k: 0,
        min_k: 2,
        delim: b',',
        file_path: String::new(),
    };

    let mut args_it = args.into_iter().skip(1);
    while let Some(arg) = args_it.next() {
        match arg.as_str() {
            // check for -k <number> flag
            "-k" | "--k" => {
                let next = args_it
                    .next()
                    .ok_or(ParseArgsError::Missing("k needs value"))?;
                fin.k = next.parse().map_err(ParseArgsError::ParseIntError)?;
            }
            // check for --mink <number> flag
            "--mink" | "-mink" | "--mk" | "-mk" => {
                let next = args_it
                    .next()
                    .ok_or(ParseArgsError::Missing("mink needs value"))?;
                fin.min_k = next.parse().map_err(ParseArgsError::ParseIntError)?;
            }
            // check for --delim <string> flag
            "--delim" => {
                let next = args_it
                    .next()
                    .ok_or(ParseArgsError::Missing("delim needs value"))?;
                fin.delim = next.as_bytes()[0];
                if next.len() != 1 {
                    return Err(ParseArgsError::Bad(
                        "delim must be a single character".into(),
                    ));
                }
            }
            _ => {
                if (fin.file_path).is_empty() {
                    fin.file_path = arg;
                } else {
                    let msg = format!("unknown argument {}", arg);
                    return Err(ParseArgsError::Bad(msg));
                }
            }
        }
    }

    if (fin.file_path).is_empty() {
        return Err(ParseArgsError::Missing("file path"));
    }
    if !fin.file_path.ends_with(".csv") {
        let msg = format!("file ({}) is not a csv file", fin.file_path);
        return Err(ParseArgsError::Bad(msg.into()));
    }

    Ok(fin)
}

impl std::fmt::Display for ParseArgsError {
    fn fmt(self: &Self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        let usage_msg =
            "Usage: xmeans <data_file_path> [-k <number>] [--mink <number] [--delim <string>]";
        match self {
            ParseArgsError::Missing(msg) => {
                write!(formatter, "Missing argument: {}\n{}", msg, usage_msg)
            }
            ParseArgsError::Bad(msg) => write!(formatter, "Bad argument: {}\n{}", msg, usage_msg),
            ParseArgsError::ParseIntError(e) => write!(formatter, "Error parsing integer: {}", e),
        }
    }
}
