use std::process;

use particleanimatorrust::run;
use process::exit;

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {}", e);
        exit(1);
    }
}
