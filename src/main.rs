use kmeans::*;
mod bic;
mod read_csv;

fn main() {
    let mut args = std::env::args();
    if args.len() < 2 {
        println!("Usage: xmeans <data_file_path> [-k <number>] [-mink <number] [--delim <string>]");
        std::process::exit(1);
    }
    let data_file_path = args.nth(1).unwrap();
    if !data_file_path.ends_with(".csv") {
        println!("Error: file ({}) is not a csv file", data_file_path);
        std::process::exit(1);
    }

    let mut k = 0;
    let mut min_k = 1;
    let mut delim = b',';
    while let Some(arg) = args.next() {
        match arg.as_str() {
            // check for -k <number> flag
            "-k" | "--k" => {
                k = args.next().unwrap().parse().unwrap();
            }
            "--mink" | "-mink" | "--mk" | "-mk" => {
                min_k = args.next().unwrap().parse().unwrap();
            }
            // check for --delim <string> flag
            "--delim" => {
                delim = args.next().unwrap().as_bytes()[0];
            }
            _ => {
                println!("Error: unknown flag {}", arg);
                std::process::exit(1);
            }
        }
    }

    // read the data file into flat numeric vector
    let (data, shape) = read_csv::read_csv_data(&data_file_path, delim);
    if k > 0 {
        // if user specified specific k, run kmeans
        run_kmeans(&data, shape, k);
    } else {
        run_xmeans(&data, shape, min_k);
    };
}

fn run_kmeans(data: &Vec<f64>, shape: usize, k: usize) {
    let data_len = data.len() / shape;
    let kmean = KMeans::new(data.clone(), data_len, shape);
    let result = kmean.kmeans_lloyd(k, 100, KMeans::init_kmeanplusplus, &KMeansConfig::default());
    let wrapped_centroids: Vec<&[f64]> = result.centroids.chunks(shape).collect();
    print_centroids(&wrapped_centroids);
}

fn run_xmeans(data: &Vec<f64>, shape: usize, start_k: usize) {
    let data_len = data.len() / shape;
    println!("Data shape: {} by {}", shape, data_len);
    let kmean = KMeans::new(data.clone(), data_len, shape);
    let result = kmean.kmeans_lloyd(
        start_k,
        100,
        KMeans::init_kmeanplusplus,
        &KMeansConfig::default(),
    );

    let wrapped_centroids: Vec<&[f64]> = result.centroids.chunks(shape).collect();
    print_centroids(&wrapped_centroids);
    let wrapped_data: Vec<&[f64]> = data.chunks(shape).collect();
    let bic = bic::compute_bic(&wrapped_data, &wrapped_centroids, result.assignments);
    println!("Computed BIC: {:?}", bic);
    unimplemented!();
}

fn print_centroids(c: &Vec<&[f64]>) {
    for centroid in c.iter() {
        let fmtted_centroid: Vec<String> = centroid.iter().map(|x| format!("{:.0}", x)).collect();
        // print with comma separated values
        println!("{}", fmtted_centroid.join(","));
    }
}
