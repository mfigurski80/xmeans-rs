use kmeans::*;
mod bic;
mod read_csv;

fn main() {
    let mut args = std::env::args();
    if args.len() < 2 {
        println!("Usage: cargo run <data_file_path>");
        std::process::exit(1);
    }
    let data_file_path = args.nth(1).unwrap();
    if !data_file_path.ends_with(".csv") {
        println!("Error: file ({}) is not a csv file", data_file_path);
        std::process::exit(1);
    }

    // read the data file into flat numeric vector
    let (data, shape) = read_csv::read_csv_data(&data_file_path, b';');
    run_xmeans(&data, shape);
}

fn run_xmeans(data: &Vec<f64>, shape: usize) {
    let data_len = data.len() / shape;
    println!("Data shape: {} by {}", shape, data_len);
    // kmeans
    let k = 30;
    let kmean = KMeans::new(data.clone(), data_len, shape);
    let result = kmean.kmeans_lloyd(k, 100, KMeans::init_kmeanplusplus, &KMeansConfig::default());

    let wrapped_centroids: Vec<&[f64]> = result.centroids.chunks(shape).collect();
    println!("wrapped centroids: {:?}", wrapped_centroids);
    let wrapped_data: Vec<&[f64]> = data.chunks(shape).collect();
    let bic = bic::compute_bic(&wrapped_data, &wrapped_centroids, result.assignments);
    println!("Computed BIC: {:?}", bic);
    // not implemented
    unimplemented!();
}
