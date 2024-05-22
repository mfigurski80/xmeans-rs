use kmeans::*;
mod bic;
mod read_csv;

fn main() {
    let mut args = std::env::args();
    if args.len() < 2 {
        println!("Usage: xmeans <data_file_path> [-k <number>]");
        std::process::exit(1);
    }
    let data_file_path = args.nth(1).unwrap();
    if !data_file_path.ends_with(".csv") {
        println!("Error: file ({}) is not a csv file", data_file_path);
        std::process::exit(1);
    }

    // read the data file into flat numeric vector
    let (data, shape) = read_csv::read_csv_data(&data_file_path, b';');

    // check for -k <number> flag
    let mut k = 0;
    while let Some(arg) = args.next() {
        if arg == "-k" || arg == "--k" {
            k = args.next().unwrap().parse().unwrap();
            break;
        }
    }
    if k > 0 {
        run_kmeans(&data, shape, k);
    } else {
        run_xmeans(&data, shape);
    };
}

fn run_kmeans(data: &Vec<f64>, shape: usize, k: usize) {
    let data_len = data.len() / shape;
    let kmean = KMeans::new(data.clone(), data_len, shape);
    let result = kmean.kmeans_lloyd(k, 100, KMeans::init_kmeanplusplus, &KMeansConfig::default());
    let wrapped_centroids: Vec<&[f64]> = result.centroids.chunks(shape).collect();
    for (_, centroid) in wrapped_centroids.iter().enumerate() {
        let fmtted_centroid: Vec<String> = centroid.iter().map(|x| format!("{:.0}", x)).collect();
        // print with comma separated values
        println!("{}", fmtted_centroid.join(", "));
    }
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
