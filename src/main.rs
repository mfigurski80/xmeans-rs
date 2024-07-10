mod args;
mod bic;
mod read_csv;
mod xmeans;

use kmeans::*;

fn main() {
    let args = match args::parse_args(std::env::args()) {
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
        Ok(args) => args,
    };

    // read the data file into flat numeric vector
    let (data, shape) = read_csv::read_csv_data(args.file_path.as_str(), args.delim);
    if args.k > 0 {
        // if user specified specific k, run kmeans
        run_kmeans(&data, shape, args.k);
    } else {
        run_xmeans(&data, shape, args.min_k);
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
    let kmean = KMeans::new(data.clone(), data_len, shape);
    let kmean_result = kmean.kmeans_lloyd(
        start_k,
        100,
        KMeans::init_kmeanplusplus,
        &KMeansConfig::default(),
    );

    let wrapped_centroids: Vec<&[f64]> = kmean_result.centroids.chunks(shape).collect();
    let wrapped_data: Vec<&[f64]> = data.chunks(shape).collect();

    let bic = bic::compute_bic(
        &wrapped_data,
        &wrapped_centroids,
        kmean_result.assignments.clone(),
    );
    println!("Initial BIC: {:?}", bic);

    loop {
        let next_centroids = xmeans::next_centroids(&wrapped_data, kmean_result);
        println!("Next Centroids: {:?}", next_centroids);
        break;
    }

    unimplemented!();
}

fn print_centroids(c: &Vec<&[f64]>) {
    for centroid in c.iter() {
        let fmtted_centroid: Vec<String> = centroid.iter().map(|x| format!("{:.0}", x)).collect();
        // print with comma separated values
        println!("{}", fmtted_centroid.join(","));
    }
}
