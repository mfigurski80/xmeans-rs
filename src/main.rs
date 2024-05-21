use csv::ReaderBuilder;
use itertools::Itertools;
use kmeans::*;

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
    // stat the data file for size
    let metadata = std::fs::metadata(&data_file_path).unwrap();
    let file_size = metadata.len();
    println!("Data file size: {} b", file_size);

    // read the data file into flat numeric vector
    let (data, shape) = read_csv_data(&data_file_path, b';');
    let data_len = data.len() / shape;
    println!("Data shape: {} by {}", shape, data_len);

    // kmeans
    let k = 5;
    let kmean = KMeans::new(data.clone(), data_len, shape);
    let result = kmean.kmeans_lloyd(k, 100, KMeans::init_kmeanplusplus, &KMeansConfig::default());

    let wrapped_centroids: Vec<&[f64]> = result.centroids.chunks(shape).collect();
    println!("wrapped centroids: {:?}", wrapped_centroids);
    let wrapped_data: Vec<&[f64]> = data.chunks(shape).collect();
    let ll = compute_ll(&wrapped_data, &wrapped_centroids, result.assignments);
    println!("Computed Log Likelihood: {:?}", ll);
}

fn compute_distance(x: &[f64], y: &[f64]) -> f64 {
    let mut dist = 0.0;
    for i in 0..x.len() {
        dist += (x[i] - y[i]).powi(2);
    }
    f64::sqrt(dist)
}

fn compute_group_pp(errors: Vec<f64>) -> Vec<f64> {
    let variance = errors.iter().fold(0.0, |acc, x| acc + x.powi(2)) / errors.len() as f64;
    let scale_factor = 1.0 / (f64::sqrt(2.0 * std::f64::consts::PI * variance));
    let ll = errors
        .iter()
        .map(|e| -e.powi(2) / (2.0 * variance))
        .map(|shift| scale_factor * shift.exp());
    ll.collect()
}

fn compute_ll(data: &Vec<&[f64]>, centroids: &Vec<&[f64]>, assignments: Vec<usize>) -> f64 {
    let assigned = assignments
        .into_iter()
        .zip(data.into_iter())
        .into_group_map();
    let mut ln_pp = 1.0;
    for (k, v) in assigned.iter() {
        let mu = centroids[*k];
        let errors = v.into_iter().map(|x| compute_distance(x, mu));
        let ll = compute_group_pp(errors.collect())
            .into_iter()
            .map(|x| x.ln())
            .sum::<f64>();
        ln_pp *= ll;
    }
    println!("error: {:?}", ln_pp);
    0.0
}

fn read_csv_data(file_path: &str, delim: u8) -> (Vec<f64>, usize) {
    let file_content = std::fs::read_to_string(file_path).unwrap();
    let mut reader = ReaderBuilder::new()
        .delimiter(delim)
        .has_headers(false)
        .from_reader(file_content.as_bytes());
    let mut shape = 0;
    let mut data = Vec::<f64>::new();
    for record in reader.records() {
        let record = record.unwrap();
        if shape == 0 {
            shape = record.len();
        }
        for field in record.iter() {
            data.push(field.parse().unwrap());
        }
    }
    data.shrink_to_fit();
    (data, shape)
}
