use csv::ReaderBuilder;
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
    let (data, shape) = read_csv_data(&data_file_path);
    let data_len = data.len() / shape;
    println!("Data shape: {} by {}", shape, data_len);

    // kmeans
    let k = 5;
    let kmean = KMeans::new(data.clone(), data_len, shape);
    let result = kmean.kmeans_lloyd(k, 100, KMeans::init_kmeanplusplus, &KMeansConfig::default());

    let wrapped_centroids: Vec<&[f64]> = result.centroids.chunks(shape).collect();
    println!("wrapped centroids: {:?}", wrapped_centroids);
    let wrapped_data: Vec<&[f64]> = data.chunks(shape).collect();
    println!("wrapped data: {:?}", wrapped_data);
    println!("Assignments: {:?}", result.assignments);
    let sse = compute_bic(&wrapped_data, &wrapped_centroids, result.assignments);
    println!("err {}", result.distsum);
    println!("Computed BIC: {:?}", sse);
}

fn compute_sse(data: &Vec<&[f64]>, centroids: &Vec<&[f64]>, assignments: Vec<usize>) -> f64 {
    let mut sse = 0.0;
    for (point, assignment) in data.into_iter().zip(assignments.into_iter()) {
        let assigned = centroids[assignment];
        let mut err = 0.0;
        for i in 0..assigned.len() {
            err += f64::abs(point[i] - assigned[i]);
        }
        println!("sse for point {:?}: {}", point, err);
        sse += err
    }
    sse
}

fn compute_bic(data: &Vec<&[f64]>, centroids: &Vec<&[f64]>, assignments: Vec<usize>) -> f64 {
    let sse = compute_sse(&data, centroids, assignments);
    let data_len = data.len() as f64;
    let k = centroids.len() as f64;
    f64::ln(sse / data_len) + 2.0 * k * f64::ln(data_len) / data_len
}

fn read_csv_data(file_path: &str) -> (Vec<f64>, usize) {
    let file_content = std::fs::read_to_string(file_path).unwrap();
    let mut reader = ReaderBuilder::new()
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
