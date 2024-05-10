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
    let file_content = std::fs::read_to_string(&data_file_path).unwrap();
    let mut reader = ReaderBuilder::new().from_reader(file_content.as_bytes());
    let shape = reader.records().next().unwrap().unwrap().len();
    println!("Data shape: {}", shape);
    let mut data = Vec::<f64>::new();
    for record in reader.records() {
        let record = record.unwrap();
        for field in record.iter() {
            data.push(field.parse().unwrap());
        }
    }
    let data_len = data.len() / shape;
    println!("Data length: {}", data.len() / shape);

    // kmeans
    let kmean = KMeans::new(data, data_len, shape);
    let result = kmean.kmeans_lloyd(4, 100, KMeans::init_kmeanplusplus, &KMeansConfig::default());
    println!("Centroids: {:?}", result.centroids);
    println!("Error: {}", result.distsum);
}
