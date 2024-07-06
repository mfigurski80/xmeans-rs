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

    let next = next_centroids(&wrapped_data, kmean_result);
    println!("Next Centroids: {:?}", next);

    unimplemented!();
}

fn print_centroids(c: &Vec<&[f64]>) {
    for centroid in c.iter() {
        let fmtted_centroid: Vec<String> = centroid.iter().map(|x| format!("{:.0}", x)).collect();
        // print with comma separated values
        println!("{}", fmtted_centroid.join(","));
    }
}

/// Got some centroids? Returns a bigger set that potentially improves on
/// them! Just train them and check the bic again to make sure
fn next_centroids<'a>(wrapped_data: &Vec<&'a [f64]>, state: kmeans::KMeansState<f64>) -> Vec<f64> {
    let mut next_centroids: Vec<f64> = vec![];

    let shape = wrapped_data[0].len();
    let centroids = state.centroids.chunks(shape);
    for (cluster_i, centroid) in centroids.enumerate() {
        let cluster_data = state
            .assignments
            .iter()
            .enumerate()
            .filter(|(_, assigned)| **assigned == cluster_i)
            .flat_map(|(i, _)| wrapped_data[i])
            .map(|x| *x)
            .collect::<Vec<f64>>();
        let cluster_data_len = cluster_data.len() / shape;
        // attempt to split the cluster
        let kmean = KMeans::new(cluster_data.clone(), cluster_data_len, shape);
        let kmeans_result = kmean.kmeans_lloyd(
            2,
            100,
            KMeans::init_random_partition,
            &KMeansConfig::default(),
        );
        let wrapped_cluster_data: Vec<&[f64]> = cluster_data.chunks(shape).collect();
        let old_bic = bic::compute_bic(
            &wrapped_cluster_data,
            &vec![centroid],
            vec![0; cluster_data.len()],
        );
        let new_bic = bic::compute_bic(
            &wrapped_cluster_data,
            &kmeans_result.centroids.chunks(shape).collect(),
            kmeans_result.assignments,
        );
        if old_bic > new_bic {
            next_centroids.extend(centroid.iter());
            println!("Old Centroid wins: {:?}", centroid);
        } else {
            next_centroids.extend(kmeans_result.centroids.iter());
            println!("New Centroid wins: {:?}", kmeans_result.centroids);
        }
    }
    next_centroids
}
