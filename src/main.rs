use kmeans::*;
mod args;
mod bic;
mod read_csv;

fn main() {
    let found = args::parse_args(std::env::args());
    if found.is_err() {
        eprintln!("{}", found.err().unwrap());
        std::process::exit(1);
    }
    let args = found.unwrap();

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
        let next_centroids = next_centroids(&wrapped_data, kmean_result);
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
        println!(
            "Comparing centroids: {:?}({}) to {:?}({})?",
            centroid, old_bic, kmeans_result.centroids, new_bic
        );
        if old_bic < new_bic {
            next_centroids.extend(centroid.iter());
        } else {
            next_centroids.extend(kmeans_result.centroids.iter());
        }
    }

    next_centroids
}
