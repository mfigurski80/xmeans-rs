use crate::bic::compute_bic;
use kmeans::*;

/// Got some centroids? Returns a bigger set that potentially improves on
/// them! Just train them and check the bic again to make sure
pub fn next_centroids<'a>(
    wrapped_data: &[&'a [f64]],
    state: &kmeans::KMeansState<f64>,
) -> Vec<f64> {
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
        if cluster_data_len <= 2 {
            next_centroids.extend(centroid.iter());
            continue;
        }
        
        // split the cluster and optimize new centroids
        let kmean = KMeans::new(cluster_data.clone(), cluster_data_len, shape);
        let kmeans_result = kmean.kmeans_lloyd(
            2,
            100,
            KMeans::init_random_partition,
            &KMeansConfig::default(),
        );
        let wrapped_cluster_data: Vec<&[f64]> = cluster_data.chunks(shape).collect();
        let old_bic = compute_bic(&wrapped_cluster_data, &state);
        let new_bic = compute_bic(&wrapped_cluster_data, &kmeans_result);
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

/// Got some centroids? Returns the biggest set that improves on them!
/// Comes out fully trained!
pub fn final_centroids(wrapped_data: &[&[f64]], state: kmeans::KMeansState<f64>, limit: usize) -> Vec<f64> {
    println!("Starting with centroids: {:?}", state.centroids);
    let shape = wrapped_data[0].len();
    let len = wrapped_data.len();
    println!("Data shape: {}x{}", shape, len);
    let mut last_state = state;
    for _ in 0..limit {
        println!("Trying to improve on: {:?}", last_state.centroids);
        let next = next_centroids(wrapped_data, &last_state);
        let count = next.len() / shape;
        if count == last_state.centroids.len() {
            break;
        }
        println!("Optimizing {} centroids: {:?}", count, next);
        let kmeans = KMeans::new(wrapped_data.concat(), len, shape);
        let lanes = 8; // TODO: find what kmeans lib expects
        // let setup_fn = |_, state, _| {
            // let set = next.iter().map(|c| {
                // let mut laned = vec![*c; lanes];
                // laned
            // }).collect::<Vec<Vec<f64>>>().concat();
            // println!("Setting centroids: {:?}", set);
            // state.centroids = set;
        // };
        last_state =
            kmeans.kmeans_lloyd(count, 100, |km, state, conf| {
              let set = next.iter().map(|c| {
                let mut laned = vec![0.0; lanes];
                laned[0] = *c;
                laned
              }).collect::<Vec<Vec<f64>>>().concat();
              state.centroids = set;
              println!("Other: {:?}", km.sample_dims);
              println!("Setting state: {:?}", state);
              println!("Setting conf: {:?}", conf);
            }, &KMeansConfig::default());
    }
    last_state.centroids
}
