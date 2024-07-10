use crate::bic::compute_bic;
use kmeans::*;

/// Got some centroids? Returns a bigger set that potentially improves on
/// them! Just train them and check the bic again to make sure
pub fn next_centroids<'a>(
    wrapped_data: &Vec<&'a [f64]>,
    state: kmeans::KMeansState<f64>,
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
        // attempt to split the cluster
        let kmean = KMeans::new(cluster_data.clone(), cluster_data_len, shape);
        let kmeans_result = kmean.kmeans_lloyd(
            2,
            100,
            KMeans::init_random_partition,
            &KMeansConfig::default(),
        );
        let wrapped_cluster_data: Vec<&[f64]> = cluster_data.chunks(shape).collect();
        let old_bic = compute_bic(
            &wrapped_cluster_data,
            &vec![centroid],
            vec![0; cluster_data.len()],
        );
        let new_bic = compute_bic(
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
