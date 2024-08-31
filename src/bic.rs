use kmeans::KMeansState;
use statrs::distribution::{Continuous, Normal};

fn compute_distance(x: &[f64], y: &[f64]) -> f64 {
    let mut dist = 0.0;
    for i in 0..x.len() {
        dist += (x[i] - y[i]).powi(2);
    }
    f64::sqrt(dist)
}

fn compute_stddev(resid: &[f64], free_params: usize) -> f64 {
  let free = free_params as f64;
  let len = resid.len() as f64;
  if len <= free {
    return f64::INFINITY;
  }
  return f64::sqrt(resid.iter().map(|v| v.powi(2)).sum::<f64>() / (len - free));
}

fn build_errors(data: &[&[f64]], model: &KMeansState<f64>) -> Vec<f64> {
    let shape = data[0].len();
    let wrapped_centroids: Vec<&[f64]> = model.centroids.chunks_exact(shape).collect();
    let assignments = &model.assignments;
    let errors = assignments
        .into_iter()
        .map(|assigned| wrapped_centroids[*assigned])
        .zip(data.into_iter())
        .map(|(mu, x)| compute_distance(mu, x))
        .collect::<Vec<f64>>();
   errors
}

fn compute_free_params(model: &KMeansState<f64>) -> usize {
    let shape = model.centroids.len() / model.k;
    (model.k - 1) + (shape * model.k) + 1
}

/// Follow reduced ll equation from Pelleg and Moore (2000)
fn compute_group_ll(errors: Vec<f64>, free_params: usize) -> f64 {
    // println!("errors: {:?}", errors);
    let std_dev = compute_stddev(&errors, free_params);
    // println!("std_dev: {:?}", std_dev);
    if std_dev == f64::INFINITY {
      return 0.0;
    }
    let distribution = Normal::new(0.0, std_dev).unwrap();
    // println!("Distribution: {:?}", distribution);
    let ll = errors
        .iter()
        // folded distribution: double ll
        .map(|x| f64::ln(2.0) + distribution.ln_pdf(*x))
        .sum::<f64>();
    // println!("LL: {:?}", ll);
    ll
}

/// Compute grouping BIC according to example set by Pelleg and Moore (2000)
pub fn compute_bic(data: &[&[f64]], model: &KMeansState<f64>) -> f64 {
    let len = data.len() as f64;
    let errors = build_errors(data, model);
    let free = compute_free_params(model);
    let ll = compute_group_ll(errors, free);
    let bic = free as f64 * f64::ln(len) - 2.0 * ll;
    bic
}

#[cfg(test)]
mod tests {
    use super::*;
    use kmeans::*;

    #[test]
    fn it_computes_distance() {
      let x = vec![1.0, 2.0, 3.0];
      let y = vec![4.0, 5.0, 6.0];
      let dist = compute_distance(&x, &y);
      assert_eq!(dist, 5.196152422706632, "Distance should be ~5.196, got: {}", dist);
    }

    #[test]
    fn it_computes_stddev() {
      // let data = vec![1.0, 2.0, 3.0, 4.0];
      // let data_mean = 2.5;
      let distance = vec![1.5, 0.5, 0.5, 1.5];
      let stddev = compute_stddev(&distance, 0);
      assert_eq!(stddev, 1.118033988749895,"Stddev should be ~1.118, got: {}", stddev);
    }

    #[test]
    fn it_computes_likelihood() {
      // let data = vec![1.0, 2.0, 3.0, 4.0];
      // let data_mean = 2.5;
      let distance = vec![1.5, 0.5, 0.5, 1.5];
      // let distribution = Normal::new(0, 1.118)
      // let approx_pdf = vec![0.322, 0.145, 0.145, 0.322];
      // let approx_folded_pdf = vec![0.644, 0.290, 0.290, 0.644]
      // let approx_multiplied = 0.0348
      let l = f64::exp(compute_group_ll(distance, 0));
      assert!(l > 0.0 && l < 1.0, "Likelihood should be in valid range, got: {}", l);
      assert_eq!(l, 0.03510356758042785, "Likelihood should be ~.035, got: {}", l);
    }

    #[test]
    fn it_computes_bic_in_range() {
      let data = vec![1.0, 2.0, 3.0, 4.0];
      let state = KMeans::<_, 1>::new(data.clone().into(), 4, 1)
        .kmeans_lloyd(2, 10, KMeans::init_random_partition, &KMeansConfig::default());
      let wrapped: Vec<&[f64]> = data.chunks(1).collect();
      let bic = compute_bic(&wrapped, &state);
      assert!(bic > 0.0, "BIC should be > 0, got: {}", bic);
      assert!(bic < 10.0, "BIC should be < 10, got: {}", bic);
    }

    #[test]
    fn it_computes_bic_penalizes_complexity() {
      let data = vec![1.0, 1.1, 3.0, 3.1]; // note 2 clusters
      let wrapped: Vec<&[f64]> = data.chunks(1).collect();

      let good_state= KMeans::<_, 1>::new(data.clone(), 4, 1)
        .kmeans_lloyd(2, 10, KMeans::init_random_partition, &KMeansConfig::default());
      let good_bic = compute_bic(&wrapped, &good_state);

      let complex_state = KMeans::<_, 1>::new(data.clone(), 4, 1)
        .kmeans_lloyd(3, 10, KMeans::init_random_partition, &KMeansConfig::default());
      let complex_bic = compute_bic(&wrapped, &complex_state);

      assert!(good_bic < complex_bic, "BIC should discount complexity (good: {}, complex: {})", good_bic, complex_bic);
    }

    #[test]
    fn it_computes_bic_penalizes_error() {
      let data = vec![1.0, 1.1, 3.0, 3.1, 5.0, 5.1]; // note 3 clusters
      let wrapped: Vec<&[f64]> = data.chunks(1).collect();

      let good_state= KMeans::<_, 1>::new(data.clone(), 6, 1)
        .kmeans_lloyd(3, 10, KMeans::init_random_partition, &KMeansConfig::default());
      let good_bic = compute_bic(&wrapped, &good_state);

      let error_state = KMeans::<_, 1>::new(data.clone(), 6, 1)
        .kmeans_lloyd(2, 10, KMeans::init_random_partition, &KMeansConfig::default());
      let error_bic = compute_bic(&wrapped, &error_state);

      assert!(good_bic < error_bic, "BIC should discount high error (good: {}, error: {})", good_bic, error_bic);
    }
}
