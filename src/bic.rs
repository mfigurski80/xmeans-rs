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
        .map(|x| 2.0 * distribution.ln_pdf(*x))
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
      // let mean = 2.5;
      let distance = vec![1.5, 0.5, 0.5, 1.5];
      let stddev = compute_stddev(&distance, 0);
      assert_eq!(stddev, 1.118033988749895,"Stddev should be ~1.118, got: {}", stddev);
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
}
