use kmeans::KMeansState;
use statrs::distribution::{Continuous, Normal};

fn compute_distance(x: &[f64], y: &[f64]) -> f64 {
    let mut dist = 0.0;
    for i in 0..x.len() {
        dist += (x[i] - y[i]).powi(2);
    }
    f64::sqrt(dist)
}

fn compute_stddev(v: &[f64], free_params: usize) -> f64 {
  let free = free_params as f64;
  let len = v.len() as f64;
  if len <= free {
    return f64::INFINITY;
  }
  return v.iter().sum::<f64>() * f64::sqrt(1.0 / (len - free));
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
fn compute_group_ll(errors: Vec<f64>, free_params: usize, model_std_dev: f64) -> f64 {
    // println!("errors: {:?}", errors);
    let std_dev = compute_stddev(&errors, free_params);
    let variance_scaling = f64::ln(model_std_dev / std_dev);
    println!("variance scaling: {:?}", variance_scaling);
    // println!("std_dev: {:?}", std_dev);
    if std_dev == f64::INFINITY {
      return 0.0;
    }
    let distribution = Normal::new(0.0, std_dev).unwrap();
    // println!("Distribution: {:?}", distribution);
    let ll = errors
        .iter()
        // folded distribution: double ll
        .map(|x| 2.0 * distribution.ln_pdf(*x) + variance_scaling)
        .sum::<f64>();
    // println!("LL: {:?}", ll);
    ll
}

/// Compute grouping BIC according to example set by Pelleg and Moore (2000)
pub fn compute_bic(data: &[&[f64]], model: &KMeansState<f64>, model_stddev: f64) -> f64 {
    let len = data.len() as f64;
    let errors = build_errors(data, model);
    let free = compute_free_params(model);
    let ll = compute_group_ll(errors, free, model_stddev);
    let bic = free as f64 * f64::ln(len) - 2.0 * ll;
    bic
}

/// Compute the assumed standard deviation of the model
pub fn compute_model_stddev(data: &[&[f64]], model: &KMeansState<f64>) -> f64 {
    let free = compute_free_params(model);
    let errors = build_errors(data, model);
    let std_dev = compute_stddev(&errors, free);
    std_dev
}
