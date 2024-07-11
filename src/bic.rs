use kmeans::KMeansState;
use statrs::distribution::{Continuous, Normal};

fn compute_distance(x: &[f64], y: &[f64]) -> f64 {
    let mut dist = 0.0;
    for i in 0..x.len() {
        dist += (x[i] - y[i]).powi(2);
    }
    f64::sqrt(dist)
}

/** Follow reduced ll equation from Pelleg and Moore (2000) */
fn compute_group_ll(errors: Vec<f64>, k: usize) -> f64 {
    let len = errors.len() as f64;
    let std_dev = errors.iter().fold(0.0, |acc, x| acc + x) / (len - k as f64);
    let distribution = Normal::new(0.0, std_dev).unwrap();
    // println!("Distribution: {:?}", distribution);
    let ll = errors
        .iter()
        .map(|x| distribution.ln_pdf(*x))
        .fold(0.0, |acc, x| acc + x);
    // println!("LL: {:?}", ll);
    // folded distribution: double ll
    ll * 2.0
}

// pub fn compute_bic(data: &[&[f64]], centroids: &Vec<&[f64]>, assignments: Vec<usize>) -> f64 {
pub fn compute_bic(data: &[&[f64]], model: &KMeansState<f64>) -> f64 {
    let shape = data[0].len();
    let centroids: Vec<&[f64]> = model.centroids.chunks(shape).collect();
    let assignments = &model.assignments;
    let errors = assignments
        .into_iter()
        .map(|assigned| centroids[*assigned])
        .zip(data.into_iter())
        .map(|(mu, x)| compute_distance(mu, x))
        .collect::<Vec<f64>>();
    let len = data.len() as f64;
    let dim = data[0].len();
    let k = centroids.len();
    let free_params = (k - 1) + (dim * k) + 1;
    let ll = compute_group_ll(errors, free_params);
    let bic = free_params as f64 * f64::ln(len) - 2.0 * ll;
    bic
}
