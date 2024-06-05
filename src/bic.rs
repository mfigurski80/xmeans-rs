use statrs::distribution::{Continuous, Normal};

fn compute_distance(x: &[f64], y: &[f64]) -> f64 {
    let mut dist = 0.0;
    for i in 0..x.len() {
        dist += (x[i] - y[i]).powi(2);
    }
    f64::sqrt(dist)
}

/** Follow reduced ll equation from Pelleg and Moore (2000) */
fn compute_group_ll(errors: Vec<f64>, full_len: usize, k: usize) -> f64 {
    let len = errors.len() as f64;
    let variance = errors.iter().fold(0.0, |acc, x| acc + x.powi(2)) / (len - k as f64);
    let distribution = Normal::new(0.0, variance).unwrap();
    let ll = errors
        .iter()
        .map(|x| distribution.ln_pdf(*x))
        .fold(0.0, |acc, x| acc + x)
        + len * f64::ln(len / full_len as f64);
    f64::ln(-ll)
}

pub fn compute_bic(data: &Vec<&[f64]>, centroids: &Vec<&[f64]>, assignments: Vec<usize>) -> f64 {
    let errors = assignments
        .into_iter()
        .map(|assigned| centroids[assigned])
        .zip(data.into_iter())
        .map(|(mu, x)| compute_distance(mu, x))
        .collect::<Vec<f64>>();
    let len = data.len();
    let k = centroids.len();
    let ll = compute_group_ll(errors, len, k);
    let p_j = (k * data[0].len() + k) as f64;
    let bic = ll - p_j * f64::ln(len as f64) / 2.0;
    bic
}
