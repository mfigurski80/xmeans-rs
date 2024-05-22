fn compute_distance(x: &[f64], y: &[f64]) -> f64 {
    let mut dist = 0.0;
    for i in 0..x.len() {
        dist += (x[i] - y[i]).powi(2);
    }
    f64::sqrt(dist)
}

/** Follow reduced ll equation from Pelleg and Moore (2000) */
fn compute_group_ll(errors: Vec<f64>, r: usize, k: usize, m: usize) -> f64 {
    let r_n = errors.len() as f64;
    let r = r as f64;
    let m = m as f64;
    let k = k as f64;
    let variance = errors.iter().fold(0.0, |acc, x| acc + x.powi(2)) / (r_n - k);
    -r_n / 2.0 * f64::log10(2.0 * std::f64::consts::PI)
        - r_n * m * f64::log10(variance) / 2.0
        - (r_n - k) / 2.0
        + r_n * f64::log10(r_n)
        - r_n * f64::log10(r)
}

pub fn compute_bic(data: &Vec<&[f64]>, centroids: &Vec<&[f64]>, assignments: Vec<usize>) -> f64 {
    let errors = assignments
        .into_iter()
        .map(|assigned| centroids[assigned])
        .zip(data.into_iter())
        .map(|(mu, x)| compute_distance(mu, x));
    let len = data.len();
    let k = centroids.len();
    let ll = compute_group_ll(errors.collect(), len, k, data[0].len());
    let p_j = (k * data[0].len() + k - 1) as f64;
    let bic = ll - p_j * f64::log10(len as f64) / 2.0;
    bic
}
