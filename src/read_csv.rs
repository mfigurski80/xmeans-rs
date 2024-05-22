use csv::ReaderBuilder;

pub fn read_csv_data(file_path: &str, delim: u8) -> (Vec<f64>, usize) {
    let file_content = std::fs::read_to_string(file_path).unwrap();
    let mut reader = ReaderBuilder::new()
        .delimiter(delim)
        .has_headers(false)
        .from_reader(file_content.as_bytes());
    let mut shape = 0;
    let mut data = Vec::<f64>::new();
    for record in reader.records() {
        let record = record.unwrap();
        if shape == 0 {
            shape = record.len();
        }
        for field in record.iter() {
            data.push(field.parse().unwrap());
        }
    }
    data.shrink_to_fit();
    (data, shape)
}
