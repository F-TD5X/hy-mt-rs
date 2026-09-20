use anyhow::Result;
use hy_mt_rs::quant::{self, DType};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    dtype: String,
    encoded_hex: String,
    expected_f32_bits: Vec<u32>,
}

#[test]
fn all_decoders_match_pinned_independent_c_oracles() -> Result<()> {
    let cases: Vec<Fixture> = serde_json::from_str(include_str!("fixtures/quant.json"))?;
    for case in cases {
        let dtype = match case.dtype.as_str() {
            "Q8_0" => DType::Q8_0,
            "Q4_K" => DType::Q4_K,
            "Q6_K" => DType::Q6_K,
            "Q2_0C" => DType::Q2_0C,
            "STQ1_0" => DType::STQ1_0,
            _ => unreachable!(),
        };
        let encoded = (0..case.encoded_hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&case.encoded_hex[i..i + 2], 16))
            .collect::<Result<Vec<_>, _>>()?;
        let mut decoded = vec![0.; case.expected_f32_bits.len()];
        quant::decode(dtype, &encoded, &mut decoded)?;
        for (i, (&actual, &expected)) in decoded.iter().zip(&case.expected_f32_bits).enumerate() {
            assert_eq!(actual.to_bits(), expected, "{} lane {i}", case.dtype);
        }
        let input: Vec<_> = (0..decoded.len())
            .map(|i| (i as f32 * 0.17).sin())
            .collect();
        let oracle_dot: f64 = case
            .expected_f32_bits
            .iter()
            .zip(&input)
            .map(|(&w, &x)| f32::from_bits(w) as f64 * x as f64)
            .sum();
        let actual = quant::dot(&decoded, &input) as f64;
        assert!(
            (oracle_dot - actual).abs() < 1e-3 + 1e-4 * oracle_dot.abs(),
            "{} dot",
            case.dtype
        );
    }
    Ok(())
}
