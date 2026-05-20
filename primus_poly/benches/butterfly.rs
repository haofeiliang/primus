use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use primus_factor::ShoupFactor;
use primus_poly::DcrtPolynomial;
use rand::distr::{Distribution, Uniform};

type ValueT = u64;

fn bench_butterfly(c: &mut Criterion) {
    let mut rng = rand::rng();

    // Production moduli from SsleParameters.
    let cases: &[(&str, &[ValueT])] = &[
        ("2mod", &[1125899906826241, 1125899906629633]),
        ("3mod", &[137438822401, 137438814209, 137438773249]),
    ];

    let poly_length = 4096;

    for &(label, moduli) in cases {
        let moduli_count = moduli.len();
        let total_len = moduli_count * poly_length;

        // Generate random data per modulus chunk.
        let mut lhs_base = vec![0u64; total_len];
        let mut rhs_base = vec![0u64; total_len];
        let result_base = vec![0u64; total_len];
        let mut w_base = Vec::with_capacity(total_len);

        for (i, &modulus) in moduli.iter().enumerate() {
            let start = i * poly_length;
            let distr = Uniform::new(0, modulus).unwrap();
            for j in 0..poly_length {
                lhs_base[start + j] = distr.sample(&mut rng);
                rhs_base[start + j] = distr.sample(&mut rng);
                let w_val = distr.sample(&mut rng);
                w_base.push(ShoupFactor::new(w_val, modulus));
            }
        }

        // Keep a pristine copy of lhs (butterfly overwrites it in-place).
        let lhs_pristine = lhs_base.clone();

        let mut lhs = DcrtPolynomial(lhs_base);
        let mut result = DcrtPolynomial(result_base);
        let rhs = DcrtPolynomial(rhs_base);

        c.bench_function(
            &format!("butterfly_mul_factor/{label}/n={poly_length}"),
            |b| {
                b.iter(|| {
                    // Restore lhs from pristine copy (butterfly modifies in-place).
                    lhs.0.as_mut_slice().copy_from_slice(&lhs_pristine);
                    result.0.as_mut_slice().fill(0);

                    lhs.butterfly_mul_factor_inplace(
                        black_box(&rhs),
                        black_box(&w_base),
                        black_box(&mut result),
                        poly_length,
                        black_box(moduli),
                    );
                });
            },
        );
    }
}

criterion_group!(benches, bench_butterfly);
criterion_main!(benches);
