#pragma once

// VSX SIMD intrinsics implementation.
#define __VEC__
#if defined(__VEC__) || defined(__ALTIVEC__) || defined(__VSX__)
#include <altivec.h>
#include <cmath>
#include <cstdint>

#include <iostream>
#include <arbor/simd/approx.hpp>
#include <arbor/simd/implbase.hpp>

namespace arb {
    namespace simd {
        namespace detail {

            struct vsx_double4;
            struct vsx_int2x2;
            struct vsx_int4;
            struct vsx_bool2x2;

            template <>
            struct simd_traits<vsx_double4> {
                static constexpr unsigned width = 4;
                using scalar_type = double;
                using vector_type = std::array<vector double, 2>;
                using mask_impl   = vsx_bool2x2;
            };

            template <>
            struct simd_traits<vsx_int4> {
                static constexpr unsigned width = 4;
                using scalar_type = int;
                using vector_type = vector int;
                using mask_impl   = vsx_int4;
            };

            template <>
            struct simd_traits<vsx_bool2x2> {
                static constexpr unsigned width = 4;
                using scalar_type = bool;
                using vector_type = std::array<vector bool long long, 2>;
                using mask_impl   = vsx_bool2x2;
            };


            struct vsx_int4 : implbase<vsx_int4> {
                using array = vector int;

                static void copy_to(const array& v, int* p) {
                    p[0] = v[0];
                    p[1] = v[1];
                    p[2] = v[2];
                    p[3] = v[3];
                }
                static array copy_from(const int* p) {
                    array result;
                    result[0] = p[0];
                    result[1] = p[1];
                    result[2] = p[2];
                    result[3] = p[3];
                    return result;
                }
                static void mask_copy_to(const array& v, bool* w) {
                    w[0] = v[0];
                    w[1] = v[1];
                    w[2] = v[2];
                    w[3] = v[3];
                }

                static array mask_copy_from(const bool* y) {
                    array v;
                    v[0] = y[0];
                    v[1] = y[1];
                    v[2] = y[2];
                    v[3] = y[3];
                    return v;
                }

                static bool mask_element(const array& v, int i) {
                    return static_cast<bool>(v[i]);
                }

                static void mask_set_element(array& v, int i, bool x) {
                    v[i] = x;
                }

                static array add(const array& a, const array& b) {
                    array result;
                    result = a + b;
                    return result;
                }

                static array sub(const array& a, const array& b) {
                    array result;
                    result = a - b;
                    return result;
                }

                static array mul(const array& a, const array& b) {
                    array result;
                    result = a * b;
                    return result;
                }

                static array div(const array& a, const array& b) {
                    array result;
                    result = a / b;
                    return result;
                }

                static array and(const array& a, const array& b) {
                    array result;
                    result = a & b;
                    return result;
                }

                static array or(const array& a, const array& b) {
                    array result;
                    result = a | b;
                    return result;
                }

                static array xor(const array& a, const array& b) {
                    array result;
                    result = a ^ b;
                    return result;
                }

                static array shift_r(const array& a, const array& b) {
                    array result;
                    result = a >> b;
                    return result;
                }

                static array shift_l(const array& a, const array& b) {
                    array result;
                    result = a << b;
                    return result;
                }

                static array min(const array& a, const array& b) {
                    array result;
                    result = vec_min(a, b);
                    return result;
                }

                static array max(const array& a, const array& b) {
                    array result;
                    result = vec_max(a, b);
                    return result;
                }

                static array abs(const array& a) {
                    array result;
                    result = vec_abs(a);
                    return result;
                }
            };

            struct vsx_bool2x2  : implbase<vsx_bool2x2> {
                using array = std::array<vector bool long long, 2>;

                static void copy_to(const array& v, double* p) {
                    p[0] = v[0][0];
                    p[1] = v[0][1];
                    p[2] = v[1][0];
                    p[3] = v[1][1];
                }

                static array copy_from(const double* p) {
                    array result;
                    result[0][0] = p[0];
                    result[0][1] = p[1];
                    result[1][0] = p[2];
                    result[1][1] = p[3];
                    return result;
                }

                static void mask_copy_to(const array& v, bool* p) {
                    p[0] = v[0][0];
                    p[1] = v[0][1];
                    p[2] = v[1][0];
                    p[3] = v[1][1];
                }

                static array mask_copy_from(const bool* p) {
                    array result;
                    result[0][0] = p[0] ? (~0) : 0;
                    result[0][1] = p[1] ? (~0) : 0;
                    result[1][0] = p[2] ? (~0) : 0;
                    result[1][1] = p[3] ? (~0) : 0;
                    return result;
                }

                static bool mask_element(const array& v, int i) {
                    auto s = i & 1;
                    auto k = (i >> 1) & 1;
                    return static_cast<bool>(v[k][s]);
                }

                static void mask_set_element(array& v, int i, bool x) {
                    auto s = i & 1;
                    auto k = (i >> 1) & 1;
                    v[k][s] = x ? (~0) : 0;
                }
            };
            
            struct vsx_double4 : implbase<vsx_double4> {
                using f64x2x2 = std::array<vector double, 2>;
                using i64x2x2 = std::array<vector long long, 2>;
                using u64x2x2 = std::array<vector unsigned long, 2>;
                using b64x2x2 = std::array<vector bool long long, 2>;
                using i32x4   = vector int;

                using array = f64x2x2;
                using ints  = i32x4;
                using bools = b64x2x2;
                
                static void copy_to(const array& v, double* p) {
                    p[0] = v[0][0];
                    p[1] = v[0][1];
                    p[2] = v[1][0];
                    p[3] = v[1][1];
                }

                static array copy_from(const double* p) {
                    array result;
                    result[0][0] = p[0];
                    result[0][1] = p[1];
                    result[1][0] = p[2];
                    result[1][1] = p[3];
                    return result;
                }

                static array add(const array& a, const array& b) {
                    array result;
                    result[0] = a[0] + b[0];
                    result[1] = a[1] + b[1];
                    return result;
                }

                static array madd(const array& a, const array& b, const array& c) {
                    array result;
                    result[0] = vec_madd(a[0], b[0], c[0]);
                    result[1] = vec_madd(a[1], b[1], c[1]);
                    return result;
                }

                static array sub(const array& a, const array& b) {
                    array result;
                    result[0] = a[0] - b[0];
                    result[1] = a[1] - b[1];
                    return result;
                }

                static array msub(const array& a, const array& b, const array& c) {
                    array result;
                    result[0] = vec_msub(a[0], b[0], c[0]);
                    result[1] = vec_msub(a[1], b[1], c[1]);
                    return result;
                }

                static array mul(const array& a, const array& b) {
                    array result;
                    result[0] = a[0] * b[0];
                    result[1] = a[1] * b[1];
                    return result;
                }

                static array div(const array& a, const array& b) {
                    array result;
                    result[0] = a[0] / b[0];
                    result[1] = a[1] / b[1];
                    return result;
                }

                static array min(const array& a, const array& b) {
                    array result;
                    result[0] = vec_min(a[0], b[0]);
                    result[1] = vec_min(a[1], b[1]);
                    return result;
                }

                static array and(const array& a, const array& b) {
                    array result;
                    result[0] = vec_and(a[0], b[0]);
                    result[1] = vec_and(a[1], b[1]);
                    return result;
                }

                static array or(const array& a, const array& b) {
                    array result;
                    result[0] = vec_or(a[0], b[0]);
                    result[1] = vec_or(a[1], b[1]);
                    return result;
                }

                static array xor(const array& a, const array& b) {
                    array result;
                    result[0] = vec_xor(a[0], b[0]);
                    result[1] = vec_xor(a[1], b[1]);
                    return result;
                }

                static array max(const array& a, const array& b) {
                    array result;
                    result[0] = vec_max(a[0], b[0]);
                    result[1] = vec_max(a[1], b[1]);
                    return result;
                }

                static array abs(const array& a) {
                    array result;
                    result[0] = vec_abs(a[0]);
                    result[1] = vec_abs(a[1]);
                    return result;
                }

                static bools cmp_eq(const array& a, const array& b) {
                    bools result;
                    result[0] = vec_cmpeq(a[0], b[0]);
                    result[1] = vec_cmpeq(a[1], b[1]);
                    return result;
                }

                static bools cmp_lt(const array& a, const array& b) {
                    bools result;
                    result[0] = vec_cmplt(a[0], b[0]);
                    result[1] = vec_cmplt(a[1], b[1]);
                    return result;
                }

                static bools cmp_gt(const array& a, const array& b) {
                    bools result;
                    result[0] = vec_cmpgt(a[0], b[0]);
                    result[1] = vec_cmpgt(a[1], b[1]);
                    return result;
                }

                static array broadcast(double v) {
                    array result;
                    result[0] = vec_splats(v);
                    result[1] = vec_splats(v);
                    return result;
                }

                static array ifelse(const bools& m,
                                    const array& u,
                                    const array& v) {
                    array result;
                    result[0] = vec_sel(v[0], u[0], m[0]);
                    result[1] = vec_sel(v[1], u[1], m[1]);
                    return result;
                }

                static array floor(const array& a) {
                    array result;
                    result[0] = vec_floor(a[0]);
                    result[1] = vec_floor(a[1]);
                    return result;
                }

                static array round(const array& a) {
                    array result;
                    result[0] = vec_cts(a[0], 0);
                    result[1] = vec_cts(a[1], 0);
                    return result;
                }

                static i64x2x2 roundi(const array& a) {
                    array result;
                    result[0] = vec_cts(a[0], 0);
                    result[1] = vec_cts(a[1], 0);
                    return result;
                }

                // Stolen from avx2
                static array exp(const array& x) {
                    const auto is_large   = cmp_gt(x, broadcast(exp_maxarg));
                    const auto is_small   = cmp_lt(x, broadcast(exp_minarg));
                    const auto is_not_nan = cmp_eq(x, x);

                    // Compute n and g.
                    const auto n = floor(madd(broadcast(ln2inv), x, broadcast(0.5)));
                    auto g = madd(n, broadcast(-ln2C1), x);
                    g = madd(n, broadcast(-ln2C2), g);
                    const auto gg = mul(g, g);

                    // Compute the g*P(g^2) and Q(g^2).
                    const auto odd  = mul(g, horner(gg, P0exp, P1exp, P2exp));
                    const auto even =        horner(gg, Q0exp, Q1exp, Q2exp, Q3exp);

                    // Compute R(g)/R(-g) = 1 + 2*g*P(g^2) / (Q(g^2)-g*P(g^2))
                    const auto expg = madd(broadcast(2.0),
                                           div(odd, sub(even, odd)),
                                           broadcast(1.0));
                    
                    // Finally, compute product with 2^n.
                    // Note: can only achieve full range using the ldexp implementation,
                    // rather than multiplying by 2^n directly.
                    const auto result = ldexp_positive(expg, cfti(n));
                    return ifelse(is_large,
                                  broadcast(HUGE_VAL),
                                  ifelse(is_small,
                                         broadcast(0.0),
                                         ifelse(is_not_nan,
                                                result,
                                                broadcast(NAN))));
                }

                static  array expm1(const array& x) {
                    const auto is_large   = cmp_gt(x, broadcast(exp_maxarg));
                    const auto is_small   = cmp_lt(x, broadcast(expm1_minarg));
                    const auto is_not_nan = cmp_eq(x, x);

                    const auto zero = broadcast(0.0);
                    const auto half = broadcast(0.5);
                    const auto one  = broadcast(1.0);
                    const auto two  = add(one, one);

                    const auto smallx = cmp_leq(abs(x), half);
                    auto n = floor(madd(broadcast(ln2inv), x, half));
                    n = ifelse(smallx, zero, n);

                    auto g = madd(n, broadcast(-ln2C1), x);
                    g =      madd(n, broadcast(-ln2C2), g);

                    const auto gg = mul(g, g);

                    const auto odd  = mul(g, horner(gg, P0exp, P1exp, P2exp));
                    const auto even =        horner(gg, Q0exp, Q1exp, Q2exp, Q3exp);

                    const auto expgm1 = div(mul(broadcast(2.0), odd), sub(even, odd));

                    const auto nm1 = cfti(sub(n, one));
                    const auto scaled = mul(add(sub(exp2int(nm1),
                                                    half),
                                                ldexp_normal(expgm1, nm1)),
                                            two);

                    return ifelse(is_large, broadcast(HUGE_VAL),
                                  ifelse(is_small, broadcast(-1),
                                         ifelse(is_not_nan, ifelse(smallx,
                                                                   expgm1,
                                                                   scaled),
                                                broadcast(NAN))));
                }

                static array log(const array& x) {
                    // Masks for exceptional cases.
                    const auto is_not_large = cmp_lt(x, broadcast(HUGE_VAL));
                    const auto is_small     = cmp_lt(x, broadcast(log_minarg));
                    const auto is_not_nan   = cmp_eq(x, x);
                    // error if x < 0 or n == nan
                    // nb. x == 0 is handled by is_small
                    auto is_domainerr = cmp_lt(x, broadcast(0.0));
                    is_domainerr[0] |= ~is_not_nan[0];
                    is_domainerr[1] |= ~is_not_nan[1];

                    const array one  = broadcast(1.0);
                    const array half = broadcast(0.5);

                    array g = logb_normal(x);
                    array u = fraction_normal(x);
                    const auto gtsqrt2 = cmp_geq(u, broadcast(sqrt2));
                    g = ifelse(gtsqrt2, add(g, one), g);
                    u = ifelse(gtsqrt2, mul(u, half), u);

                    const auto z = sub(u, one);
                    const auto pz = horner(z, P0log, P1log, P2log, P3log, P4log, P5log);
                    const auto qz = horner1(z, Q0log, Q1log, Q2log, Q3log, Q4log);

                    const auto z2 = mul(z, z);
                    const auto z3 = mul(z2, z);

                    auto r = div(mul(z3, pz), qz);
                    r = madd(g,  broadcast(ln2C4), r);
                    r = msub(z2, half, r);
                    r = sub(z, r);
                    r = madd(g,  broadcast(ln2C3), r);

                    // Return NaN if x is NaN or negative, +inf if x is +inf,
                    // or -inf if zero or (positive) denormal.
                    return ifelse(is_domainerr,
                                  broadcast(NAN),
                                  ifelse(is_not_large,
                                         ifelse(is_small,
                                                broadcast(-HUGE_VAL),
                                                r),
                                         broadcast(HUGE_VAL)));
                }

                static array pow(const array& b, const array& e) {
                    // Taken from Agner Fog's Vector Class Library (VCL) version 2.
                    // define constants
                    const double ln2d_hi = 0.693145751953125;           // log(2) in extra precision, high bits
                    const double ln2d_lo = 1.42860682030941723212E-6;   // low bits of log(2)
                    const double log2e   = VM_LOG2E;                    // 1/log(2)

                    // coefficients for Pade polynomials
                    const double P0logl =  2.0039553499201281259648E1;
                    const double P1logl =  5.7112963590585538103336E1;
                    const double P2logl =  6.0949667980987787057556E1;
                    const double P3logl =  2.9911919328553073277375E1;
                    const double P4logl =  6.5787325942061044846969E0;
                    const double P5logl =  4.9854102823193375972212E-1;
                    const double P6logl =  4.5270000862445199635215E-5;
                    const double Q0logl =  6.0118660497603843919306E1;
                    const double Q1logl =  2.1642788614495947685003E2;
                    const double Q2logl =  3.0909872225312059774938E2;
                    const double Q3logl =  2.2176239823732856465394E2;
                    const double Q4logl =  8.3047565967967209469434E1;
                    const double Q5logl =  1.5062909083469192043167E1;

                    // Taylor coefficients for exp function, 1/n!
                    const double p2  = 1./2.;
                    const double p3  = 1./6.;
                    const double p4  = 1./24.;
                    const double p5  = 1./120.;
                    const double p6  = 1./720.;
                    const double p7  = 1./5040.;
                    const double p8  = 1./40320.;
                    const double p9  = 1./362880.;
                    const double p10 = 1./3628800.;
                    const double p11 = 1./39916800.;
                    const double p12 = 1./479001600.;
                    const double p13 = 1./6227020800.;

                    const auto c0  = broadcast(0.0);
                    const auto c1  = broadcast(1.0);
                    const auto ci2 = broadcast(0.5);

                    // remove sign
                    const auto x1 = abs(x0);

                    // Separate mantissa from exponent
                    // This gives the mantissa * 0.5
                    auto x = fraction_normal(x1); // x must be multiplied by 0.5?!

                    // reduce range of x = +/- sqrt(2)/2
                    const auto blend = cmp_gt(x, broadcast(VM_SQRT2*0.5));
                    // conditional add
                    x = add(x, ifelse(blend, x, c0));

                    // Pade approximation
                    // Higher precision than in log function. Still higher precision wanted
                    x = sub(x, one);
                    const auto x2 = mul(x, x);
                    auto px = polynomial6(x, P0logl, P1logl, P2logl, P3logl, P4logl, P5logl, P6logl);
                    px = mul(px, mul(x, x2));
                    const auto qx = polynomial6n(x, Q0logl, Q1logl, Q2logl, Q3logl, Q4logl, Q5logl);
                    const auto lg1 = div(px, qx);

                    // extract exponent
                    auto ef = exponent_f(x1);
                    ef = add(ef, ifelse(blend, c1, c0));                  // conditional add

                    // multiply exponent by y
                    // nearest integer e1 goes into exponent of result, remainder yr is added to log
                    const auto e1 = round(mul(ef, y));
                    // NB. we know fms exists, so we are fine?!
                    const auto yr = msub(ef, y, e1);                   // calculate remainder yr. precision very important here

                    // add initial terms to Pade expansion
                    const auto lg = add(nmadd(ci2, x2, x), lg1);             // lg = (x - 0.5 * x2) + lg1;
                    // calculate rounding errors in lg
                    // rounding error in multiplication 0.5*x*x
                    const auto x2err = mul_sub_x(mul(ci2, x), x, mul(ci2, x2));
                    // rounding error in additions and subtractions
                    const auto lgerr = sub(madd(ci2, x2, sub(lg, x)), lg1);      // lgerr = ((lg - x) + 0.5 * x2) - lg1;

                    // extract something for the exponent
                    const auto e2 = round(mul(mul(lg, y), broadcast(VM_LOG2E)));
                    // subtract this from lg, with extra precision
                    auto v = msub(lg, y, mul(e2, ln2d_hi)); // We use msub here since we know that fused instruction are implemented
                    v = nmadd(e2, ln2d_lo, v);                // v -= e2 * ln2d_lo;
                    // add remainder from ef * y
                    v = madd(yr, broadcast(VM_LN2), v);                  // v += yr * VM_LN2;
                    // correct for previous rounding errors
                    v = nmadd(add(lgerr, x2err), y, v);           // v -= (lgerr + x2err) * y;

                    // exp function
                    // extract something for the exponent if possible
                    x = v;
                    const auto e3 = round(mul(x, broadcast(log2e)));
                    // high precision multiplication not needed here because abs(e3) <= 1
                    x = nmadd(e3, broadcast(VM_LN2), x);                 // x -= e3 * VM_LN2;

                    auto z = horner(x, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13);
                    z = add(z, c1);

                    // contributions to exponent
                    const auto ee = add(e1, add(e2, e3));
                    // 2 x 2 x f64 -> 4 x i32
                    const auto ei = roundi(ee);
                    // biased exponent of result:
                    const auto ej = or(ei, shift_r((reinterpret_cast<>(z)), 52));
                    // check exponent for overflow and underflow
                    const auto overflow  = or(vsx_int2x2::cmp_lt(vsx_int2x2::broadcast(0x07FF), ej),
                                              cmp_gt(ee, broadcast( 3000.0)));
                    const auto underflow = or(vsx_int2x2::cmp_gt(vsx_int2x2::broadcast(0x0000), ej),
                                              cmp_lt(ee, broadcast(-3000.0)));

                    // add exponent by integer addition
                    const auto z = reinterpret_cast<array>(add(reinterpret_cast<i64x2x2>(z), shift_l(ei, 52)));

                    // check for special cases
                    const auto xfinite   = is_finite(x0);
                    const auto yfinite   = is_finite(y);
                    const auto efinite   = is_finite(ee);
                    const auto xzero     = is_zero_or_subnormal(x0);
                    const auto xsign     = sign_bit(x0);  // sign of x0. include -0.

                    // check for overflow and underflow
                    if (any(or(overflow, underflow))) {
                        // handle errors
                        z = ifelse(underflow, c0, z);
                        z = ifelse(overflow, HUGE_VAL, z);
                    }

                    // check for x == 0
                    z = wm_pow_case_x0(xzero, y, z);
                    z = ifelse(xzero,
                               ifelse(cmp_lt(y, c0),
                                      broadcast(NAN),
                                      ifelse(cmp_eq(y, c0),
                                             c1,
                                             c0)),
                               z);

                    // check for sign of x (include -0.). y must be integer
                    array z1;
                    array yodd;
                    if (any(xsign)) {
                        // test if y is an integer
                        auto yinteger = cmp_eq(y, round(y));
                        // test if y is odd: convert to int and shift bit 0 into position of sign bit.
                        // this will be 0 if overflow
                        yodd = reinterpret_cast<array>(shift_l(roundi(y), 63));
                        z1 = ifelse(yinteger,
                                    or(z, yodd),                    // y is integer. get sign if y is odd
                                    select(x0 == 0.,
                                           z,
                                           broadcast(NAN))); // NAN unless x0 == -0.
                        yodd = select(yinteger, yodd, 0.);                 // yodd used below. only if y is integer
                        z = select(xsign, z1, z);
                    }

                    // check for range errors; fast return if no special cases
                    if (all(and(and(xfinite, yfinite), or(efinite, xzero)))) {
                        return z;
                    }

                    // handle special error cases: y infinite
                    z1 = ifelse(or(yfinite, efinite),
                                z,
                                ifelse(cmp_eq(x1, c1),
                                       c1,
                                       ifelse(xor(cmp_gt(x1, c1), sign_bit(y)),
                                              broadcast(HUGE_VAL)
                                              c0)));

                    // handle x infinite
                    z1 = ifelse(xfinite,
                                z1,
                                ifelse(cmp_eq(y, c0),
                                       c1,
                                       ifelse(cmp_lt(y, c0),
                                              and(yodd, z),                  // 0.0 with the sign of z from above
                                              or(abs(x0), and(x0, yodd))))); // get sign of x0 only if y is odd integer

                    // Always propagate nan:
                    // Deliberately differing from the IEEE-754 standard which has pow(0,nan)=1, and pow(1,nan)=1
                    return ifelse(or(is_nan(x0), is_nan(y)),
                                add(x0, y),
                                z1);
                }

                // extract something for the exponent if possible
                x = v;
                e3 = round(x*log2e);
                // high precision multiplication not needed here because abs(e3) <= 1
                x = nmadd(e3, broadcast(VM_LN2), x);                 // x -= e3 * VM_LN2;

                z = horner(x, c0, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13);
                z = add(z, c1);
                
                // contributions to exponent
                ee = add(add(e1, e2), e3);
                ei = roundi(ee);
                // biased exponent of result:
                ej = ei + shift_r(reinterpret_cast<i64x2x2>(z), 52);
                // check exponent for overflow and underflow
                overflow  = BVTYPE(ej >= 0x07FF) | (ee >  3000.);
                underflow = BVTYPE(ej <= 0x0000) | (ee < -3000.);

                // add exponent by integer addition
                z = reinterpret_cast<f64x2x2>(add(reinterpret_cast<i64x2x2>(z), shift_l(ei, 52)));

                // check for special cases
                xfinite   = is_finite(x0);
                yfinite   = is_finite(y);
                efinite   = is_finite(ee);
                xzero     = is_zero_or_subnormal(x0);
                xsign     = sign_bit(x0);  // sign of x0. include -0.

                // check for overflow and underflow
                if (any(or(overflow, underflow))) {
                    // handle errors
                    z = ifelse(underflow, c0, z);
                    z = ifelse(overflow, broadcast(NAN), z);
                }
            }

            protected:

            static inline array polynomial_6(array const x, double c0, double c1, double c2, double c3, double c4, double c5, double c6) {
                // calculates polynomial c6*x^6 + c5*x^5 + c4*x^4 + c3*x^3 + c2*x^2 + c1*x + c0
                array x2 = mul(x, x);
                array x4 = mul(x2, x2);
                //return  (c4+c5*x+c6*x2)*x4 + ((c2+c3*x)*x2 + (c0+c1*x));
                return madd(madd(broadcast(c6),
                                 x2,
                                 madd(broadcast(c5),
                                      x,
                                      broadcast(c4))),
                            x4,
                            madd(madd(broadcast(c3),
                                      x,
                                      broadcast(c2)),
                                 x2,
                                 madd(broadcast(c1),
                                      x,
                                      broadcast(c0))));
            }

            static inline array polynomial_6n(array const x, double c0, double c1, double c2, double c3, double c4, double c5) {
                // calculates polynomial 1*x^6 + c5*x^5 + c4*x^4 + c3*x^3 + c2*x^2 + c1*x + c0
                array x2 = mul(x, x);
                array x4 = mul(x2, x2);
                // (c4+c5*x+x2)*x4 + ((c2+c3*x)*x2 + (c0+c1*x));
                return madd(madd(broadcast(c5),           // (c5*x + x2 + c4)*x4
                                 x,
                                 add(broadcast(c4), x2)),
                            x4,
                            madd(madd(broadcast(c3),
                                      x,
                                      broadcast(c2)),
                                 x2,
                                 madd(broadcast(c1),
                                      x,
                                      broadcast(c0))));
            }

            static inline array polynomial_13(array const x, double c0, double c1, double c2, double c3, double c4, double c5, double c6, double c7, double c8, double c9, double c10, double c11, double c12, double c13) {
                // calculates polynomial c13*x^13 + c12*x^12 + ... + c1*x + c0
                array x2 = x  * x;
                array x4 = x2 * x2;
                array x8 = x4 * x4;
                return madd(
                    madd(
                        madd(broadcast(c13), x, broadcast(c12)), x4,
                        madd(madd(broadcast(c11), x, broadcast(c10)), x2, madd(broadcast(c9), x, broadcast(c8)))), x8,
                    madd(
                        madd(madd(broadcast(c7), x, broadcast(c6)), x2, madd(broadcast(c5), x, broadcast(c4))), x4,
                        madd(madd(broadcast(c3), x, broadcast(c2)), x2, madd(broadcast(c1), x, broadcast(c0)))));
            }

            static inline array horner(const array& x, const double a0) { return broadcast(a0); }

            template <typename... T>
            static array horner(const array& x, const double a0, T... tail) { return madd(x, horner(x, tail...), broadcast(a0)); }

            static inline array horner1(const array& x, const double a0) { return add(x, broadcast(a0)); }

            template <typename... T>
            static array horner1(const array& x, const double a0, T... tail) { return madd(x, horner1(x, tail...), broadcast(a0)); }

            // Compute 2^n·x
            static array ldexp_positive(const array &x, const ints n) {
                const auto nshift_l = vec_vupkhsw(n) << 52; // the unpack *high* unpacks indices 0 and 1. BigEndian, I guess
                const auto nshift_h = vec_vupklsw(n) << 52;
                const auto x_l = reinterpret_cast<vector long long>(x[0]);
                const auto x_h = reinterpret_cast<vector long long>(x[1]);
                const auto s_l = x_l + nshift_l;
                const auto s_h = x_h + nshift_h;
                array res;
                res[0] = reinterpret_cast<vector double>(s_l);
                res[1] = reinterpret_cast<vector double>(s_h);
                return res;
            }

            array exponent_f(const array& a) {
                const vector double pow2_52 {4503599627370496.0, 4503599627370496.0 };   // 2^52
                const vector double bias { 1023.0, 1023.0};
                const auto a_l = reinterpret_cast<vector long long>(a[0]) >> 52;
                const auto a_h = reinterpret_cast<vector long long>(a[1]) >> 52;

                const auto b_l = a[0] | reinterpret_cast<vector long long>(pow2_52);
                const auto b_h = a[1] | reinterpret_cast<vector long long>(pow2_52);

                array result;
                result[0] = reinterpret_cast<vector double>(b_l) - (pow2_52 + bias);
                result[1] = reinterpret_cast<vector double>(b_h) - (pow2_52 + bias);
                return result;
            }

            // Compute n and f such that x = 2^n·f, with |f| ∈ [1,2), given x is finite
            static array logb_normal(const array& x) {
                auto xw = vec_perm(reinterpret_cast<vector unsigned>(x[0]),
                                   reinterpret_cast<vector unsigned>(x[1]),
                                   (vector unsigned char) {4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31});
                auto mask = (vector unsigned) { 0x7ff00000,
                                                0x7ff00000,
                                                0x7ff00000,
                                                0x7ff00000 };
                auto c1023 = (vector int) { 1023,
                                            1023,
                                            1023,
                                            1023 };
                auto res = reinterpret_cast<vector int>((xw & mask) >> 20) - c1023;
                array r;
                r[0][0] = res[0];
                r[0][1] = res[1];
                r[1][0] = res[2];
                r[1][1] = res[3];
                return r;
            }

            static array fraction_normal(const array& x) {
                const auto mask = (vector long long) { -0x7ff0000000000001ll, -0x7ff0000000000001ll };
                const auto bias = (vector long long) {  0x3ff0000000000000ll,  0x3ff0000000000000ll };
                const auto x_l = reinterpret_cast<vector long long>(x[0]);
                const auto x_h = reinterpret_cast<vector long long>(x[1]);
                array result;
                result[0] = reinterpret_cast<vector double>(bias | (mask & x_l));
                result[1] = reinterpret_cast<vector double>(bias | (mask & x_h));
                return result;
            }

            static inline array fraction_2(const array& a) {
                const auto mask = (vector long long) { 0x000FFFFFFFFFFFFFll, 0x000FFFFFFFFFFFFFll };
                const auto bias = (vector long long) { 0x3FE0000000000000ll, 0x3FE0000000000000ll };
                const auto x_l = reinterpret_cast<vector long long>(a[0]);
                const auto x_h = reinterpret_cast<vector long long>(a[1]);
                array result;
                result[0] = reinterpret_cast<vector double>(bias | (mask & x_l));
                result[1] = reinterpret_cast<vector double>(bias | (mask & x_h));
                return result;
            }

            // Compute 2^n*x when both x and 2^n*x are normal and finite.
            static array ldexp_normal(const array& x, const ints n) {
                const auto x_l = reinterpret_cast<vector long long>(x[0]);
                const auto x_h = reinterpret_cast<vector long long>(x[1]);
                const auto smask = (vector long long) { 0x7fffffffffffffffll,
                                                        0x7fffffffffffffffll };
                const auto sbits_l = (~smask) & x_l;
                const auto sbits_h = (~smask) & x_h;
                const auto nshift_l = vec_vupkhsw(n) << 52; // the unpack *high* unpacks indices 0 and 1. BigEndian, I guess
                const auto nshift_h = vec_vupklsw(n) << 52;
                const auto s_l = x_l + nshift_l;
                const auto s_h = x_h + nshift_h;
                const auto nzans_l = (s_l & smask) | sbits_l;
                const auto nzans_h = (s_h & smask) | sbits_h;
                array result;
                result[0] = reinterpret_cast<vector double>(nzans_l);
                result[1] = reinterpret_cast<vector double>(nzans_h);
                return ifelse(cmp_eq(x,
                                     broadcast(0.0)),
                              broadcast(0.0),
                              result);
            }

            static array exp2int(const ints n) { return ldexp_positive(broadcast(1.0), n); }

            static ints cfti(const array& v) {
                return (ints) { static_cast<int>(v[0][0]), static_cast<int>(v[0][1]), static_cast<int>(v[1][0]), static_cast<int>(v[1][1]) };
            }
        };
    }  // namespace detail

    namespace simd_abi {
        template <typename T, unsigned N>
        struct vsx;

        template <>
        struct vsx<double, 4> {
            using type = detail::vsx_double4;
        };

        template <>
        struct vsx<int, 4> {
            using type = detail::vsx_int4;
        };
    }  // namespace simd_abi

}  // namespace simd
}  // namespace arb

#endif  // def __VSX
