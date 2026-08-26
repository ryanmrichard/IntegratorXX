#pragma once

#include <cmath>
#include <type_traits>

namespace IntegratorXX {

/**
 *  @brief Customization point for floating-point operations used by the
 *         quadrature generators.
 *
 *  IntegratorXX is templated on the type used to represent quadrature points
 *  and weights. For the built-in floating point types the required math
 *  functions live in `std`, but user-supplied types (interval arithmetic,
 *  uncertainty-propagating scalars, arbitrary precision reals, ...) provide
 *  their own. This trait is the single seam through which every such operation
 *  is routed, so that supporting a new type is a matter of specializing one
 *  class rather than auditing every quadrature.
 *
 *  The primary template performs a two-step (`using std::fn; fn(x);`) call, so
 *  any type whose math functions are reachable by argument-dependent lookup
 *  works with **no specialization at all**. Specialize only when ADL is not
 *  sufficient, or when the default behaviour is wrong for the type -- see
 *  `from_inexact` below for the motivating case.
 *
 *  @tparam T      The floating-point-like type.
 *  @tparam Enable Hook for SFINAE-constrained partial specializations.
 */
template <typename T, typename Enable = void>
struct fp_traits {

  /// Natural logarithm of @p x.
  static T log(const T& x) { using std::log; return log(x); }

  /// Base-e exponential of @p x.
  static T exp(const T& x) { using std::exp; return exp(x); }

  /// Square root of @p x.
  static T sqrt(const T& x) { using std::sqrt; return sqrt(x); }

  /// Cosine of @p x.
  static T cos(const T& x) { using std::cos; return cos(x); }

  /// Sine of @p x.
  static T sin(const T& x) { using std::sin; return sin(x); }

  /// Absolute value of @p x.
  static T abs(const T& x) { using std::abs; return abs(x); }

  /// @p x raised to the power @p p.
  template <typename U>
  static T pow(const T& x, const U& p) { using std::pow; return pow(x, p); }

  /**
   *  @brief Convert a `double` that is *exactly* representable in binary
   *         floating point into @p T.
   *
   *  For literals such as `1.0`, `0.5`, `2.0`, `0.25` and `-2.0`, whose decimal
   *  spelling has an exact binary representation. Types that model a set of
   *  values (e.g. intervals) must map these to a *degenerate* value: the
   *  conversion introduces no error, so none should be manufactured.
   *
   *  Using this where `from_inexact` is required understates the uncertainty of
   *  a tabulated constant. Using `from_inexact` here is worse: it injects
   *  spurious width into expressions such as `1.0 - x`, which sit on the
   *  cancellation-sensitive path of several radial transformations, and so
   *  inflates the reported error precisely where a tight bound matters most.
   *
   *  @param[in] v An exactly-representable value.
   *  @return    @p v as a @p T.
   */
  static constexpr T from_exact(double v) { return T(v); }

  /**
   *  @brief Convert a `double` that is *not* exactly representable into @p T.
   *
   *  For tabulated data (the solid-angle quadrature abscissae and weights) and
   *  for irrational constants such as `ln(2)`. Such values carry at minimum the
   *  representation error of the decimal literal, and in the tabulated case the
   *  residual error of whatever nonlinear solve produced them.
   *
   *  The default is a plain conversion, which is correct for types that do not
   *  claim to bound their own error. Types that *do* make such a claim should
   *  specialize this to widen the result outward; see the documentation for the
   *  chosen widening policy.
   *
   *  @param[in] v A value whose binary representation is inexact.
   *  @return    @p v as a @p T.
   */
  static T from_inexact(double v) { return T(v); }
};

} // namespace IntegratorXX
