#pragma once

#include <cmath>
#include <integratorxx/config.hpp>
#include <type_traits>

#ifdef ENABLE_STRING_REALS
#include <charconv>
#endif

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
 *  `from_real` below for the motivating case.
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

  /** @brief Convert an integral value to @p T.
   *
   *  Integral values are exactly representable in every supported arithmetic
   *  type, so this conversion introduces no error and types that model a set of
   *  values must map it to a degenerate one.
   *
   *  @param[in] v An integral value, normally spelled `IXX_INT(...)`.
   *  @return    @p v as a @p T.
   */
  static T from_integer(ixx_int v) { return T(v); }

  /** @brief Convert a non-integral literal to @p T.
   *
   *  The argument is normally spelled `IXX_REAL(...)`. Its type depends on the
   *  build: a `double` by default, or the literal's decimal source text when
   *  `ENABLE_STRING_REALS` is defined.
   *
   *  @warning In string mode this default parses via `double`, which discards
   *  exactly the precision the string form exists to preserve. That is the best
   *  a generic implementation can do, and it keeps `float`/`double` correct, but
   *  it means **string mode only pays off for types that specialize this
   *  function**. A type that bounds its own error should parse the text
   *  directly -- for an interval type, twice under directed rounding -- to
   *  obtain a tight enclosure of the decimal rather than of an already-rounded
   *  `double`.
   *
   *  @param[in] v The literal, as `ixx_real`.
   *  @return    @p v as a @p T.
   */
  static T from_real(ixx_real v) {
#ifdef ENABLE_STRING_REALS
    double d{};
    std::from_chars(v.data(), v.data() + v.size(), d);
    return T(d);
#else
    return T(v);
#endif
  }

  /** @brief Convert the exact rational @p num / @p den to @p T.
   *
   *  Prefer this over a pre-divided literal wherever a constant is the ratio of
   *  two integers. Both operands are exact, so the result of the division is
   *  the only rounding: for `double` that is the correctly-rounded quotient,
   *  and for an interval type it is a true enclosure of the rational, neither
   *  of which is guaranteed by converting a constant that was already rounded
   *  to `double` by the compiler.
   *
   *  @param[in] num The numerator.
   *  @param[in] den The denominator.
   *  @return    @p num / @p den as a @p T.
   */
  static T divide_integer(ixx_int num, ixx_int den) {
    return from_integer(num) / from_integer(den);
  }
};

} // namespace IntegratorXX
