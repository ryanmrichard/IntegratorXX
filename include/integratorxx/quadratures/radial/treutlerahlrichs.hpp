#pragma once

#include <integratorxx/quadratures/primitive/gausschebyshev2.hpp>
#include <integratorxx/quadratures/radial/radial_transform.hpp>
#include <integratorxx/util/fp_traits.hpp>

namespace IntegratorXX {

/**
 *  @brief Implementation of the Treutler-Ahlrichs M3+M4 radial 
 *  quadrature transformation rules.
 *
 *  Reference:
 *  J. Chem. Phys. 102, 346 (1995)
 *  DOI: https://doi.org/10.1063/1.469408
 */
template <typename T>
class TreutlerAhlrichsRadialTraits : public RadialTraits<T> {

  size_t npts_; ///< Number of grid points
  T R_; ///< Radial scaling factor
  T alpha_;

public:

  inline static constexpr ixx_int a    = IXX_INT(1);
  inline static constexpr ixx_real ln_2 = IXX_REAL(0.693147180559945309417232);

  /**
   *  Specify Treutler-Ahlrichs quadrature parameters
   *
   *  M3: Equation (18) of J. Chem. Phys. 102, 346 (1995)
   *  M4: Equation (19) of J. Chem. Phys. 102, 346 (1995)
   *
   *  Default to M4 (alpha = 0.6). M3 resolved with alpha = 0.0
   *
   *  @param[in] R     Radial scaling factor
   *  @param[in] alpha TA exponential factor
   */
  TreutlerAhlrichsRadialTraits(size_t npts,
    T R     = fp_traits<T>::from_real(IXX_REAL(1.0)),
    T alpha = fp_traits<T>::from_real(IXX_REAL(0.6))) :
    npts_(npts), R_(R), alpha_(alpha) { }

  size_t npts() const noexcept { return npts_; }

  std::unique_ptr<RadialTraits<T>> clone() const {
    return std::make_unique<TreutlerAhlrichsRadialTraits>(*this);
  }

  bool compare(const RadialTraits<T>& other) const noexcept {
    auto ptr = dynamic_cast<const TreutlerAhlrichsRadialTraits*>(&other);
    return ptr ? *this == *ptr : false;
  }

  bool operator==(const TreutlerAhlrichsRadialTraits& other) const noexcept {
    return npts_ == other.npts_ && R_ == other.R_ && alpha_ == other.alpha_;
  }

  /**
   *  @brief Transformation rule for the TA M3+M4 radial quadratures
   *  
   *  @param[in] x Point in (-1,1)
   *  @return    r = (a+x)^alpha * log((a+1)/(1-x)) / ln(2) 
   */
  template <typename PointType>
  inline auto radial_transform(PointType x) const noexcept {
    using traits = fp_traits<PointType>;
    const auto a_    = traits::from_integer(a);
    const auto one   = traits::from_integer(1);
    const auto pow_term = traits::pow(a_ + x, alpha_);
    const auto log_term = traits::log((a_ + one) / (one - x));
    return R_ * pow_term * log_term / traits::from_real(ln_2);
  };


  /**
   *  @brief Jacobian of the TA M3+M4 radial transformations
   *
   *  @param[in] x Point in (-1,1)
   *  @returns   dr/dx (see `radial_transform`)
   */
  template <typename PointType>
  inline auto radial_jacobian(PointType x) const noexcept {
    using traits = fp_traits<PointType>;
    const auto a_    = traits::from_integer(a);
    const auto one   = traits::from_integer(1);
    const auto pow_term = traits::pow(a_ + x, alpha_);
    const auto log_term = traits::log((a_ + one) / (one - x));
    return R_ * pow_term / traits::from_real(ln_2)
         * ( alpha_ * log_term / (a_ + x) + (one / (one - x)) );
  }

};


/**
 *  @brief Implementation of the Treutler-Ahlrichs M4 radial quadrature.
 *
 *  Taken as the convolution of the Gauss-Chebyshev (second kind) quadrature 
 *  with the TA M4 radial transformation. See TreutlerAhlrichsRadialTraits for
 *  details.
 *
 *  Suitable for integrands which tend to zero as their argument tends 
 f(r), with 
 *  lim_{r->inf} f(r) = 0.
 *
 *  Reference:
 *  J. Chem. Phys. 102, 346 (1995)
 *  DOI: https://doi.org/10.1063/1.469408
 *
 *  @tparam PointType  Type describing the quadrature points  
 *  @tparam WeightType Type describing the quadrature weights 
 */
template <typename PointType, typename WeightType>
using TreutlerAhlrichs = RadialTransformQuadrature<
  GaussChebyshev2<PointType, WeightType>,
  TreutlerAhlrichsRadialTraits<PointType>
>;

namespace detail {

template <typename QuadType>
static constexpr bool is_ta_v = std::is_same_v<
  QuadType, 
  TreutlerAhlrichs<typename QuadType::point_type, typename QuadType::weight_type>
>;

}

}
