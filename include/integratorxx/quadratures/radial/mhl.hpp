#pragma once

#include <integratorxx/quadratures/primitive/uniform.hpp>
#include <integratorxx/quadratures/radial/radial_transform.hpp>
#include <integratorxx/util/fp_traits.hpp>

namespace IntegratorXX {

/**
 *  @brief Implementation of the Murray-Handy-Laming radial quadrature
 *  transformation rules.
 *
 *  Reference:
 *  Molecular Physics, 78:4, 997-1014,
 *  DOI: https://doi.org/10.1080/00268979300100651
 *
 *  @tparam M Integer to modulate the MHL transformation. 
 *            Typically taken to be 2.
 */
template <typename T, size_t M>
class MurrayHandyLamingRadialTraits : public RadialTraits<T> {

  size_t npts_; ///< Number of grid points
  T R_; ///< Radial scaling factor


public:

  MurrayHandyLamingRadialTraits(size_t npts, T R = fp_traits<T>::from_real(IXX_REAL(1.0))) :
    npts_(npts), R_(R) {}

  size_t npts() const noexcept { return npts_; }

  std::unique_ptr<RadialTraits<T>> clone() const {
    return std::make_unique<MurrayHandyLamingRadialTraits>(*this);
  }

  bool compare(const RadialTraits<T>& other) const noexcept {
    auto ptr = dynamic_cast<const MurrayHandyLamingRadialTraits*>(&other);
    return ptr ? *this == *ptr : false;
  }

  bool operator==(const MurrayHandyLamingRadialTraits& other) const noexcept {
    return npts_ == other.npts_ && R_ == other.R_;
  }

  /**
   *  @brief Transformation rule for the MHL radial quadrature
   *  
   *  @param[in] x Point in (0,1)
   *  @return    r = R * (x / (1-x))^M
   */
  template <typename PointType>
  inline auto radial_transform(PointType x) const noexcept {
    using traits = fp_traits<PointType>;
    return R_ * traits::pow( x / (traits::from_integer(1) - x), M );
  }

  /**
   *  @brief Jacobian of the MHL radial transformation
   *
   *  @param[in] x Point in (0,1)
   *  @returns   dr/dx (see `radial_transform`)
   */
  template <typename PointType>
  inline auto radial_jacobian(PointType x) const noexcept {
    using traits = fp_traits<PointType>;
    // M is a raw size_t (a non-type template parameter), not a PointType --
    // routed through fp_traits<PointType>::from_integer for the same reason
    // spherical_micro_batcher.hpp's loop counters are: `PointType * size_t`
    // requires an exact-type overload of operator*, which template argument
    // deduction won't find via an implicit conversion for a non-builtin
    // PointType like sigma::Interval.
    return R_ * traits::from_integer(static_cast<ixx_int>(M)) * traits::pow(x, M-1)
         / traits::pow(traits::from_integer(1) - x, M+1);
  }

};


/**
 *  @brief Implementation of the Murray-Handy-Laming radial quadrature.
 *
 *  Taken as the convolution of the Uniform (Trapezoid) quadrature with
 *  the MHL radial transformation. See MurrayHandyLamingRadialTraits for
 *  details.
 *
 *  Suitable for integrands which tend to zero as their argument tends 
 *  to 0 and inf. Tailored for radial integrands, i.e. r^2 * f(r), with 
 *  lim_{r->inf} f(r) = 0.
 *
 *  Reference:
 *  Molecular Physics, 78:4, 997-1014,
 *  DOI: https://doi.org/10.1080/00268979300100651
 *
 *  @tparam PointType  Type describing the quadrature points  
 *  @tparam WeightType Type describing the quadrature weights 
 */
template <typename PointType, typename WeightType>
using MurrayHandyLaming = RadialTransformQuadrature<
  UniformTrapezoid<PointType,WeightType>,
  MurrayHandyLamingRadialTraits<PointType,2>
>;

namespace detail {

template <typename QuadType>
static constexpr bool is_mhl_v = std::is_same_v<
  QuadType, 
  MurrayHandyLaming<typename QuadType::point_type, typename QuadType::weight_type>
>;

}
}

