#pragma once

#include <cstddef>
#include <integratorxx/util/fp_traits.hpp>
#include <type_traits>

namespace IntegratorXX {

namespace detail {

/// Detects an equal-weight tabulated grid, which records its point count in
/// place of a weight table. See the Womersley grids.
template <typename StaticGrid, typename = void>
struct has_uniform_weight : std::false_type {};

template <typename StaticGrid>
struct has_uniform_weight<
  StaticGrid, std::void_t<decltype(StaticGrid::uniform_weight_npts)>>
  : std::true_type {};

template <typename StaticGrid>
inline constexpr bool has_uniform_weight_v = has_uniform_weight<StaticGrid>::value;

/**
 *  @brief Copy a tabulated solid-angle grid into caller-supplied containers,
 *         converting to the containers' value types.
 *
 *  The tables are stored as `double` regardless of the type the quadrature is
 *  instantiated over. They must be: the tables are `constexpr`, which requires
 *  a literal type, and the types this library is expected to support (interval
 *  arithmetic, uncertainty-propagating scalars) are not literal. Storing them
 *  once as `double` and converting on read keeps the tables usable from
 *  constant expressions while leaving the quadrature type-generic.
 *
 *  Conversion goes through `fp_traits::from_real`. Every non-integral entry in
 *  these tables is inexact: measured across all 231 tables, 1,913,026 of the
 *  1,919,804 entries are non-integral and *none* of those is representable as a
 *  `double`. The only exactly-representable values are the 6,778 occurrences of
 *  0 and +/-1 at the axis points, and those cannot be spelled `IXX_INT` because
 *  a `std::array` is homogeneous.
 *
 *  @tparam StaticGrid       The tabulated grid; supplies `points`/`weights`.
 *  @tparam PointContainer   Destination for the abscissae.
 *  @tparam WeightsContainer Destination for the weights.
 */
template <class StaticGrid, typename PointContainer, typename WeightsContainer>
void copy_grid(PointContainer &points, WeightsContainer &weights) {

  using point_type  = typename PointContainer::value_type;
  using coord_type  = typename point_type::value_type;
  using weight_type = typename WeightsContainer::value_type;

  using coord_traits  = fp_traits<coord_type>;
  using weight_traits = fp_traits<weight_type>;

  const auto &static_points = StaticGrid::points;

  for(size_t i = 0; i < static_points.size(); ++i)
    for(size_t j = 0; j < static_points[i].size(); ++j)
      points[i][j] = coord_traits::from_real(static_points[i][j]);

  if constexpr(has_uniform_weight_v<StaticGrid>) {
    // Every weight is the sphere's area divided by the point count. Forming it
    // as pi times the exact rational 4/npts is one rounding rather than two,
    // and unlike a pre-divided `double` constant it is correct in string mode
    // and a true enclosure for types that bound their own error.
    const auto w = weight_traits::from_real(ixx_pi)
                 * weight_traits::divide_integer(4, StaticGrid::uniform_weight_npts);
    for(size_t i = 0; i < weights.size(); ++i) weights[i] = w;
  } else {
    const auto &static_weights = StaticGrid::weights;
    for(size_t i = 0; i < static_weights.size(); ++i)
      weights[i] = weight_traits::from_real(static_weights[i]);
  }
}

} // namespace detail
} // namespace IntegratorXX
