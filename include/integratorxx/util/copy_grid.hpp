#pragma once

#include <cstddef>
#include <integratorxx/util/fp_traits.hpp>

namespace IntegratorXX {

namespace detail {

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
 *  Conversion goes through `fp_traits::from_inexact` because the tabulated
 *  values are *not* exact. Each carries at least the representation error of
 *  its decimal literal, and the abscissae are additionally the output of a
 *  nonlinear solve of unpublished per-point accuracy. A type that bounds its
 *  own error should widen accordingly rather than claim the tabulated value
 *  is exact.
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

  const auto &static_points  = StaticGrid::points;
  const auto &static_weights = StaticGrid::weights;

  for(size_t i = 0; i < static_points.size(); ++i)
    for(size_t j = 0; j < static_points[i].size(); ++j)
      points[i][j] = coord_traits::from_inexact(static_points[i][j]);

  for(size_t i = 0; i < static_weights.size(); ++i)
    weights[i] = weight_traits::from_inexact(static_weights[i]);
}

} // namespace detail
} // namespace IntegratorXX
