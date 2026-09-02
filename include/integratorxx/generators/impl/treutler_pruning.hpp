#pragma once
#include <integratorxx/generators/spherical_factory.hpp>

#include <integratorxx/generators/impl/s2_types.hpp>

namespace IntegratorXX {

namespace detail {

template <typename AngularQuad>
auto get_treutler_low_med_sizes() {
  using traits = quadrature_traits<AngularQuad>;
  AngularSize med_sz(traits::npts_by_algebraic_order(traits::next_algebraic_order(11)));
  AngularSize low_sz(traits::npts_by_algebraic_order(traits::next_algebraic_order(7 )));

  return std::make_pair(low_sz, med_sz);
}

template <typename T>
PrunedSphericalGridSpecification<T> treutler_pruning_scheme_impl(
  size_t low_sz, size_t med_sz, AngularQuad angular_quad,
  UnprunedSphericalGridSpecification<T> unp ) {

  const size_t rsz = unp.radial_traits->npts();
  const size_t r_div_3 = rsz / 3ul + 1ul;
  const size_t r_div_2 = rsz / 2ul + 1ul;
  std::vector<PruningRegion> pruning_regions = {
    {0ul,     r_div_3, angular_quad, low_sz},
    {r_div_3, r_div_2, angular_quad, med_sz},
    {r_div_2, rsz,     angular_quad, unp.angular_size}
  };

  return PrunedSphericalGridSpecification<T>(
    unp.radial_quad, unp.radial_traits->clone(), pruning_regions
  );

}

} // Implementation details



template <typename T>
PrunedSphericalGridSpecification<T> treutler_pruning_scheme(
  UnprunedSphericalGridSpecification<T> unp ) {

  size_t low_sz, med_sz;
  const auto angular_quad = unp.angular_quad;
  switch(angular_quad) {
    case AngularQuad::AhrensBeylkin:
      std::tie(low_sz, med_sz) = detail::get_treutler_low_med_sizes<detail::ah_type<T>>();
      break;
    case AngularQuad::Delley:
      std::tie(low_sz, med_sz) = detail::get_treutler_low_med_sizes<detail::de_type<T>>();
      break;
    case AngularQuad::LebedevLaikov:
      std::tie(low_sz, med_sz) = detail::get_treutler_low_med_sizes<detail::ll_type<T>>();
      break;
    case AngularQuad::Womersley:
      std::tie(low_sz, med_sz) = detail::get_treutler_low_med_sizes<detail::wo_type<T>>();
      break;
    default:
      throw std::runtime_error("Unsupported Angular Quadrature");
      abort();
  }

  return detail::treutler_pruning_scheme_impl(low_sz, med_sz, angular_quad, unp);
}


}
