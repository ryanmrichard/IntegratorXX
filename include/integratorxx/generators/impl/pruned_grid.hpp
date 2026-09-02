#pragma once

#include <integratorxx/generators/spherical_factory.hpp>

#include <integratorxx/generators/impl/radial_types.hpp>
#include <integratorxx/generators/impl/s2_types.hpp>

#include <integratorxx/generators/impl/robust_pruning.hpp>
#include <integratorxx/generators/impl/treutler_pruning.hpp>

namespace IntegratorXX {

namespace detail {

template <template<typename> class AngularQuadType, typename RadialQuadType>
auto make_pruned_grid_impl(const RadialQuadType& rq,
  const std::vector<PruningRegion>& pruning_regions) {

  using T = typename RadialQuadType::point_type;
  RadialGridPartition<AngularQuadType<T>> rgp;
  for( auto& region : pruning_regions ) {
    rgp.add_quad( rq, region.idx_st, AngularQuadType<T>(region.angular_size) );
  }
  rgp.finalize(rq);

  return SphericalGridFactory<T>::generate_pruned_grid(rq, std::move(rgp));

}

template <typename RadialQuadType>
auto make_pruned_grid(const RadialQuadType& rq,
  const std::vector<PruningRegion>& pruning_regions) {

  if(pruning_regions.size() == 0)
    throw std::runtime_error("No Pruning Regions");

  auto angular_quad = pruning_regions[0].angular_quad;
  for(auto r : pruning_regions) {
    if(r.angular_quad != angular_quad)
      throw std::runtime_error("Mixed Angular Pruning Not Supported");
  }

  switch(angular_quad) {
    case AngularQuad::AhrensBeylkin:
      return make_pruned_grid_impl<detail::ah_type>(rq, pruning_regions);
    case AngularQuad::Delley:
      return make_pruned_grid_impl<detail::de_type>(rq, pruning_regions);
    case AngularQuad::LebedevLaikov:
      return make_pruned_grid_impl<detail::ll_type>(rq, pruning_regions);
    case AngularQuad::Womersley:
      return make_pruned_grid_impl<detail::wo_type>(rq, pruning_regions);
    default:
      throw std::runtime_error("Unsupported Angular Quadrature");
      abort();
  }


}

} // Implementation Details

template <typename T>
typename SphericalGridFactory<T>::spherical_grid_ptr
  SphericalGridFactory<T>::generate_pruned_grid( RadialQuad rq,
  const RadialTraits<T>& traits,
  const std::vector<PruningRegion>& pruning_regions) {

  switch( rq ) {

    case RadialQuad::Becke:
      return detail::make_pruned_grid( detail::bk_type<T>(traits), pruning_regions );
    case RadialQuad::MuraKnowles:
      return detail::make_pruned_grid( detail::mk_type<T>(traits), pruning_regions );
    case RadialQuad::MurrayHandyLaming:
      return detail::make_pruned_grid( detail::mhl_type<T>(traits), pruning_regions );
    case RadialQuad::TreutlerAhlrichs:
      return detail::make_pruned_grid( detail::ta_type<T>(traits), pruning_regions );

    default:
      throw std::runtime_error("Unsupported Radial Quadrature");
      abort();

  }

}


template <typename T>
PrunedSphericalGridSpecification<T> create_pruned_spec(
  PruningScheme scheme, UnprunedSphericalGridSpecification<T> unp
) {

  if(!unp.radial_traits) throw std::runtime_error("RadialTraits Not Set");
  switch(scheme) {
    case PruningScheme::Robust:
      return robust_psi4_pruning_scheme(unp);
    case PruningScheme::Treutler:
      return treutler_pruning_scheme(unp);

    // Default to Unpruned Grid
    case PruningScheme::Unpruned:
    default:
      std::vector<PruningRegion> pruning_regions = {
        {0ul, unp.radial_traits->npts(), unp.angular_quad, unp.angular_size}
      };
      return PrunedSphericalGridSpecification<T>(
        unp.radial_quad, unp.radial_traits->clone(), pruning_regions
      );
  }

}

}
