#pragma once

#include <integratorxx/generators/spherical_factory.hpp>

#include <integratorxx/generators/impl/radial_types.hpp>
#include <integratorxx/generators/impl/s2_types.hpp>

namespace IntegratorXX {

namespace detail {

template <typename T, typename AngularQuadType>
auto generate_unpruned_grid_impl(RadialQuad rq, const RadialTraits<T>& traits,
  AngularQuadType&& ang_quad) {

  switch( rq ) {

    case RadialQuad::Becke:
      return SphericalGridFactory<T>::generate_unpruned_grid( detail::bk_type<T>(traits), std::forward<AngularQuadType>(ang_quad) );

    case RadialQuad::MuraKnowles:
      return SphericalGridFactory<T>::generate_unpruned_grid( detail::mk_type<T>(traits), std::forward<AngularQuadType>(ang_quad) );

    case RadialQuad::MurrayHandyLaming:
      return SphericalGridFactory<T>::generate_unpruned_grid( detail::mhl_type<T>(traits), std::forward<AngularQuadType>(ang_quad) );

    case RadialQuad::TreutlerAhlrichs:
      return SphericalGridFactory<T>::generate_unpruned_grid( detail::ta_type<T>(traits), std::forward<AngularQuadType>(ang_quad) );

    default:
      throw std::runtime_error("Unsupported Radial Quadrature");
      abort();

  }

}

} // Implementation details

template <typename T>
typename SphericalGridFactory<T>::spherical_grid_ptr
  SphericalGridFactory<T>::generate_unpruned_grid( RadialQuad rq,
  const RadialTraits<T>& traits, AngularQuad aq, AngularSize nang) {

  switch(aq) {
    case AngularQuad::AhrensBeylkin:
      return detail::generate_unpruned_grid_impl(rq, traits, detail::ah_type<T>(nang));
    case AngularQuad::Delley:
      return detail::generate_unpruned_grid_impl(rq, traits, detail::de_type<T>(nang));
    case AngularQuad::LebedevLaikov:
      return detail::generate_unpruned_grid_impl(rq, traits, detail::ll_type<T>(nang));
    case AngularQuad::Womersley:
      return detail::generate_unpruned_grid_impl(rq, traits, detail::wo_type<T>(nang));
    default:
      throw std::runtime_error("Unsupported Angular Quadrature");
      abort();
  }
}

}
