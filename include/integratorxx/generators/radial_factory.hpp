#pragma once
#include <integratorxx/quadratures/radial.hpp>
#include <integratorxx/generators/impl/radial_types.hpp>

namespace IntegratorXX {

/// High-level specification of radial quadratures
enum class RadialQuad : uint32_t {
  Becke             = 0x0010,
  MurrayHandyLaming = 0x0020,
  MuraKnowles       = 0x0030,
  TreutlerAhlrichs  = 0x0040
};

template <typename RadQuadType>
RadialQuad radial_from_type() {
  if constexpr (detail::is_becke_v<RadQuadType>) return RadialQuad::Becke;
  if constexpr (detail::is_mk_v<RadQuadType>   ) return RadialQuad::MuraKnowles;
  if constexpr (detail::is_mhl_v<RadQuadType>)   return RadialQuad::MurrayHandyLaming;
  if constexpr (detail::is_ta_v<RadQuadType>)    return RadialQuad::TreutlerAhlrichs;

  throw std::runtime_error("Unrecognized Radial Quadrature");
};

RadialQuad radial_from_string(std::string name);

namespace detail {

template <typename T, typename RadialTraitsType, typename... Args>
std::unique_ptr<RadialTraits<T>> make_radial_traits(Args&&... args) {
  using traits_type = RadialTraitsType;
  if constexpr (std::is_constructible_v<traits_type,Args...>)
    return std::make_unique<traits_type>(std::forward<Args>(args)...);
  else return nullptr;
}

}

template <typename T, typename... Args>
std::unique_ptr<RadialTraits<T>> make_radial_traits(RadialQuad rq, Args&&... args) {
  std::unique_ptr<RadialTraits<T>> ptr;
  switch(rq) {
    case RadialQuad::Becke:
      ptr =
        detail::make_radial_traits<T, BeckeRadialTraits<T>>(std::forward<Args>(args)...);
      break;
    case RadialQuad::MurrayHandyLaming:
      ptr =
        detail::make_radial_traits<T, MurrayHandyLamingRadialTraits<T,2>>(std::forward<Args>(args)...);
      break;
    case RadialQuad::MuraKnowles:
      ptr =
        detail::make_radial_traits<T, MuraKnowlesRadialTraits<T>>(std::forward<Args>(args)...);
      break;
    case RadialQuad::TreutlerAhlrichs:
      ptr =
        detail::make_radial_traits<T, TreutlerAhlrichsRadialTraits<T>>(std::forward<Args>(args)...);
      break;
  }

  if(!ptr) throw std::runtime_error("RadialTraits Construction Failed");
  return ptr;
}


template <typename T>
struct RadialFactory {

  using radial_grid_ptr = std::shared_ptr<
    QuadratureBase<
      std::vector<T>,
      std::vector<T>
    >
  >;

  static radial_grid_ptr generate(RadialQuad rq, const RadialTraits<T>& traits) {
    switch(rq) {
      case RadialQuad::Becke:
        return std::make_unique<detail::bk_type<T>>(traits);
      case RadialQuad::MuraKnowles:
        return std::make_unique<detail::mk_type<T>>(traits);
      case RadialQuad::MurrayHandyLaming:
        return std::make_unique<detail::mhl_type<T>>(traits);
      case RadialQuad::TreutlerAhlrichs:
        return std::make_unique<detail::ta_type<T>>(traits);
      default:
        throw std::runtime_error("Unsupported Radial Quadrature");
    }
  }

};

}
