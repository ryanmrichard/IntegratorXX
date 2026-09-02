#pragma once
#include <integratorxx/generators/radial_factory.hpp>

#include <algorithm>

namespace IntegratorXX {

RadialQuad radial_from_string(std::string name) {
  std::transform(name.begin(), name.end(), name.begin(),
    [](unsigned char c){ return static_cast<char>(std::toupper(c)); });
  if(name == "BECKE")             return RadialQuad::Becke;
  if(name == "MURAKNOWLES")       return RadialQuad::MuraKnowles;
  if(name == "MK")                return RadialQuad::MuraKnowles;
  if(name == "MURRAYHANDYLAMING") return RadialQuad::MurrayHandyLaming;
  if(name == "MHL")               return RadialQuad::MurrayHandyLaming;
  if(name == "TREUTLERAHLRICHS")  return RadialQuad::TreutlerAhlrichs;
  if(name == "TA")                return RadialQuad::TreutlerAhlrichs;

  throw std::runtime_error("Unrecognized Radial Quadrature");
}

}
