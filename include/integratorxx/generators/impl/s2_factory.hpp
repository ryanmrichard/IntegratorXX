#pragma once
#include <integratorxx/generators/s2_factory.hpp>

#include <integratorxx/generators/impl/s2_types.hpp>

#include <algorithm>

namespace IntegratorXX {

AngularQuad angular_from_string(std::string name) {
  std::transform(name.begin(), name.end(), name.begin(),
    [](unsigned char c){ return static_cast<char>(std::toupper(c)); });
  if(name == "AHRENSBEYLKIN") return AngularQuad::AhrensBeylkin;
  if(name == "AB")            return AngularQuad::AhrensBeylkin;
  if(name == "DELLEY")        return AngularQuad::Delley;
  if(name == "LEBEDEVLAIKOV") return AngularQuad::LebedevLaikov;
  if(name == "LEBEDEV")       return AngularQuad::LebedevLaikov;
  if(name == "LL")            return AngularQuad::LebedevLaikov;
  if(name == "WOMERSLEY")     return AngularQuad::Womersley;

  throw std::runtime_error("Unrecognized Angular Quadrature");
}

}
