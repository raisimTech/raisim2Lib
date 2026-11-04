#pragma once
// The town's apartment blocks, assembled from Poly Haven's modular urban apartments facade (CC0)
// as in the rayrai city example: corner pieces wrap each corner, 3 m bays of windows (and one door
// on the ground floor) fill the faces, and base, dado, cornice and crown trims run around them.
// Dark walls behind the glass read as unlit rooms. Burnt-out blocks have lost their crown and carry
// rubble on their roofs.

#include <map>

#include "common.hpp"

namespace tank {

struct Lot {
  double x0, y0, x1, y1;  // footprint, sides multiples of 3 m and at least 9 m
  double ground;          // the levelled ground under it
  int floors;
  bool ruined;
  int pattern;  // the window columns' pattern
};

class Town {
 public:
  static constexpr double kFloor0 = 0.55, kFloor = 3.0, kCrown = 0.75;

  // The top of a lot's walls (with the crown) above its ground.
  static double height(const Lot& lot) { return kFloor0 + kFloor * lot.floors + (lot.ruined ? 0.0 : kCrown); }

  void build(Viewer& viewer, const std::string& assets, const std::vector<Lot>& lots) {
    clear(viewer);
    folder_ = assets + "/tank_survival/buildings/apartments/";
    std::map<std::string, std::vector<raisin::InstancedVisuals::InstanceSpec>> batches;
    for (const auto& lot : lots) facades(lot, batches);
    for (auto& [module, instances] : batches) {
      const std::string name = "town " + module + " " + std::to_string(revision_++);
      auto visual = viewer.addInstancedVisuals(name, raisim::Shape::Mesh, glm::vec3(1), glm::vec4(1), glm::vec4(1),
                                               folder_ + "modules/" + module + ".gltf", true);
      visual->addInstances(instances);
      visual->setCastsShadows(true);
      names_.push_back(name);
    }
    shells(viewer, assets, lots);
  }

  void clear(Viewer& viewer) {
    for (const auto& name : names_) viewer.removeInstancedVisuals(name);
    names_.clear();
    for (const auto& name : meshes_) viewer.removeVisualObject(name);
    meshes_.clear();
  }

 private:
  using Batches = std::map<std::string, std::vector<raisin::InstancedVisuals::InstanceSpec>>;

  struct Face {
    double x, y, tx, ty, yaw, length;
  };

  static std::array<Face, 4> faces(const Lot& lot) {
    return {Face{lot.x0, lot.y0, 1, 0, 0.0, lot.x1 - lot.x0}, Face{lot.x1, lot.y0, 0, 1, 0.5 * kPi, lot.y1 - lot.y0},
            Face{lot.x1, lot.y1, -1, 0, kPi, lot.x1 - lot.x0}, Face{lot.x0, lot.y1, 0, -1, 1.5 * kPi, lot.y1 - lot.y0}};
  }

  // A module is modelled on x in [-width, 0] with its outer side towards -y: at arc length s of a
  // face it stands at the face's start + tangent * (s + width), turned by the face's yaw.
  static void place(Batches& batches, const Lot& lot, const Face& face, const std::string& module, double s, double z) {
    raisin::InstancedVisuals::InstanceSpec instance;
    instance.position = {float(face.x + face.tx * (s + 3.0)), float(face.y + face.ty * (s + 3.0)), float(lot.ground + z)};
    instance.orientation = {float(std::cos(0.5 * face.yaw)), 0.f, 0.f, float(std::sin(0.5 * face.yaw))};
    instance.scale = glm::vec3(1.f);
    batches[module].push_back(instance);
  }

  static void facades(const Lot& lot, Batches& batches) {
    static const std::array<std::array<const char*, 3>, 4> kPatterns{{{"large", "large", "large"},
                                                                       {"large", "small", "large"},
                                                                       {"double", "large", "double"},
                                                                       {"small", "double", "small"}}};
    const auto& pattern = kPatterns[size_t(lot.pattern) % kPatterns.size()];
    const double top = kFloor0 + kFloor * lot.floors;
    for (const auto& face : faces(lot)) {
      for (int floor = 0; floor < lot.floors; ++floor) place(batches, lot, face, "wall_standard_corner_large_01", 0.0, kFloor0 + kFloor * floor);
      place(batches, lot, face, "base_corner_large_01", 0.0, kFloor0 - 0.75);
      place(batches, lot, face, "dado_standard_corner_large_01", 0.0, kFloor0);
      place(batches, lot, face, "cornice_standard_corner_large_01", 0.0, kFloor0 + kFloor);
      if (!lot.ruined) place(batches, lot, face, "crown_standard_corner_large_01", 0.0, top);
      const int slots = int(std::lround((face.length - 6.0) / 3.0));
      for (int i = 0; i < slots; ++i) {
        const double s = 3.0 + 3.0 * i;
        const std::string column = pattern[size_t(i) % pattern.size()];
        place(batches, lot, face, "base_standard_01", s, kFloor0 - 0.75);
        place(batches, lot, face, "cornice_standard_standard_01", s, kFloor0 + kFloor);
        if (!lot.ruined) place(batches, lot, face, "crown_standard_standard_01", s, top);
        if (i == slots / 2) {
          const std::string size = column == "small" ? "small" : "large";
          place(batches, lot, face, "wall_door_centered_" + size + "_01", s, kFloor0);
          place(batches, lot, face, "dado_door_centered_" + size + "_01", s, kFloor0);
          place(batches, lot, face, "door_centered_" + size + "_01", s, kFloor0);
        } else {
          place(batches, lot, face, "wall_window_centered_" + column + "_01", s, kFloor0);
          place(batches, lot, face, "dado_standard_standard_01", s, kFloor0);
          place(batches, lot, face, "window_centered_" + column + "_01", s, kFloor0);
        }
        for (int floor = 1; floor < lot.floors; ++floor) {
          place(batches, lot, face, "wall_window_centered_" + column + "_02", s, kFloor0 + kFloor * floor);
          place(batches, lot, face, "window_centered_" + column + "_02", s, kFloor0 + kFloor * floor);
        }
      }
    }
  }

  // Roofs (bitumen, or rubble on the burnt-out blocks) and the dark rooms behind the facades.
  void shells(Viewer& viewer, const std::string& assets, const std::vector<Lot>& lots) {
    constexpr double kInset = 0.3;
    std::array<std::vector<::Vertex>, 3> vertices;
    std::array<std::vector<unsigned int>, 3> indices;
    const auto quad = [&](int layer, const std::array<glm::vec3, 4>& p, const glm::vec3& normal, double tile) {
      const auto base = unsigned(vertices[size_t(layer)].size());
      const glm::vec3 u = glm::normalize(p[1] - p[0]);
      for (const auto& point : p) {
        ::Vertex v;
        v.Position = point;
        v.Normal = normal;
        v.Tangent = glm::vec4(u, 1.f);
        const bool flat = std::abs(normal.z) > 0.5f;
        v.TexCoords = flat ? glm::vec2(point.x, point.y) / float(tile) : glm::vec2(glm::dot(point, u), point.z) / float(tile);
        vertices[size_t(layer)].push_back(v);
      }
      indices[size_t(layer)].insert(indices[size_t(layer)].end(), {base, base + 1, base + 2, base, base + 2, base + 3});
    };
    for (const auto& lot : lots) {
      const float x0 = float(lot.x0 + kInset), y0 = float(lot.y0 + kInset), x1 = float(lot.x1 - kInset), y1 = float(lot.y1 - kInset);
      const float bottom = float(lot.ground + kFloor0), top = float(lot.ground + kFloor0 + kFloor * lot.floors);
      const int roof = lot.ruined ? 1 : 0;
      const float z = lot.ruined ? top - 0.02f : top + 0.02f;
      quad(roof, {glm::vec3(float(lot.x0), float(lot.y0), z), glm::vec3(float(lot.x1), float(lot.y0), z),
                  glm::vec3(float(lot.x1), float(lot.y1), z), glm::vec3(float(lot.x0), float(lot.y1), z)},
           glm::vec3(0, 0, 1), lot.ruined ? 3.0 : 6.0);
      const std::array<glm::vec2, 4> c{glm::vec2(x0, y0), glm::vec2(x1, y0), glm::vec2(x1, y1), glm::vec2(x0, y1)};
      for (int k = 0; k < 4; ++k) {
        const glm::vec2 a = c[size_t(k)], b = c[size_t((k + 1) % 4)], d = b - a;
        const glm::vec3 inward = glm::normalize(glm::vec3(-d.y, d.x, 0.f));
        quad(2, {glm::vec3(a, bottom), glm::vec3(b, bottom), glm::vec3(b, top), glm::vec3(a, top)}, inward, 3.0);
      }
    }
    const auto texture = [&](const std::string& file, bool color) {
      return Viewer::loadTextureWithTiling((assets + "/tank_survival/" + file).c_str(), color);
    };
    std::array<raisin::Material, 3> materials{raisin::Material::pbr("roof", glm::vec4(0.8f, 0.8f, 0.8f, 1.f), 0.f, 0.9f),
                                               raisin::Material::pbr("rubble", glm::vec4(0.55f, 0.52f, 0.5f, 1.f), 0.f, 0.95f),
                                               raisin::Material::pbr("rooms", glm::vec4(0.025f, 0.024f, 0.023f, 1.f), 0.f, 0.9f)};
    materials[0].albedoMap = texture("buildings/apartments/textures/bitumen_diff_1k.jpg", true);
    materials[0].normalMap = texture("buildings/apartments/textures/bitumen_nor_gl_1k.jpg", false);
    materials[1].albedoMap = texture("ground/rubble_diffuse_1k.jpg", true);
    materials[1].normalMap = texture("ground/rubble_nor_gl_1k.jpg", false);
    materials[2].doubleSided = true;
    for (int layer = 0; layer < 3; ++layer) {
      if (indices[size_t(layer)].empty()) continue;
      auto meshes = std::make_shared<std::vector<std::shared_ptr<::OpenGLMesh>>>();
      meshes->push_back(std::make_shared<::OpenGLMesh>(std::move(vertices[size_t(layer)]), std::move(indices[size_t(layer)])));
      meshes->front()->material = materials[size_t(layer)];
      meshes_.push_back("town shell " + std::to_string(layer) + " " + std::to_string(revision_++));
      viewer.addVisualCustomMesh(meshes_.back(), meshes, layer == 2 ? materials[2].baseColorFactor : glm::vec4(1.f), false);
    }
  }

  std::string folder_;
  std::vector<std::string> names_, meshes_;
  int revision_ = 0;
};

}  // namespace tank
