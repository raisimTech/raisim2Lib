//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#include "rayrai_tcp_viewer_hint.hpp"

#include <raisim/RaisimServer.hpp>
#include <raisim/World.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include "example_resources.hpp"

namespace {

struct RotationMatrix {
  double r[3][3];
};

RotationMatrix eulerRotation(double roll, double pitch, double yaw) {
  const double cr = std::cos(roll);
  const double sr = std::sin(roll);
  const double cp = std::cos(pitch);
  const double sp = std::sin(pitch);
  const double cy = std::cos(yaw);
  const double sy = std::sin(yaw);

  return {{{cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr},
           {sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr},
           {-sp, cp * sr, cp * cr}}};
}

RotationMatrix randomCubeRotation(std::mt19937& rng) {
  constexpr double pi = 3.14159265358979323846;
  std::uniform_real_distribution<double> fullTurn(-pi, pi);
  return eulerRotation(fullTurn(rng), fullTurn(rng), fullTurn(rng));
}

std::array<double, 3> rotatePoint(const RotationMatrix& rotation,
                                  const std::array<double, 3>& point) {
  return {rotation.r[0][0] * point[0] + rotation.r[0][1] * point[1] + rotation.r[0][2] * point[2],
          rotation.r[1][0] * point[0] + rotation.r[1][1] * point[1] + rotation.r[1][2] * point[2],
          rotation.r[2][0] * point[0] + rotation.r[2][1] * point[1] + rotation.r[2][2] * point[2]};
}

double rotatedCubeVerticalHalfExtent(double size, const RotationMatrix& rotation) {
  const double h = 0.5 * size;
  const std::array<std::array<double, 3>, 8> vertices = {{{-h, -h, -h},
                                                          {h, -h, -h},
                                                          {h, h, -h},
                                                          {-h, h, -h},
                                                          {-h, -h, h},
                                                          {h, -h, h},
                                                          {h, h, h},
                                                          {-h, h, h}}};

  double maxAbsZ = 0.0;
  for (const auto& vertex : vertices) {
    const auto rotated = rotatePoint(rotation, vertex);
    maxAbsZ = std::max(maxAbsZ, std::abs(rotated[2]));
  }
  return maxAbsZ;
}

void makeClothGrid(int n,
                   double width,
                   double z,
                   std::vector<raisim::Vec<3>>& vertices,
                   std::vector<raisim::DeformableObject::Triangle>& triangles) {
  vertices.clear();
  triangles.clear();
  const double step = width / static_cast<double>(n - 1);
  const double half = 0.5 * width;

  for (int y = 0; y < n; ++y) {
    for (int x = 0; x < n; ++x) {
      vertices.push_back({-half + step * static_cast<double>(x),
                          -half + step * static_cast<double>(y),
                          z});
    }
  }

  auto idx = [n](int x, int y) {
    return static_cast<unsigned int>(y * n + x);
  };

  for (int y = 0; y + 1 < n; ++y) {
    for (int x = 0; x + 1 < n; ++x) {
      const unsigned int i00 = idx(x, y);
      const unsigned int i10 = idx(x + 1, y);
      const unsigned int i01 = idx(x, y + 1);
      const unsigned int i11 = idx(x + 1, y + 1);
      triangles.push_back({i00, i10, i11});
      triangles.push_back({i00, i11, i01});
    }
  }
}

bool writeClosedCubeObj(const std::string& path,
                        double size,
                        const RotationMatrix& rotation) {
  const double h = 0.5 * size;
  std::ofstream out(path);
  if (!out) {
    return false;
  }

  const std::array<std::array<double, 3>, 8> vertices = {{{-h, -h, -h},
                                                          {h, -h, -h},
                                                          {h, h, -h},
                                                          {-h, h, -h},
                                                          {-h, -h, h},
                                                          {h, -h, h},
                                                          {h, h, h},
                                                          {-h, h, h}}};

  for (const auto& vertex : vertices) {
    const auto rotated = rotatePoint(rotation, vertex);
    out << "v " << rotated[0] << " " << rotated[1] << " " << rotated[2] << "\n";
  }

  out << "f 1 3 2\n";
  out << "f 1 4 3\n";
  out << "f 5 6 7\n";
  out << "f 5 7 8\n";
  out << "f 1 2 6\n";
  out << "f 1 6 5\n";
  out << "f 2 3 7\n";
  out << "f 2 7 6\n";
  out << "f 3 4 8\n";
  out << "f 3 8 7\n";
  out << "f 4 1 5\n";
  out << "f 4 5 8\n";
  return true;
}


struct CubeStackPose {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double roll = 0.0;
  double pitch = 0.0;
  double yaw = 0.0;
  double verticalHalfExtent = 0.0;
};

double cubeVerticalHalfExtent(double size, double roll, double pitch) {
  const double half = 0.5 * size;
  const double sr = std::sin(roll);
  const double cr = std::cos(roll);
  const double sp = std::sin(pitch);
  const double cp = std::cos(pitch);
  return half * (std::abs(sp) + std::abs(cp * sr) + std::abs(cp * cr));
}

double cubeStackSurfaceSpacing(double size, int resolution) {
  return size / static_cast<double>(std::max(2, resolution) - 1);
}

double cubeStackCollisionRadius(double size, int resolution) {
  (void) resolution;
  return std::max(0.004, 0.04 * size);
}

int cubeStackCollisionSafeResolution(double size, double collisionRadius) {
  const double maxSpacing = std::max(1e-6, 2.0 * collisionRadius);
  return std::max(2, static_cast<int>(std::ceil(size / maxSpacing)) + 1);
}

int cubeStackEffectiveResolution(int requestedResolution, double size, double collisionRadius) {
  return std::max(std::max(2, requestedResolution),
                  cubeStackCollisionSafeResolution(size, collisionRadius));
}

double cubeStackClearance(double collisionRadius) {
  return 2.0 * collisionRadius + 0.02;
}

raisim::Vec<3> rotateXyz(const raisim::Vec<3>& p, double roll, double pitch, double yaw) {
  const double cr = std::cos(roll);
  const double sr = std::sin(roll);
  const double cp = std::cos(pitch);
  const double sp = std::sin(pitch);
  const double cy = std::cos(yaw);
  const double sy = std::sin(yaw);

  const double x1 = p[0];
  const double y1 = cr * p[1] - sr * p[2];
  const double z1 = sr * p[1] + cr * p[2];
  const double x2 = cp * x1 + sp * z1;
  const double y2 = y1;
  const double z2 = -sp * x1 + cp * z1;
  return {cy * x2 - sy * y2, sy * x2 + cy * y2, z2};
}

void makeCubeSurface(int resolution,
                     double size,
                     const raisim::Vec<3>& center,
                     double roll,
                     double pitch,
                     double yaw,
                     std::vector<raisim::Vec<3>>& vertices,
                     std::vector<raisim::DeformableObject::Triangle>& triangles) {
  vertices.clear();
  triangles.clear();
  const int n = std::max(2, resolution);
  std::vector<int> ids(static_cast<size_t>(n * n * n), -1);
  auto gridId = [n](int i, int j, int k) {
    return static_cast<size_t>((k * n + j) * n + i);
  };

  const double step = size / static_cast<double>(n - 1);
  const double half = 0.5 * size;
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        const bool boundary =
            (i == 0 || i == n - 1 || j == 0 || j == n - 1 || k == 0 || k == n - 1);
        if (!boundary) {
          continue;
        }
        raisim::Vec<3> p{-half + step * static_cast<double>(i),
                         -half + step * static_cast<double>(j),
                         -half + step * static_cast<double>(k)};
        p = rotateXyz(p, roll, pitch, yaw) + center;
        ids[gridId(i, j, k)] = static_cast<int>(vertices.size());
        vertices.push_back(p);
      }
    }
  }

  auto idx = [&](int i, int j, int k) {
    return static_cast<unsigned int>(ids[gridId(i, j, k)]);
  };
  auto addQuad = [&](unsigned int a, unsigned int b, unsigned int c, unsigned int d) {
    triangles.push_back(raisim::DeformableObject::Triangle{a, c, b});
    triangles.push_back(raisim::DeformableObject::Triangle{a, d, c});
  };

  for (int a = 0; a + 1 < n; ++a) {
    for (int b = 0; b + 1 < n; ++b) {
      addQuad(idx(a, b, 0), idx(a + 1, b, 0), idx(a + 1, b + 1, 0), idx(a, b + 1, 0));
      addQuad(idx(a, b, n - 1), idx(a, b + 1, n - 1), idx(a + 1, b + 1, n - 1), idx(a + 1, b, n - 1));
      addQuad(idx(a, 0, b), idx(a, 0, b + 1), idx(a + 1, 0, b + 1), idx(a + 1, 0, b));
      addQuad(idx(a, n - 1, b), idx(a + 1, n - 1, b), idx(a + 1, n - 1, b + 1), idx(a, n - 1, b + 1));
      addQuad(idx(0, a, b), idx(0, a + 1, b), idx(0, a + 1, b + 1), idx(0, a, b + 1));
      addQuad(idx(n - 1, a, b), idx(n - 1, a, b + 1), idx(n - 1, a + 1, b + 1), idx(n - 1, a + 1, b));
    }
  }
}

CubeStackPose cubeStackPose(int index, double size, double collisionRadius) {
  constexpr double groundClearance = 0.20;
  const double cubeClearance = cubeStackClearance(collisionRadius);

  CubeStackPose pose;
  double top = groundClearance;
  for (int i = 0; i <= index; ++i) {
    CubeStackPose current;
    current.x = (i % 2 == 0 ? -0.035 : 0.035);
    current.y = ((i / 2) % 2 == 0 ? -0.025 : 0.025);
    current.roll = 0.17 * static_cast<double>(i + 1);
    current.pitch = 0.11 * static_cast<double>(i + 2);
    current.yaw = 0.23 * static_cast<double>(i + 3);
    current.verticalHalfExtent = cubeVerticalHalfExtent(size, current.roll, current.pitch);
    current.z = top + current.verticalHalfExtent;
    top = current.z + current.verticalHalfExtent + cubeClearance;
    pose = current;
  }
  return pose;
}

bool writeObjMesh(const std::string& path,
                  const std::vector<raisim::Vec<3>>& vertices,
                  const std::vector<raisim::DeformableObject::Triangle>& triangles) {
  std::ofstream out(path);
  if (!out) {
    return false;
  }
  for (const auto& vertex : vertices) {
    out << "v " << vertex[0] << " " << vertex[1] << " " << vertex[2] << "\n";
  }
  for (const auto& tri : triangles) {
    out << "f " << (tri[0] + 1) << " " << (tri[1] + 1) << " " << (tri[2] + 1) << "\n";
  }
  return true;
}

}  // namespace

int main(int argc, char* argv[]) {
  raisim::World::setActivationKey(exampleRscPath(argv[0], "activation.raisim"));

  raisim::World world;
  world.setTimeStep(0.001);
  world.setGravity({0.0, 0.0, -9.81});
  world.addGround(0.0, "ground");
  world.setMaterialPairProp("ground", "deformable_cube", 0.9, 0.75, 0.02);
  world.setMaterialPairProp("deformable_cube", "deformable_cube", 0.9, 0.65, 0.02);

  auto* sphere = world.addSphere(0.28, 1.0, "default");
  sphere->setName("sphere");
  sphere->setPosition(0.0, 0.0, 0.88);
  sphere->setBodyType(raisim::BodyType::STATIC);

  raisim::DeformableObject::Material clothMaterial;
  clothMaterial.totalMass = 1.5;
  clothMaterial.distanceCompliance = 5.0e-2;
  clothMaterial.bendCompliance = 1.0e3;  // a soft fabric: bending modulus 1e-3 N m
  clothMaterial.damping = 0.002;
  clothMaterial.airDrag = 2.0;
  clothMaterial.collisionRadius = 0.01;
  clothMaterial.iterations = 3;

  std::vector<raisim::Vec<3>> clothVertices;
  std::vector<raisim::DeformableObject::Triangle> clothTriangles;
  makeClothGrid(48, 1.8, 1.85, clothVertices, clothTriangles);
  auto* cloth = world.addDeformableCloth(clothVertices, clothTriangles, {}, clothMaterial);
  cloth->setName("explicit_vertex_cloth");

  constexpr int cubeCount = 8;
  constexpr int requestedResolution = 4;
  constexpr double cubeSize = 0.30;
  constexpr double cubeStackXOffset = 1.55;
  int cubeResolution = requestedResolution;
  double cubeCollisionRadius = cubeStackCollisionRadius(cubeSize, cubeResolution);
  cubeResolution = cubeStackEffectiveResolution(requestedResolution, cubeSize, cubeCollisionRadius);
  const double cubeParticleSpacing = cubeStackSurfaceSpacing(cubeSize, cubeResolution);
  const double apiParticleSpacing = 1.5 * cubeParticleSpacing;
  cubeCollisionRadius = std::max(cubeCollisionRadius, 0.58 * apiParticleSpacing);

  raisim::DeformableObject::Material cubeMaterial;
  cubeMaterial.totalMass = 1.0;
  cubeMaterial.distanceCompliance = 7.5e-5;
  cubeMaterial.damping = 0.004;
  cubeMaterial.collisionRadius = cubeCollisionRadius;
  cubeMaterial.iterations = 10;
  cubeMaterial.substeps = 1;

  raisim::DeformableObject::MeshBuildOptions cubeBuild;
  cubeBuild.particles.mode = raisim::DeformableObject::MeshParticleOptions::Mode::Surface;
  cubeBuild.particles.spacing = apiParticleSpacing;
  cubeBuild.internalStruts.mode =
      raisim::DeformableObject::InternalStrutOptions::Mode::PairsWithinRadius;
  cubeBuild.internalStruts.radius = 2.05 * cubeParticleSpacing;

  std::vector<std::string> cubeObjFiles;
  const std::filesystem::path cubeObjPrefix =
      std::filesystem::temp_directory_path() /
      ("raisim_deformable_cube_stack_example_" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_");
  size_t totalCubeParticles = 0;
  size_t totalCubeTriangles = 0;
  for (int i = 0; i < cubeCount; ++i) {
    const CubeStackPose pose = cubeStackPose(i, cubeSize, cubeCollisionRadius);
    const std::string cubeObj = (cubeObjPrefix.string() + std::to_string(i) + ".obj");
    std::vector<raisim::Vec<3>> cubeVertices;
    std::vector<raisim::DeformableObject::Triangle> cubeTriangles;
    makeCubeSurface(cubeResolution, cubeSize, {0.0, 0.0, 0.0}, pose.roll, pose.pitch, pose.yaw,
                    cubeVertices, cubeTriangles);
    if (!writeObjMesh(cubeObj, cubeVertices, cubeTriangles)) {
      std::cerr << "Could not write temporary OBJ file: " << cubeObj << std::endl;
      return 1;
    }

    cubeObjFiles.push_back(cubeObj);
    auto* cube = world.addDeformableObject(cubeObj, cubeMaterial, cubeBuild, {}, "deformable_cube");
    cube->setName("benchmark_surface_cube_" + std::to_string(i));
    cube->setPositionOffset({pose.x + cubeStackXOffset, pose.y, pose.z});
    totalCubeParticles += cube->getNumParticles();
    totalCubeTriangles += cube->getTriangles().size();
  }

  // an elastic solid: a filled mesh with a Young's modulus is cut into tetrahedra, here a tilted
  // block of soft rubber dropped on the other side of the cloth
  raisim::DeformableObject::Material rubberMaterial;
  rubberMaterial.totalMass = 2.0;
  rubberMaterial.youngsModulus = 5.0e4;
  rubberMaterial.poissonRatio = 0.45;
  rubberMaterial.damping = 0.002;
  rubberMaterial.iterations = 6;

  raisim::DeformableObject::MeshBuildOptions rubberBuild;
  rubberBuild.particles.mode = raisim::DeformableObject::MeshParticleOptions::Mode::Filled;
  rubberBuild.particles.spacing = 0.04;

  const std::string rubberObj = cubeObjPrefix.string() + "rubber.obj";
  if (!writeClosedCubeObj(rubberObj, 0.36, eulerRotation(0.35, 0.25, 0.4))) {
    std::cerr << "Could not write temporary OBJ file: " << rubberObj << std::endl;
    return 1;
  }
  cubeObjFiles.push_back(rubberObj);
  auto* rubber = world.addDeformableObject(rubberObj, rubberMaterial, rubberBuild, {}, "deformable_cube");
  rubber->setName("elastic_solid_block");
  rubber->setPositionOffset({-1.55, 0.0, 0.7});

  std::cout << "deformable_objects: cloth_particles=" << cloth->getNumParticles()
            << " cloth_triangles=" << cloth->getTriangles().size()
            << " cube_particles=" << totalCubeParticles
            << " cube_triangles=" << totalCubeTriangles
            << " cube_count=" << cubeCount
            << " cube_resolution=" << cubeResolution
            << " requested_cube_resolution=" << requestedResolution
            << " cloth_iterations=" << clothMaterial.iterations
            << " cube_iterations=" << cubeMaterial.iterations
            << " cube_compliance=" << cubeMaterial.distanceCompliance
            << " cube_collision_radius=" << cubeMaterial.collisionRadius
            << " solid_particles=" << rubber->getNumParticles()
            << " solid_tetrahedra=" << rubber->getTetrahedra().size()
            << std::endl;

  raisim::RaisimServer server(&world);
  server.launchServer();
  raisim_examples::warnIfNoClientConnected(server);

  bool connectedOnce = false;
  while (!server.isConnected()) {
    server.waitForNewClients(10000);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  connectedOnce = true;

  server.setCameraPositionAndLookAt({3.2, -3.2, 2.5}, {0.85, 0.0, 1.25});
  server.focusOn(cloth);

  const auto wallStart = std::chrono::steady_clock::now();
  const double simStart = world.getWorldTime();
  while (server.isConnected() || !connectedOnce) {
    server.integrateWorldThreadSafe();

    const double targetElapsed = world.getWorldTime() - simStart;
    const auto targetWallTime =
        wallStart + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                        std::chrono::duration<double>(targetElapsed));
    const auto now = std::chrono::steady_clock::now();
    if (targetWallTime > now) {
      std::this_thread::sleep_until(targetWallTime);
    }
  }

  server.killServer();
  for (const auto& cubeObj : cubeObjFiles) {
    std::remove(cubeObj.c_str());
  }
  return 0;
}
