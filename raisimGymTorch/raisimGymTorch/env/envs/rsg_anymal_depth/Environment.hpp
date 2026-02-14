//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <set>
#include <vector>
#include <cmath>
#include <memory>
#include <array>
#include <limits>
#include <SDL.h>
#include <omp.h>
#include "raisim/object/terrain/HeightMap.hpp"
#include "raisim/Terrain.hpp"
#include "raisim/sensors/DepthSensor.hpp"
#include "rayrai/RayraiWindow.hpp"
#include "rayrai/Camera.hpp"
#include "rayrai/example_common.hpp"
#include "../../RaisimGymEnv.hpp"

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:
  static constexpr bool kStaticSchedule = true; // Static schedule keeps each env on a consistent thread for per-env OpenGL context usage.

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable), normDist_(0, 1) {

    /// create world
    world_ = std::make_unique<raisim::World>();

    /// add objects
    anymal_ = world_->addArticulatedSystem(resourceDir_+"/anymal/urdf/anymal_depth.urdf");
    anymal_->setName("anymal");
    anymal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    terrainProperties_.frequency = 0.2;
    terrainProperties_.zScale = 0.5;
    terrainProperties_.xSize = 20.0;
    terrainProperties_.ySize = 20.0;
    terrainProperties_.xSamples = 200;
    terrainProperties_.ySamples = 200;
    terrainProperties_.fractalOctaves = 3;
    terrainProperties_.fractalLacunarity = 2.0;
    terrainProperties_.fractalGain = 0.25;
    terrainProperties_.heightOffset = 0.0;
    heightmap_ = world_->addHeightMap(0.0, 0.0, terrainProperties_);

    /// get robot data
    gcDim_ = anymal_->getGeneralizedCoordinateDim();
    gvDim_ = anymal_->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);

    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.90, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(50.0);
    jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(0.2);
    anymal_->setPdGains(jointPgain, jointDgain);
    anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    depthCam_ = anymal_->getSensorSet("d455_front")->getSensor<raisim::DepthCamera>("depth");
    RSFATAL_IF(!depthCam_, "Failed to find depth camera sensor set d455_front/depth.");
    depthCam_->setMeasurementSource(raisim::Sensor::MeasurementSource::MANUAL);
    depthCam_->updatePose();
    const auto& depthProp = depthCam_->getProperties();
    depthWidth_ = depthProp.width;
    depthHeight_ = depthProp.height;
    depthSize_ = depthWidth_ * depthHeight_;
    depthMin_ = depthProp.clipNear;
    depthMax_ = depthProp.clipFar;
    if (!cfg_["depth"].IsNone()) {
      if (!cfg_["depth"]["min"].IsNone())
        depthMin_ = cfg_["depth"]["min"].As<double>();
      if (!cfg_["depth"]["max"].IsNone())
        depthMax_ = cfg_["depth"]["max"].As<double>();
    }
    depthScale_ = (depthMax_ > depthMin_) ? (1.0 / (depthMax_ - depthMin_)) : 1.0;
    depthBuffer_.resize(depthSize_);

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = proprioDim_ + depthSize_;
    criticObDim_ = proprioDim_ + heightPatchDim_;
    actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);
    criticObDouble_.setZero(criticObDim_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    double action_std;
    READ_YAML(double, action_std, cfg_["action_std"]) /// example of reading params from the config
    actionStd_.setConstant(action_std);

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);

    /// indices of links that should not make contact with ground
    footIndices_.insert(anymal_->getBodyIdx("LF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("LH_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RH_SHANK"));

    /// visualize if it is the first environment (Rayrai)
    if (visualizable_) {
      app_ = std::make_unique<ExampleApp>();
      if (app_->init("rayrai_anymal_depth", 1280, 720)) {
        viewerWindow_ = std::make_shared<raisin::RayraiWindow>(*world_, 1280, 720);
        viewerWindow_->setBackgroundColor({0.14f, 0.14f, 0.18f, 1.0f});
        viewerDepthCamera_ = std::make_shared<raisin::Camera>(*depthCam_);
        renderEnabled_ = true;
      } else {
        app_.reset();
      }
    }
  }

  void init() final { }

  void reset() final {
    allowRayrai_ = false; // Avoid GL work on the main thread during reset.
    randomizeState();
    updateObservation();
    allowRayrai_ = true;
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;

    anymal_->setPdTarget(pTarget_, vTarget_);

    for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
      world_->integrate();
    }

    updateObservation();

    rewards_.record("torque", anymal_->getGeneralizedForce().squaredNorm());
    rewards_.record("forwardVel", std::min(4.0, bodyLinearVel_[0]));

    return rewards_.sum();
  }

  void updateObservation() {
    anymal_->getState(gc_, gv_);
    Vec<4> quat;
    Mat<3,3> rot;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    quatToRotMat(quat, rot);
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);

    obDouble_.segment(0, proprioDim_) << gc_[2], /// body height
        rot.e().row(2).transpose(), /// body orientation
        gc_.tail(12), /// joint angles
        bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity
        gv_.tail(12); /// joint velocity

    if (allowRayrai_) {
      const int currentThread = omp_get_thread_num();
      if (!rayraiContextReady_ || rayraiThreadId_ != currentThread) {
        // Must create and bind the GL context on the same worker thread that will render.
        raisin::RayraiWindow::createOffscreenGlContext(rayraiSdlWindow_, rayraiGlContext_);
        raisin::RayraiWindow::makeOffscreenContextCurrent(rayraiSdlWindow_, rayraiGlContext_);
        rayraiWindow_ = std::make_shared<raisin::RayraiWindow>(*world_, depthWidth_, depthHeight_);
        rayraiDepthCamera_ = std::make_shared<raisin::Camera>(*depthCam_);
        rayraiContextReady_ = true;
        rayraiThreadId_ = currentThread;
      } else {
        raisin::RayraiWindow::makeOffscreenContextCurrent(rayraiSdlWindow_, rayraiGlContext_);
      }
      rayraiWindow_->renderWithExternalCamera(*depthCam_, *rayraiDepthCamera_, {});
      rayraiWindow_->renderDepthPlaneDistance(*depthCam_, *rayraiDepthCamera_);
      rayraiDepthCamera_->getRawImage(*depthCam_,
                                      raisin::Camera::SensorStorageMode::CUSTOM_BUFFER,
                                      depthBuffer_.data(),
                                      depthBuffer_.size(),
                                      false);
    } else {
      std::fill(depthBuffer_.begin(), depthBuffer_.end(), static_cast<float>(depthMax_));
    }

    if (allowRayrai_ && renderEnabled_ && app_ && viewerWindow_ && viewerDepthCamera_) {
      if (app_->window && app_->context) {
        SDL_GL_MakeCurrent(app_->window, app_->context);
      }
      app_->processEvents();
      if (!app_->quit) {
        viewerWindow_->renderWithExternalCamera(*depthCam_, *viewerDepthCamera_, {});
        viewerWindow_->renderDepthPlaneDistance(*depthCam_, *viewerDepthCamera_);

        app_->beginFrame();
        app_->renderViewer(*viewerWindow_);

        const auto& prop = depthCam_->getProperties();
        const int w = std::max(1, prop.width);
        const int h = std::max(1, prop.height);
        const float aspect = float(w) / float(h);

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        constexpr float kDepthScale = 6.0f;
        ImGui::SetNextWindowSize(ImVec2(static_cast<float>(w) * kDepthScale,
                                        static_cast<float>(h) * kDepthScale),
                                 ImGuiCond_Always);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::Begin("Depth Sensor", nullptr,
                     ImGuiWindowFlags_NoDecoration |
                         ImGuiWindowFlags_NoResize |
                         ImGuiWindowFlags_NoScrollbar |
                         ImGuiWindowFlags_NoScrollWithMouse);
        ImVec2 size = ImGui::GetContentRegionAvail();
        ImTextureID tex = (ImTextureID)(intptr_t)viewerDepthCamera_->getLinearDepthTexture();
        ImGui::Image(tex, size, ImVec2(0, 1), ImVec2(1, 0));
        ImGui::End();
        ImGui::PopStyleVar();

        app_->endFrame();
      }
    }
    const auto& depth = depthBuffer_;
    for (int i = 0; i < depthSize_; ++i) {
      double d = depth[i];
      if (!std::isfinite(d)) {
        d = depthMax_;
      }
      if (d < depthMin_) d = depthMin_;
      if (d > depthMax_) d = depthMax_;
      obDouble_[proprioDim_ + i] = (d - depthMin_) * depthScale_;
    }
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obDouble_.cast<float>();
  }

  void observeCritic(Eigen::Ref<EigenVec> ob) override {
    updateCriticObservation();
    ob = criticObDouble_.cast<float>();
  }

  void turnOnVisualization() override { renderEnabled_ = true; }
  void turnOffVisualization() override { renderEnabled_ = false; }
  void startRecordingVideo(const std::string&) override { }
  void stopRecordingVideo() override { }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = float(terminalRewardCoeff_);

    /// if the contact body is not feet
    for(auto& contact: anymal_->getContacts())
      if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end())
        return true;

    terminalReward = 0.f;
    return false;
  }

  void curriculumUpdate() override {
    heightmapZScale_ += 0.3;
    terrainProperties_.zScale = heightmapZScale_;
    heightmap_->update(0.0, 0.0, terrainProperties_);
  };

 private:
  static ImVec2 fitToAspect(ImVec2 avail, float aspect) {
    if (aspect <= 0.0f)
      return avail;
    ImVec2 size = avail;
    size.y = size.x / aspect;
    if (size.y > avail.y) {
      size.y = avail.y;
      size.x = size.y * aspect;
    }
    return size;
  }

  void randomizeState() {
    constexpr double kMaxAngleRad = 0.3490658504; // ~20 deg
    std::uniform_real_distribution uni(-kMaxAngleRad, kMaxAngleRad);

    gc_ = gc_init_;
    gv_.setZero();

    const double roll = uni(gen_);
    const double pitch = uni(gen_);
    const double yaw = uni(gen_);
    const Eigen::AngleAxisd rollA(roll, Eigen::Vector3d::UnitX());
    const Eigen::AngleAxisd pitchA(pitch, Eigen::Vector3d::UnitY());
    const Eigen::AngleAxisd yawA(yaw, Eigen::Vector3d::UnitZ());
    const Eigen::Quaterniond q = yawA * pitchA * rollA;
    gc_[3] = q.w();
    gc_[4] = q.x();
    gc_[5] = q.y();
    gc_[6] = q.z();

    for (int i = 0; i < nJoints_; ++i)
      gc_[7 + i] += uni(gen_);

    anymal_->setState(gc_, gv_);

    const std::array<const char*, 4> footFrames = {
        "LF_ADAPTER_TO_FOOT",
        "RF_ADAPTER_TO_FOOT",
        "LH_ADAPTER_TO_FOOT",
        "RH_ADAPTER_TO_FOOT"};
    double minClearance = std::numeric_limits<double>::infinity();
    for (const auto* name : footFrames) {
      Vec<3> pos;
      anymal_->getFramePosition(name, pos);
      const double terrainHeight = heightmap_->getHeight(pos[0], pos[1]);
      // std::cout<<"terrainHeight="<<terrainHeight<< "  " << "foot height "<<pos[2]<<std::endl;
      minClearance = std::min(minClearance, pos[2] - 0.03 - terrainHeight);
    }

    gc_[2] -= minClearance;
    anymal_->setState(gc_, gv_);
  }

  void updateCriticObservation() {
    criticObDouble_.segment(0, proprioDim_) = obDouble_.segment(0, proprioDim_);

    const double centerX = gc_[0];
    const double centerY = gc_[1];
    const double half = heightPatchSize_ * 0.5;
    const double step = heightPatchSize_ / double(heightPatchSamples_ - 1);

    int idx = 0;
    for (int ix = 0; ix < heightPatchSamples_; ++ix) {
      const double x = centerX - half + step * ix;
      for (int iy = 0; iy < heightPatchSamples_; ++iy) {
        const double y = centerY - half + step * iy;
        criticObDouble_[proprioDim_ + idx] = heightmap_->getHeight(x, y);
        ++idx;
      }
    }
  }

  int gcDim_, gvDim_, nJoints_;
  int proprioDim_ = 34;
  int depthWidth_ = 0;
  int depthHeight_ = 0;
  int depthSize_ = 0;
  int heightPatchSamples_ = 50;
  double heightPatchSize_ = 2.0;
  int heightPatchDim_ = heightPatchSamples_ * heightPatchSamples_;
  bool visualizable_ = false;
  ArticulatedSystem* anymal_;
  DepthCamera* depthCam_ = nullptr;
  std::shared_ptr<raisin::RayraiWindow> rayraiWindow_;
  std::shared_ptr<raisin::Camera> rayraiDepthCamera_;
  std::vector<float> depthBuffer_;
  SDL_Window* rayraiSdlWindow_ = nullptr;
  SDL_GLContext rayraiGlContext_ = nullptr;
  bool rayraiContextReady_ = false;
  int rayraiThreadId_ = -1;
  bool allowRayrai_ = true;
  bool renderEnabled_ = false;
  std::unique_ptr<ExampleApp> app_;
  std::shared_ptr<raisin::RayraiWindow> viewerWindow_;
  std::shared_ptr<raisin::Camera> viewerDepthCamera_;
  raisim::HeightMap* heightmap_ = nullptr;
  raisim::TerrainProperties terrainProperties_;
  double heightmapZScale_ = 0.0;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  double terminalRewardCoeff_ = -10.;
  double depthMin_ = 0.;
  double depthMax_ = 0.;
  double depthScale_ = 1.0;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_, criticObDouble_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> footIndices_;

  /// these variables are not in use. They are placed to show you how to create a random number sampler.
  std::normal_distribution<> normDist_;
  thread_local static std::mt19937 gen_;
};
thread_local std::mt19937 ENVIRONMENT::gen_;

}
