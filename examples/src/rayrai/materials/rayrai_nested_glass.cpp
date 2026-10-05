// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
#include "rayrai/example_common.hpp"
#include "rayrai/Visuals.hpp"
#include "rayrai/TextureBindingCache.hpp"
#include "glass_geometry.hpp"
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

namespace {
using namespace raisin; using namespace gl;
struct Options {
  int width=1280,height=800,samples=10,bounces=10,frames=128,warmup=8,objects=0,msaa=1,layers=0;
  bool benchmark=false,animate=false,opaqueOnly=false;
  float roughness=0.f;
  GeometryRefractionBackend backend=GeometryRefractionBackend::Automatic;
  std::string output;
  std::filesystem::path recording;
};
Options parse(int argc,char** argv) {
  Options o;
  for(int i=1;i<argc;++i) {
    const std::string a=argv[i]; auto p=a.find('='); auto k=a.substr(0,p),v=p==a.npos?"":a.substr(p+1);
    if(k=="--out") o.output=v;
    else if(k=="--record") o.recording=v;
    else if(k=="--backend") {
      if(v=="auto") o.backend=GeometryRefractionBackend::Automatic;
      else if(v=="portable") o.backend=GeometryRefractionBackend::Portable;
      else if(v=="vulkan") o.backend=GeometryRefractionBackend::VulkanRayQuery;
      else throw std::runtime_error("--backend must be auto, portable or vulkan");
    }
    else if(k=="--width") o.width=std::stoi(v); else if(k=="--height") o.height=std::stoi(v);
    else if(k=="--samples") o.samples=std::stoi(v); else if(k=="--bounces") o.bounces=std::stoi(v);
    else if(k=="--frames") o.frames=std::stoi(v); else if(k=="--warmup") o.warmup=std::stoi(v);
    else if(k=="--objects") o.objects=std::stoi(v); else if(k=="--msaa") o.msaa=std::stoi(v);
    else if(k=="--layers") o.layers=std::stoi(v);
    else if(k=="--benchmark") o.benchmark=true;
    else if(k=="--animate") o.animate=true;
    else if(k=="--roughness") o.roughness=std::stof(v);
    else if(k=="--opaque-only") o.opaqueOnly=true;
    else throw std::runtime_error("Unknown option: "+a);
  }
  if(o.width<16||o.height<16||o.width>8192||o.height>8192||o.samples<1||o.samples>64||
      o.bounces<1||o.bounces>128||o.frames<1||o.warmup<0||o.objects<0||o.objects>1024||
      o.layers<0||o.layers>16||(o.layers>0&&o.objects>0)||
      !std::isfinite(o.roughness)||o.roughness<0||o.roughness>1||
      (o.msaa!=1&&o.msaa!=2&&o.msaa!=4&&o.msaa!=8)) throw std::runtime_error("Invalid dimensions, samples, bounces or counts");
  return o;
}
void save(Camera& camera,const Options& o) {
  std::vector<unsigned char> pixels(size_t(o.width)*o.height*4);
  glBindTexture(GL_TEXTURE_2D,camera.getFinalTexture()); glGetTexImage(GL_TEXTURE_2D,0,GL_RGBA,GL_UNSIGNED_BYTE,pixels.data());
  resetTextureBindingCache();
  for(int y=0;y<o.height/2;++y) std::swap_ranges(pixels.begin()+y*o.width*4,pixels.begin()+(y+1)*o.width*4,
      pixels.begin()+(o.height-1-y)*o.width*4);
  auto parent=std::filesystem::path(o.output).parent_path(); if(!parent.empty()) std::filesystem::create_directories(parent);
  if(!stbi_write_png(o.output.c_str(),o.width,o.height,4,pixels.data(),o.width*4)) throw std::runtime_error("Capture failed");
}
void run(ExampleApp& app,const Options& o) {
  raisim::World world; RayraiWindow viewer(world,o.width,o.height,RayraiWindow::ThreadingMode::SingleThread);
  auto q=RenderQualitySettings::preset(RenderQualityPreset::High);
  q.highFidelityPbr=true; q.geometryRefraction=true; q.screenSpaceRefraction=true;
  q.geometryRefractionBackend=o.backend;
  q.geometryRefractionSamples=o.samples; q.geometryRefractionMaxBounces=o.bounces;
  q.geometryRefractionProgressive=!o.benchmark; q.geometryRefractionMaxSamples=65536;
  q.viewerMsaaSamples=o.msaa; q.temporalAaEnabled=false; q.fxaaEnabled=true;
  q.autoExposureEnabled=false; q.screenSpaceAoEnabled=false; q.reflectiveGround=false;
  q.shadowsEnabled=true; q.addViewerFillLights=false; q.mainLightAmbient=glm::vec3(.09f,.11f,.14f);
  q.mainLightDiffuse=glm::vec3(2.7f); q.mainLightDirection=glm::normalize(glm::vec3(-.4,.2,-1));
  q.bloomEnabled=false; q.depthOfFieldEnabled=false; q.fogDensity=0; q.heightFogEnabled=false;
  q.proceduralSkyBackgroundEnabled=false; q.pbrExposure=1; q.gamma=2.2f;
  viewer.setRenderQualitySettings(q); viewer.setLinearHdrRenderingEnabled(true); viewer.clearAdditionalLights();
  const auto box=[&](const std::string& name,glm::vec3 p,glm::vec3 size,glm::vec3 c,bool emissive=false) {
    auto v=viewer.addVisualBox(name,size.x,size.y,size.z,glm::vec4(1)); v->setPosition(p.x,p.y,p.z);
    auto m=Material::pbr(name,glm::vec4(c,1),0,.8f); m.unlit=emissive; m.disableFog=true;
    v->setMaterialOverride(m); v->setDetectable(true); return v;
  };
  box("floor",{0,0,-.15},{16,15,.2},{.12,.15,.2});
  box("backdrop",{0,3.7,2.3},{15,.15,5},{.7,.72,.76});
  for(int i=0;i<30;++i) box("stripe"+std::to_string(i),{-7.25f+.5f*i,3.59,2},
    {.18,.04,4},i%3==0?glm::vec3(.04,.22,.32):i%3==1?glm::vec3(.7,.2,.035):glm::vec3(.055,.065,.085));
  // Real offscreen emissive geometry: refraction and internal reflection rays
  // can hit these softboxes even though the primary camera cannot see them.
  box("softbox left",{-5,-6,4.8},{.15,2.5,2.2},{9.5,9.1,8.6},true)->setCastsShadows(false);
  box("softbox right",{5,-6,4},{.15,2.5,2},{4,6,9},true)->setCastsShadows(false);
  box("softbox ceiling",{0,1,6},{6,3,.1},{4.5,4.5,4.5},true)->setCastsShadows(false);
  auto clear=Material::glass("clear solid",1,glm::vec3(.96,.99,1),4,0,1.5f);
  clear.roughnessFactor=o.roughness;
  clear.dielectricPriority=20;
  auto water=Material::glass("water",1,glm::vec3(.62,.91,.94),1.5,0,1.333f);
  water.dielectricPriority=10;
  auto air=Material::glass("air bubble",1,glm::vec3(1),1,0,1); air.dielectricPriority=30;
  std::vector<std::pair<std::shared_ptr<Visuals>,glm::vec3>> moving;
  const auto mesh=[&](const std::string& name,const glass_example::Geometry& g,glm::vec3 p,Material material) {
    auto v=viewer.addVisualCustomMesh(name,g.meshes(),glm::vec4(1)); v->setPosition(p.x,p.y,p.z);
    v->setMaterialOverride(material); v->setDetectable(true); v->setCastsShadows(false);
    moving.emplace_back(v,p); return v;
  };
  const auto sphere=[&](const std::string& name,glm::vec3 p,float radius,Material material) {
    auto v=viewer.addVisualSphere(name,radius,glm::vec4(1)); v->setPosition(p.x,p.y,p.z);
    v->setMaterialOverride(material); v->setDetectable(true); v->setCastsShadows(false);
    moving.emplace_back(v,p); return v;
  };
  if(o.layers>0) {
    // Stress repeated medium transitions, not triangle count. Distinct nested
    // solids alternate glass/water IOR; inner priorities resolve their overlap.
    // At most 16 layers respect the transport's active-medium capacity.
    for(int i=0;i<o.layers;++i) {
      auto material=clear; material.ior=i%2?1.333f:1.5f;
      material.dielectricPriority=i;
      material.volumeAttenuationColor=i%2?glm::vec3(.8f,.94f,.99f):glm::vec3(.99f,.91f,.8f);
      material.volumeAttenuationDistance=8.f;
      const glm::vec3 extent(1.25f-.06f*i);
      mesh("nested layer "+std::to_string(i),glass_example::box(-extent,extent),{0,.4f,1.3f},material);
    }
  } else if(o.objects==0) {
    mesh("water glass",glass_example::cup(),{-3,0,0},clear);
    // Water deliberately overlaps the inner wall and bottom; glass priority
    // wins there, producing glass/water rather than two air interfaces.
    mesh("water",glass_example::lathe({{0,.06f},{.74f,.06f},{.74f,1.32f},{0,1.32f}}),{-3,0,0},water);
    sphere("bubble",{-2.9,-.12,.85},.18f,air);
    mesh("empty hollow cup",glass_example::cup(),{0,.7,0},clear);
    auto amber=clear; amber.volumeAttenuationColor={.94,.55,.16}; amber.volumeAttenuationDistance=.65f;
    auto torus=mesh("concave amber torus",glass_example::torus(),{3,.3,1.1},amber);
    torus->setOrientation(.70710678118, .70710678118,0,0);
    auto blue=clear; blue.volumeAttenuationColor={.25,.7,.96}; blue.volumeAttenuationDistance=1;
    blue.dielectricPriority=5;
    sphere("overlap blue",{-.55,-2, .85},.8,blue);
    auto pink=clear; pink.volumeAttenuationColor={.98,.48,.6}; pink.volumeAttenuationDistance=1;
    pink.ior=1.38f; pink.dielectricPriority=15;
    sphere("overlap pink",{.55,-1.8,.85},.8,pink);
  } else {
    const int columns=int(std::ceil(std::sqrt(o.objects)));
    auto torus=glass_example::torus(.35,.11,32,12);
    for(int i=0;i<o.objects;++i) {
      auto m=clear; m.dielectricPriority=i%3;
      mesh("glass"+std::to_string(i),torus,{(i%columns-(columns-1)*.5f),float(i/columns)-columns*.5f,.5f},m);
    }
  }
  if(o.opaqueOnly) for(const auto& item:moving) item.first->setColorPassVisible(false);
  viewer.updateObjectLists();
  const glm::vec3 eye=o.objects?glm::vec3(10,-17,17):glm::vec3(1.6f,-8.3f,3.22f);
  const glm::vec3 lookAt(0,.4f,.9f);
  if(o.benchmark || !o.output.empty() || !o.recording.empty()) {
    // Offscreen paths use a private camera for reproducible captures, recordings and timings.
    Camera camera; camera.ensureRenderTargets(o.width,o.height); camera.setSceneMsaaSamples(o.msaa);
    camera.position=eye; camera.front=glm::normalize(lookAt-eye);
    camera.up=camera.worldUp={0,0,1}; camera.zoom=42; camera.nearPlane=camera.zNear=.05;
    camera.farPlane=camera.zFar=100; camera.setGamma(q.gamma); camera.setBackgroundColorRgb255({15,20,30,255});
    RayraiWindow::RenderOverrides ov; ov.postProcess=true; ov.doShadows=true; ov.drawVisualizationObjects=true;
    ov.drawPointClouds=ov.drawCoordinateFrames=ov.drawWeatherEffects=false; ov.allowTemporalAa=false;
    ov.allowViewerMsaa=true; ov.allowViewerUpscale=false;
    const auto render=[&](int frame) {
      if(o.animate) for(const auto& item:moving) {
        const auto p=item.second+glm::vec3(.04f*std::sin(frame*.11f),0,0);
        item.first->setPosition(p.x,p.y,p.z);
      }
      camera.position=eye;
      if(!o.recording.empty()) {
        const auto offset=eye-lookAt;
        const float angle=.10f*std::sin(2.f*3.14159265359f*frame/o.frames);
        camera.position=lookAt+glm::vec3(std::cos(angle)*offset.x-std::sin(angle)*offset.y,
                                       std::sin(angle)*offset.x+std::cos(angle)*offset.y,offset.z);
      } else if(o.benchmark) camera.position.x+=.001f*std::sin(frame*.037f);
      camera.front=glm::normalize(lookAt-camera.position); viewer.renderWithExternalCamera(camera,ov);
    };
    for(int i=0;i<o.warmup;++i) render(i);
    glFinish(); const auto start=std::chrono::steady_clock::now(); double submitMs=0;
    for(int i=0;i<o.frames;++i) {
      const auto submitStart=std::chrono::steady_clock::now(); render(o.recording.empty()?i+o.warmup:i);
      submitMs+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-submitStart).count();
      glFinish();
      if(!o.recording.empty()) {
        char name[32]; std::snprintf(name,sizeof(name),"frame_%03d.png",i);
        auto capture=o; capture.output=(o.recording/name).string(); save(camera,capture);
        if((i+1)%25==0) std::cout<<"Nested glass: recorded "<<i+1<<'/'<<o.frames<<std::endl;
      }
    }
    const double totalMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    const double ms=totalMs/o.frames;
    if(!o.output.empty()) save(camera,o);
    const auto d=viewer.geometryRefractionDiagnostics();
    std::cout<<"backend="<<(d.backend==GeometryRefractionBackend::VulkanRayQuery?"vulkan":"portable")<<" status="<<d.backendStatus<<'\n';
    std::cout<<"GPU: "<<glGetString(GL_RENDERER)<<'\n'<<std::fixed<<std::setprecision(4)
      <<"completed_ms="<<ms<<" submit_ms="<<submitMs/o.frames<<" total_ms="<<totalMs<<" objects="<<o.objects<<" triangles="<<d.triangles<<" volumes="<<d.volumes
      <<" samples_per_frame="<<o.samples<<" accumulated_samples="<<d.accumulatedSamples
      <<" bounces="<<o.bounces<<" frames="<<o.frames<<" warmup="<<o.warmup
      <<" animate="<<o.animate<<" roughness="<<o.roughness<<'\n';
    const auto statistics=viewer.geometryRefractionSamplingStatistics();
    std::cout<<"sampling_pixels="<<statistics.pixels<<" mean_samples="<<statistics.meanSamples
      <<" min_samples="<<statistics.minSamples<<" max_samples="<<statistics.maxSamples
      <<" below_budget="<<statistics.belowBudget<<'\n';
    if(o.layers) std::cout<<"nested_layers="<<o.layers<<'\n';
    if(glGetError()!=GL_NO_ERROR) throw std::runtime_error("OpenGL error in nested glass example");
    return;
  }
  // Interactive mode drives the viewer's own camera, so this example gets the
  // standard rayrai navigation: WASD/Space to fly, drag to look, and a click on
  // an object to lock the orbit target onto it (click empty space to release).
  auto& camera=viewer.getCamera();
  camera.position=eye; camera.front=glm::normalize(lookAt-eye);
  camera.up=camera.worldUp={0,0,1};
  camera.yaw=glm::degrees(std::atan2(camera.front.y,camera.front.x));
  camera.pitch=glm::degrees(std::asin(std::clamp(camera.front.z,-1.f,1.f)));
  camera.target=lookAt; camera.zoom=42; camera.movementSpeed=3.f;
  camera.nearPlane=camera.zNear=.05f; camera.farPlane=camera.zFar=100;
  camera.setGamma(q.gamma); camera.setBackgroundColorRgb255({15,20,30,255});
  camera.setCameraFixedTarget(false); camera.setCameraFixedDistance(false);
  while(!app.quit) {
    app.processEvents(); app.beginFrame();
    app.renderViewer(viewer);
    ImGui::Begin("Geometry refraction");
    if(o.layers) ImGui::Text("%d nested glass/water layers",o.layers);
    else ImGui::TextUnformatted("Water + air bubble / hollow cup / concave torus\nFront: overlapping glass solids");
    bool changed=ImGui::Checkbox("Geometry tracing",&q.geometryRefraction);
    int backend=int(q.geometryRefractionBackend);
    if(ImGui::Combo("Tracing backend",&backend,"Automatic\0Portable OpenGL\0Vulkan ray query (required)\0")) {
      q.geometryRefractionBackend=GeometryRefractionBackend(backend);changed=true;
    }
    changed|=ImGui::SliderInt("Paths / frame",&q.geometryRefractionSamples,1,32);
    changed|=ImGui::SliderInt("Max bounces",&q.geometryRefractionMaxBounces,1,128);
    changed|=ImGui::Checkbox("Low-discrepancy samples (experimental)",&q.geometryRefractionLowDiscrepancy);
    changed|=ImGui::Checkbox("Adaptive samples (experimental)",&q.geometryRefractionAdaptive);
    if(changed) viewer.setRenderQualitySettings(q);
    const auto d=viewer.geometryRefractionDiagnostics();
    ImGui::TextWrapped("%s",d.backendStatus.c_str());
    ImGui::Text("%s: %d samples | %zu triangles",q.geometryRefractionAdaptive?"Scheduled budget":"Accumulated",
      d.accumulatedSamples,d.triangles);
    if(const auto* focus=viewer.getTargetVisual()) ImGui::Text("Orbiting: %s",focus->getName().c_str());
    else ImGui::TextUnformatted("Free fly: WASD + Space, drag to look, click an object to orbit");
    ImGui::TextUnformatted("Pause the camera to converge. No caustic illumination of the floor.");
    ImGui::End(); app.endFrame();
  }
}
}
int main(int argc,char** argv) {
  ExampleApp app;
  try {
    const auto o=parse(argc,argv);
    if(!app.init("Rayrai nested glass",o.width,o.height,!o.benchmark&&o.output.empty()&&o.recording.empty())) return 1;
    SDL_GL_SetSwapInterval(0); run(app,o); app.shutdown(); return 0;
  } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; if(app.context) app.shutdown(); return 1; }
}
