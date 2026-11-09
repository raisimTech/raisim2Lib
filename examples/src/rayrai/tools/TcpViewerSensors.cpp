// Copyright (c) 2026 Raion Robotics Inc.
// All rights reserved.

#include "TcpViewerSensors.hpp"
#include "TcpViewerSensorLimits.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <utility>

#include <SDL.h>
#include <glbinding/gl/gl.h>
#include <glm/gtc/quaternion.hpp>

#include "rayrai/Camera.hpp"
#include "rayrai/GlStateGuard.hpp"
#include "rayrai/RayraiWindow.hpp"
#include "rayrai/TextureBindingCache.hpp"
#include "raisim/sensors/DepthSensor.hpp"
#include "raisim/sensors/RGBSensor.hpp"

namespace raisin::tcp_viewer
{
namespace
{

std::string sensorKey(const SensorInfo& info) {
  return std::to_string(info.parentTag) + ":" + std::to_string(static_cast<int>(info.type)) +
         ":" + info.name;
}

void configureCamera(Camera& camera, const PendingSensorUpdate& update) {
  const auto& info = update.info;
  camera.ensureRenderTargets(info.width, info.height);
  camera.aspect = static_cast<float>(info.width) / static_cast<float>(info.height);
  camera.nearPlane = camera.zNear =
    static_cast<float>(sensorRenderNearClip(info.clipNear, info.clipFar));
  camera.farPlane = camera.zFar = static_cast<float>(info.clipFar);
  camera.setImageWindowOffset(glm::vec2(static_cast<float>(info.xOffset) / static_cast<float>(info.width),
                                        static_cast<float>(info.yOffset) / static_cast<float>(info.height)));
  camera.setLensModel(info.lens, info.hFov);
  const double rasterHFov = info.lens.isFisheye() ? std::min(info.hFov, 3.124139361) : info.hFov;
  camera.zoom = glm::degrees(static_cast<float>(
    2.0 * std::atan(std::tan(0.5 * rasterHFov) / std::max(1.0e-6f, camera.aspect))));
  camera.setProjectionMode(Camera::ProjectionMode::PERSPECTIVE);

  glm::quat orientation(update.orientation.w, update.orientation.x,
                        update.orientation.y, update.orientation.z);
  const float norm = glm::length(orientation);
  orientation = std::isfinite(norm) && norm > 1.0e-6f
                  ? orientation / norm
                  : glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
  const glm::mat3 rotation = glm::mat3_cast(orientation);
  camera.position = update.position;
  camera.front = glm::normalize(rotation[0]);
  camera.up = glm::normalize(rotation[2]);
  camera.right = glm::normalize(glm::cross(camera.front, camera.up));
  camera.worldUp = camera.up;
  camera.target = camera.position + camera.front;
}

/** RaiSim camera properties matching a remote sensor, for reading back its image. */
template <typename Properties>
Properties sensorCameraProperties(const SensorInfo& info) {
  Properties properties;
  properties.name = info.name;
  properties.full_name = info.name;
  properties.width = info.width;
  properties.height = info.height;
  properties.clipNear = info.clipNear;
  properties.clipFar = info.clipFar;
  properties.hFOV = info.hFov;
  properties.lens = info.lens;
  properties.xOffset = info.xOffset;
  properties.yOffset = info.yOffset;
  return properties;
}

/**
 * Returns a clean readback camera; the server applies the sensor's noise model to the received
 * pixels. Constructing a DepthCamera computes a ray direction per pixel, so one camera is kept
 * per stream and rebuilt when the requested resolution changes.
 */
template <typename SensorType, typename Properties>
SensorType& readbackSensor(std::unique_ptr<SensorType>& sensor, const SensorInfo& info) {
  if (!sensor || sensor->getProperties().width != info.width ||
      sensor->getProperties().height != info.height ||
      sensor->getProperties().xOffset != info.xOffset ||
      sensor->getProperties().yOffset != info.yOffset) {
    raisim::Vec<3> position;
    position.setZero();
    raisim::Mat<3, 3> rotation;
    rotation.setIdentity();
    sensor = std::make_unique<SensorType>(
      sensorCameraProperties<Properties>(info), nullptr, position, rotation);
  }
  return *sensor;
}

struct PreviewTexture {
  unsigned int texture = 0;
  int width = 0, height = 0;
};

// Upload tightly packed bytes without inheriting the application's PBO or row
// offsets. Restoring the texture binding also keeps the renderer's binding cache valid.
class ScopedPreviewUpload {
 public:
  ScopedPreviewUpload() {
    gl::glGetIntegerv(gl::GL_TEXTURE_BINDING_2D, &texture_);
    for (size_t i = 0; i < kFields.size(); ++i) {
      gl::glGetIntegerv(kFields[i], &previous_[i]);
      if (previous_[i] != kValues[i]) gl::glPixelStorei(kFields[i], kValues[i]);
    }
  }
  ~ScopedPreviewUpload() {
    gl::glBindTexture(gl::GL_TEXTURE_2D, static_cast<gl::GLuint>(texture_));
    for (size_t i = 0; i < kFields.size(); ++i)
      if (previous_[i] != kValues[i]) gl::glPixelStorei(kFields[i], previous_[i]);
  }
  ScopedPreviewUpload(const ScopedPreviewUpload&) = delete;
  ScopedPreviewUpload& operator=(const ScopedPreviewUpload&) = delete;

 private:
  ScopedPixelUnpackBufferZero noUnpackBuffer_;
  static constexpr std::array<gl::GLenum, 4> kFields = {gl::GL_UNPACK_ALIGNMENT,
      gl::GL_UNPACK_ROW_LENGTH, gl::GL_UNPACK_SKIP_ROWS, gl::GL_UNPACK_SKIP_PIXELS};
  static constexpr std::array<gl::GLint, 4> kValues = {1, 0, 0, 0};
  std::array<gl::GLint, 4> previous_{};
  gl::GLint texture_ = 0;
};

void uploadPreview(PreviewTexture& target, int width, int height,
                   const std::vector<unsigned char>& rgba) {
  const ScopedPreviewUpload upload;
  const bool created = target.texture == 0;
  if (created) gl::glGenTextures(1, &target.texture);
  gl::glBindTexture(gl::GL_TEXTURE_2D, target.texture);
  if (created) {
    gl::glTexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_MIN_FILTER, gl::GL_LINEAR);
    gl::glTexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_MAG_FILTER, gl::GL_LINEAR);
    gl::glTexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_WRAP_S, gl::GL_CLAMP_TO_EDGE);
    gl::glTexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_WRAP_T, gl::GL_CLAMP_TO_EDGE);
  }
  if (created || target.width != width || target.height != height) {
    gl::glTexImage2D(gl::GL_TEXTURE_2D, 0, gl::GL_RGBA8, width, height, 0,
                     gl::GL_RGBA, gl::GL_UNSIGNED_BYTE, rgba.data());
    target.width = width;
    target.height = height;
  } else {
    gl::glTexSubImage2D(gl::GL_TEXTURE_2D, 0, 0, 0, width, height,
                        gl::GL_RGBA, gl::GL_UNSIGNED_BYTE, rgba.data());
  }
}

void uploadDepthPreview(PreviewTexture& target, int width, int height,
                        const std::vector<float>& depth,
                        const detail::DepthPreviewShading& shading,
                        std::vector<unsigned char>& rgba,
                        float& minimumDepth, float& maximumDepth) {
  detail::makeDepthPreviewRgba(depth, shading, rgba, minimumDepth, maximumDepth);
  uploadPreview(target, width, height, rgba);
}

void uploadSegmentationPreview(PreviewTexture& target, const PendingSensorUpdate& update,
                               std::vector<unsigned char>& rgba) {
  const auto& ids = update.labels.empty() ? update.objects : update.labels;
  raisim::SegmentationCamera::colorize(ids, rgba, {0, 0, 0, 255});
  uploadPreview(target, update.info.width, update.info.height, rgba);
}

} // namespace

namespace detail
{

namespace {

// The caller checks the image length. Confirm eight missing returns before
// taking the bulk path; ordinary images use the shade/range loops.
template <size_t BlockSize>
#if defined(_MSC_VER)
__declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
size_t writeInvalidDepthPreviewPrefix(const float* depth, size_t count,
                                      unsigned char* rgba, uint32_t opaque) {
  static_assert(BlockSize >= 8u);
  std::array<uint32_t, 8> first;
  std::memcpy(first.data(), depth, sizeof(first));
  bool firstValid = false;
  for (uint32_t word : first) firstValid |= word - 1u < UINT32_C(0x7f7fffff);
  if (firstValid) return 0;
  std::array<uint32_t, BlockSize> block;
  block.fill(opaque);
  // The first eight samples were already classified above.
  std::memcpy(rgba, block.data(), sizeof(first));
  size_t i = first.size();
  while (count - i >= block.size()) {
    std::array<uint32_t, BlockSize> words;
    std::memcpy(words.data(), depth + i, sizeof(words));
    // The first word is known to be missing. Repeated words need only an
    // equality check; otherwise the eight-sample loop below classifies the
    // rest of the prefix without assuming a common representation.
    bool stop = false;
    for (uint32_t word : words) {
      if constexpr (BlockSize > 8u) stop |= word != first[0];
      else stop |= word - 1u < UINT32_C(0x7f7fffff);
    }
    if (stop) break;
    std::memcpy(rgba + i * 4u, block.data(), sizeof(block));
    i += block.size();
  }
  if constexpr (BlockSize > 8u) {
    while (count - i >= 8u) {
      std::array<uint32_t, 8> words;
      std::memcpy(words.data(), depth + i, sizeof(words));
      bool anyValid = false;
      for (uint32_t word : words) anyValid |= word - 1u < UINT32_C(0x7f7fffff);
      if (anyValid) break;
      std::memcpy(rgba + i * 4u, block.data(), sizeof(words));
      i += words.size();
    }
  }
  while (i < count) {
    uint32_t word;
    std::memcpy(&word, depth + i, sizeof(word));
    if (word - 1u < UINT32_C(0x7f7fffff)) break;
    std::memcpy(rgba + i * 4u, &opaque, sizeof(opaque));
    ++i;
  }
  return i;
}

#if defined(_MSC_VER)
__declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
size_t writeVariedDepthPreviewBlocks(const float* depth, size_t count,
    const DepthPreviewShading& shading, unsigned char* rgba, uint32_t opaque,
    float& minimum, float& maximum) {
  const auto lookup = shading.batchLookup();
  uint32_t lower, upper;
  const float first = shading.interval(0).second, last = shading.interval(255).first;
  std::memcpy(&lower, &first, sizeof(lower));
  std::memcpy(&upper, &last, sizeof(upper));
  const uint32_t span = upper - lower;
  std::array<float, 4> blockMinimum;
  blockMinimum.fill(std::numeric_limits<float>::infinity());
  std::array<float, 4> blockMaximum{};
  size_t i = 0;
  while (count - i >= 4u) {
    std::array<uint32_t, 4> bits;
    std::memcpy(bits.data(), depth + i, sizeof(bits));
    bool allValid = true;
#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(disable) unroll(disable)
#endif
    for (size_t lane = 0; lane < 4u; ++lane) allValid &= bits[lane] - lower < span;
    // Return flat runs to the shade cache in the ordinary preview loop.
    if (!allValid || bits[0] == bits[1]) break;
    std::array<float, 4> values;
    std::memcpy(values.data(), bits.data(), sizeof(values));
    const auto shades = lookup.shadesWithinRange(values);
    // Different depths can share one shade, including clipped returns. Let the
    // ordinary loop cache that interval instead of repeating four lookups.
    bool sameShade = true;
#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(disable) unroll(disable)
#endif
    for (size_t lane = 0; lane < 4u; ++lane) sameShade &= shades[lane] == shades[0];
    if (sameShade) break;
    std::array<uint32_t, 4> pixels;
#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(disable) unroll(disable)
#endif
    for (size_t lane = 0; lane < 4u; ++lane) {
      const uint32_t pair = shades[lane] | (shades[lane] << 8u);
      pixels[lane] = pair | (pair << 16u) | opaque;
      blockMinimum[lane] = std::min(blockMinimum[lane], values[lane]);
      blockMaximum[lane] = std::max(blockMaximum[lane], values[lane]);
    }
    std::memcpy(rgba + i * 4u, pixels.data(), sizeof(pixels));
    i += 4u;
  }
  // Combine lanes locally before updating the caller's range once per run.
  float runMinimum = blockMinimum[0], runMaximum = blockMaximum[0];
  for (size_t lane = 1; lane < 4u; ++lane) {
    runMinimum = std::min(runMinimum, blockMinimum[lane]);
    runMaximum = std::max(runMaximum, blockMaximum[lane]);
  }
  minimum = std::min(minimum, runMinimum);
  maximum = std::max(maximum, runMaximum);
  return i;
}

#if defined(_MSC_VER)
__declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
void writeCachedDepthPreviewBlocks(const float* samples, size_t count,
    const DepthPreviewShading& shading, unsigned char* output, uint32_t opaque,
    float& imageMinimum, float& imageMaximum) {
  float minimum = std::numeric_limits<float>::infinity();
  float maximum = 0.0f;
  std::array<float, 4> blockMinimum{minimum, minimum, minimum, minimum};
  std::array<float, 4> blockMaximum{};
  uint32_t lower = 1, span = 0;
  unsigned char previousShade = 0;
  size_t i = 0;
  while (i < count) {
    if (count - i >= 4u) {
      std::array<uint32_t, 4> bits;
      std::memcpy(bits.data(), samples + i, sizeof(bits));
      // Positive IEEE floats have ordered bits. This exact half-open interval excludes
      // zero, infinities, NaNs and negative values, including when shade 0 or 255 is cached.
      const bool sameShade = int(bits[0] - lower < span) & int(bits[1] - lower < span) &
          int(bits[2] - lower < span) & int(bits[3] - lower < span);
      if (sameShade) {
        // Reuse the classified words instead of loading the same depths again.
        std::array<float, 4> values;
        std::memcpy(values.data(), bits.data(), sizeof(values));
        // Reduce each lane across blocks, then combine lanes once at the end.
        // Keep the loop in lane order so Clang emits packed min/max operations.
#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(disable) unroll(disable)
#endif
        for (size_t lane = 0; lane < 4u; ++lane) {
          blockMinimum[lane] = std::min(blockMinimum[lane], values[lane]);
          blockMaximum[lane] = std::max(blockMaximum[lane], values[lane]);
        }
        const uint32_t pixel = UINT32_C(0x01010101) * previousShade | opaque;
        const std::array<uint32_t, 4> block{pixel, pixel, pixel, pixel};
        std::memcpy(output + i * 4u, block.data(), sizeof(block));
        i += 4u;
        continue;
      }
      const bool anyValid = int(bits[0] - 1u < UINT32_C(0x7f7fffff)) |
          int(bits[1] - 1u < UINT32_C(0x7f7fffff)) |
          int(bits[2] - 1u < UINT32_C(0x7f7fffff)) |
          int(bits[3] - 1u < UINT32_C(0x7f7fffff));
      if (!anyValid) {
        const std::array<uint32_t, 4> block{opaque, opaque, opaque, opaque};
        std::memcpy(output + i * 4u, block.data(), sizeof(block));
        i += 4u;
        continue;
      }
    }
    const size_t end = i + std::min(size_t(4), count - i);
    do {
      const float value = samples[i];
      unsigned char shade = 0;
      uint32_t bits;
      std::memcpy(&bits, &value, sizeof(bits));
      // Exactly the finite, positive floats, including subnormals.
      if (bits - 1u < UINT32_C(0x7f7fffff)) {
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
        if (bits - lower < span) {
          shade = previousShade;
        } else {
          shade = previousShade = shading.shade(value);
          const auto interval = shading.interval(shade);
          uint32_t upper;
          std::memcpy(&lower, &interval.first, sizeof(lower));
          std::memcpy(&upper, &interval.second, sizeof(upper));
          if (!(interval.first > 0.f)) lower = 1;
          span = upper - lower;
        }
      }
      const uint32_t pixel = UINT32_C(0x01010101) * shade | opaque;
      std::memcpy(output + i * 4u, &pixel, sizeof(pixel));
      ++i;
    } while (i < end);
  }
  for (size_t lane = 0; lane < 4u; ++lane) {
    minimum = std::min(minimum, blockMinimum[lane]);
    maximum = std::max(maximum, blockMaximum[lane]);
  }
  imageMinimum = std::min(imageMinimum, minimum);
  imageMaximum = std::max(imageMaximum, maximum);
}

// Long leading runs need one shade lookup and fewer loop branches per pixel.
#if defined(_MSC_VER)
__declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
size_t writeUniformDepthPreviewPrefix(const float* samples, size_t count,
    const DepthPreviewShading& shading, unsigned char* output, uint32_t opaque,
    float& imageMinimum, float& imageMaximum) {
  uint32_t first;
  std::memcpy(&first, samples, sizeof(first));
  if (first - 1u >= UINT32_C(0x7f7fffff)) return 0;
  const unsigned char shade = shading.shade(samples[0]);
  const auto interval = shading.interval(shade);
  uint32_t lower, upper;
  std::memcpy(&lower, &interval.first, sizeof(lower));
  std::memcpy(&upper, &interval.second, sizeof(upper));
  if (!(interval.first > 0.f)) lower = 1;
  const uint32_t span = upper - lower;
  // The caller supplies at least 4096 samples. Confirm a long run before
  // adding the wide loop's setup to short surfaces and varying images.
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
  for (size_t offset = 0; offset < 1024u; offset += 4u) {
    std::array<uint32_t, 4> bits;
    std::memcpy(bits.data(), samples + offset, sizeof(bits));
    if (!(int(bits[0] - lower < span) & int(bits[1] - lower < span) &
          int(bits[2] - lower < span) & int(bits[3] - lower < span))) return 0;
  }
  const uint32_t pixel = UINT32_C(0x01010101) * shade | opaque;
  std::array<uint32_t, 16> pixels;
  pixels.fill(pixel);
  std::array<float, 4> minimum;
  minimum.fill(std::numeric_limits<float>::infinity());
  std::array<float, 4> maximum{};
  size_t i = 0;
  while (count - i >= 16u) {
    std::array<uint32_t, 16> bits;
    std::memcpy(bits.data(), samples + i, sizeof(bits));
    bool sameShade = true;
#if defined(__clang__)
#pragma clang loop unroll(full)
#endif
    for (size_t lane = 0; lane < 16u; ++lane) sameShade &= bits[lane] - lower < span;
    if (!sameShade) break;
    std::array<float, 16> values;
    std::memcpy(values.data(), bits.data(), sizeof(values));
    // Four lanes per update keep the reductions in registers on 128-bit targets.
#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(disable) unroll(disable)
#endif
    for (size_t lane = 0; lane < 4u; ++lane) {
      minimum[lane] = std::min(minimum[lane], values[lane]);
      maximum[lane] = std::max(maximum[lane], values[lane]);
      minimum[lane] = std::min(minimum[lane], values[lane + 4u]);
      maximum[lane] = std::max(maximum[lane], values[lane + 4u]);
      minimum[lane] = std::min(minimum[lane], values[lane + 8u]);
      maximum[lane] = std::max(maximum[lane], values[lane + 8u]);
      minimum[lane] = std::min(minimum[lane], values[lane + 12u]);
      maximum[lane] = std::max(maximum[lane], values[lane + 12u]);
    }
    std::memcpy(output + i * 4u, pixels.data(), sizeof(pixels));
    i += 16u;
  }
  for (size_t lane = 0; lane < 4u; ++lane) {
    imageMinimum = std::min(imageMinimum, minimum[lane]);
    imageMaximum = std::max(imageMaximum, maximum[lane]);
  }
  return i;
}

// Sampling chooses an execution path only; both paths produce every exact shade.
// Keep flat, clipped and mostly missing images on the cached loop.
// Fixed bounds preserve the compiler's short-window loop optimizations.
template <size_t ProbeLimit>
#if defined(_MSC_VER)
__declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
bool probeVariedDepthPreview(const float* depth, size_t count,
                             const DepthPreviewShading& shading) {
  std::array<uint32_t, 4> firstBits;
  std::memcpy(firstBits.data(), depth, sizeof(firstBits));
  if (firstBits[0] - 1u < UINT32_C(0x7f7fffff)) {
    const auto interval = shading.interval(shading.shade(depth[0]));
    uint32_t firstLower, firstUpper;
    std::memcpy(&firstLower, &interval.first, sizeof(firstLower));
    std::memcpy(&firstUpper, &interval.second, sizeof(firstUpper));
    if (!(interval.first > 0.f)) firstLower = 1;
    const uint32_t firstSpan = firstUpper - firstLower;
    if (int(firstBits[1] - firstLower < firstSpan) & int(firstBits[2] - firstLower < firstSpan) &
        int(firstBits[3] - firstLower < firstSpan)) return false;
  }
  uint32_t lower, upper;
  const float first = shading.interval(0).second, last = shading.interval(255).first;
  std::memcpy(&lower, &first, sizeof(lower));
  std::memcpy(&upper, &last, sizeof(upper));
  const uint32_t span = upper - lower;
  size_t varying = 0;
  for (size_t probe = 0; probe < 8u; ++probe) {
    size_t offset = (count - 4u) / 7u * probe + (count - 4u) % 7u * probe / 7u;
    // Stagger the grid so periodic invalid blocks do not all land between probes.
    const size_t stagger = probe * (probe + 1u) / 2u;
    offset = stagger <= count - 4u - offset ? offset + stagger : offset - stagger;
    // Look within a short window when a probe lands on missing returns. The
    // staggered grid alone can still alias alternating missing/varied runs.
    const size_t probeCount = std::min(ProbeLimit, count - offset);
    for (size_t step = 0; step + 4u <= probeCount; step += 4u) {
      std::array<uint32_t, 4> bits;
      std::memcpy(bits.data(), depth + offset + step, sizeof(bits));
      const bool interior = int(bits[0] - lower < span) & int(bits[1] - lower < span) &
          int(bits[2] - lower < span) & int(bits[3] - lower < span);
      if (!interior) {
        bool clipped = false;
        for (uint32_t word : bits)
          clipped |= int(word - 1u < UINT32_C(0x7f7fffff)) & int(word - lower >= span);
        if (clipped) break;
        continue;
      }
      const auto interval = shading.interval(shading.shade(depth[offset + step]));
      uint32_t shadeLower, shadeUpper;
      std::memcpy(&shadeLower, &interval.first, sizeof(shadeLower));
      std::memcpy(&shadeUpper, &interval.second, sizeof(shadeUpper));
      const uint32_t shadeSpan = shadeUpper - shadeLower;
      const bool sameShade = int(bits[1] - shadeLower < shadeSpan) &
          int(bits[2] - shadeLower < shadeSpan) & int(bits[3] - shadeLower < shadeSpan);
      varying += !sameShade;
      break;
    }
  }
  return varying >= 6u;
}

bool useVariedDepthPreview(const float* depth, size_t count,
                           const DepthPreviewShading& shading) {
  // Extend through longer gaps only after a dense prefix, and keep small
  // previews on the short windows. This avoids scanning farther for sparse
  // isolated returns; eight full windows inspect at most 512 samples.
  if (count >= 4096u) {
    uint32_t lower, upper;
    const float first = shading.interval(0).second, last = shading.interval(255).first;
    std::memcpy(&lower, &first, sizeof(lower));
    std::memcpy(&upper, &last, sizeof(upper));
    const uint32_t span = upper - lower;
    std::array<uint32_t, 8> prefix;
    std::memcpy(prefix.data(), depth, sizeof(prefix));
    bool densePrefix = true;
    for (uint32_t word : prefix) densePrefix &= word - lower < span;
    if (densePrefix) return probeVariedDepthPreview<64>(depth, count, shading);
  }
  return probeVariedDepthPreview<16>(depth, count, shading);
}

} // namespace

DepthPreviewShading::DepthPreviewShading(float clipMinimum, float clipMaximum)
    : clipMinimum_(clipMinimum), clipMaximum_(clipMaximum) {
  scaleMinimum_ = std::isfinite(clipMinimum) ? std::max(clipMinimum, 1.0e-6f) : 1.0e-6f;
  scaleMaximum_ = std::isfinite(clipMaximum) && clipMaximum > scaleMinimum_
    ? clipMaximum
    : scaleMinimum_ + 1.0e-6f;
  if (!(scaleMaximum_ > scaleMinimum_)) {
    // The fallback increment can round away at large depths. Use an adjacent
    // finite float pair even at the largest representable clipping plane.
    scaleMaximum_ = std::nextafter(scaleMinimum_, std::numeric_limits<float>::infinity());
    if (!std::isfinite(scaleMaximum_)) {
      scaleMaximum_ = scaleMinimum_;
      scaleMinimum_ = std::nextafter(scaleMaximum_, 0.f);
    }
  }
  const float scaleMinimum = scaleMinimum_;
  const float scaleMaximum = scaleMaximum_;
  const bool wideRange = double(scaleMaximum) / scaleMinimum > std::numeric_limits<float>::max();
  const float inverseLogRange = wideRange ? 0.f : 1.0f / std::log(scaleMaximum / scaleMinimum);
  const double inverseWideLogRange = wideRange ?
      1. / std::log(double(scaleMaximum) / scaleMinimum) : 0.;
  const auto formulaShade = [&](float value) {
    if (wideRange) {
      const double normalized = std::log(double(value) / scaleMinimum) * inverseWideLogRange;
      return static_cast<unsigned char>(std::lround(255. * normalized));
    }
    // Float rounding defines the exact shade boundaries for ordinary ranges.
    const float normalized = std::log(value / scaleMinimum) * inverseLogRange;
    return static_cast<unsigned char>(std::lround(255.0f * normalized));
  };
  // Positive floats are ordered like their bit patterns, so bisect over those for the first
  // depth of each shade. Every shade starts at or after the previous one.
  const auto bits = [](float value) {
    uint32_t result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
  };
  const auto fromBits = [](uint32_t value) {
    float result = 0.0f;
    std::memcpy(&result, &value, sizeof(result));
    return result;
  };
  const unsigned char maximumShade = formulaShade(scaleMaximum);
  bounds_[0] = -std::numeric_limits<float>::infinity();
  uint32_t low = bits(scaleMinimum);
  for (int shade = 1; shade <= 255; ++shade) {
    if (maximumShade < shade) {
      bounds_[static_cast<size_t>(shade)] = std::numeric_limits<float>::infinity();
      continue;
    }
    uint32_t high = bits(scaleMaximum);
    while (low < high) {
      const uint32_t middle = low + (high - low) / 2u;
      if (formulaShade(fromBits(middle)) >= shade) {
        high = middle;
      } else {
        low = middle + 1u;
      }
    }
    bounds_[static_cast<size_t>(shade)] = fromBits(low);
  }
  bounds_[256] = std::numeric_limits<float>::infinity();

  // A bucket narrower than the smallest boundary gap crosses at most one
  // boundary. Repeated boundaries require one float per bucket instead.
  uint32_t minimumGap = std::numeric_limits<uint32_t>::max();
  for (size_t i = 2; i < 256u; ++i) {
    if (std::isfinite(bounds_[i]))
      minimumGap = std::min(minimumGap, bits(bounds_[i]) - bits(bounds_[i - 1u]));
  }
  uint32_t shift = 16u;
  while (shift > 0u && (UINT32_C(1) << shift) > minimumGap) --shift;
  // Align the table origin to at least 32 float words. Its five zero low bits
  // can carry the shift without a second per-pixel load or a larger object.
  const uint32_t alignment = UINT32_C(1) << std::max(shift, 5u);
  const uint32_t origin = bits(scaleMinimum) & ~(alignment - 1u);
  bucketLayout_ = origin | shift;
  const uint32_t firstBucket = origin >> shift;
  const uint32_t lastBucket = bits(scaleMaximum) >> shift;
  bucketEntries_.resize(lastBucket - firstBucket + 1u);
  const auto firstBound = bounds_.begin() + 1;
  const uint32_t width = UINT32_C(1) << shift;
  for (uint32_t bucket = firstBucket; bucket <= lastBucket; ++bucket) {
    const float edge = std::max(fromBits(bucket << shift), scaleMinimum);
    const uint32_t shade = static_cast<uint32_t>(
      std::upper_bound(firstBound, bounds_.begin() + 256, edge) - firstBound);
    const uint32_t boundaryOffset = std::min(bits(bounds_[shade + 1u]) - (bucket << shift), width);
    // Cancel the bucket's position, then add a bias whose carry changes the
    // shade at the exact boundary. A boundary beyond this bucket adds no bias.
    bucketEntries_[bucket - firstBucket] = (shade << shift) - ((bucket - firstBucket) << shift) +
        width - boundaryOffset;
  }
}

unsigned char DepthPreviewShading::shade(float depth) const {
  return shadeWithinRange(std::clamp(depth, scaleMinimum_, scaleMaximum_));
}

void makeDepthPreviewRgba(const std::vector<float>& depth, const DepthPreviewShading& shading,
  std::vector<unsigned char>& rgba, float& minimumDepth, float& maximumDepth) {
  size_t count = depth.size();
  rgba.resize(count * 4u);
  // Byte writes can alias the vector objects and the output references. Keep pointers and
  // range reductions local so they stay in registers until the whole image is complete.
  const float* samples = depth.data();
  unsigned char* output = rgba.data();
  // Repeating a byte fills all four channels; OR in alpha using its native byte position.
  // memcpy keeps both packed writes and unaligned destinations portable.
  const std::array<unsigned char, 4> opaqueBytes{0, 0, 0, 255};
  uint32_t opaque;
  std::memcpy(&opaque, opaqueBytes.data(), sizeof(opaque));
  if (count >= 32u) {
    uint32_t first;
    std::memcpy(&first, samples, sizeof(first));
    if (first - 1u >= UINT32_C(0x7f7fffff)) {
      // Restrict wider equality checks to the initial prefix of large images.
      // Later gaps retain the smaller classifier for short missing runs.
      const size_t skipped = count >= 4096u
          ? writeInvalidDepthPreviewPrefix<64>(samples, count, output, opaque)
          : writeInvalidDepthPreviewPrefix<8>(samples, count, output, opaque);
      samples += skipped;
      output += skipped * 4u;
      count -= skipped;
    }
  }
  float minimum = std::numeric_limits<float>::infinity();
  float maximum = 0.0f;
  // Choose the converter before consuming the prefix so its removal does not
  // change the execution path for the rest of a mostly uniform image.
  const bool varied = count >= 256u && useVariedDepthPreview(samples, count, shading);
  if (!varied && count >= 4096u) {
    // A missing prefix can leave every four-pixel block crossing a vector boundary.
    // Peel at most three pixels through the ordinary converter before batching.
    const size_t leading = (4u - ((reinterpret_cast<uintptr_t>(output) >> 2u) & 3u)) & 3u;
    if (leading != 0u) {
      writeCachedDepthPreviewBlocks(samples, leading, shading, output, opaque, minimum, maximum);
      samples += leading;
      output += leading * 4u;
      count -= leading;
    }
    const size_t written = count >= 4096u ? writeUniformDepthPreviewPrefix(samples, count, shading,
        output, opaque, minimum, maximum) : 0u;
    samples += written;
    output += written * 4u;
    count -= written;
  }
  if (varied) {
    while (count >= 64u) {
      const size_t written = writeVariedDepthPreviewBlocks(samples, count, shading,
          output, opaque, minimum, maximum);
      samples += written;
      output += written * 4u;
      count -= written;
      if (count == 0u) break;
      if (count >= 32u) {
        std::array<uint32_t, 2> first;
        std::memcpy(first.data(), samples, sizeof(first));
        if (int(first[0] - 1u >= UINT32_C(0x7f7fffff)) &
            int(first[1] - 1u >= UINT32_C(0x7f7fffff))) {
          const size_t skipped = writeInvalidDepthPreviewPrefix<8>(samples, count, output, opaque);
          if (skipped != 0u) {
            samples += skipped;
            output += skipped * 4u;
            count -= skipped;
            continue;
          }
        }
      }
      // Handle a boundary or missing return with the cached loop. Amortize its
      // setup over 64 samples when the next block already shares one shade.
      size_t cachedCount = std::min(count, size_t(4));
      if (count >= 64u) {
        std::array<uint32_t, 4> bits;
        std::memcpy(bits.data(), samples, sizeof(bits));
        const bool anyValid = int(bits[0] - 1u < UINT32_C(0x7f7fffff)) |
            int(bits[1] - 1u < UINT32_C(0x7f7fffff)) |
            int(bits[2] - 1u < UINT32_C(0x7f7fffff)) |
            int(bits[3] - 1u < UINT32_C(0x7f7fffff));
        if (!anyValid) cachedCount = 64u;
        else if (bits[0] - 1u < UINT32_C(0x7f7fffff)) {
          const auto interval = shading.interval(shading.shade(samples[0]));
          uint32_t lower, upper;
          std::memcpy(&lower, &interval.first, sizeof(lower));
          std::memcpy(&upper, &interval.second, sizeof(upper));
          if (!(interval.first > 0.f)) lower = 1;
          const uint32_t span = upper - lower;
          if (int(bits[1] - lower < span) & int(bits[2] - lower < span) & int(bits[3] - lower < span))
            cachedCount = 64u;
        }
      }
      writeCachedDepthPreviewBlocks(samples, cachedCount, shading, output, opaque, minimum, maximum);
      samples += cachedCount;
      output += cachedCount * 4u;
      count -= cachedCount;
    }
  }
  if (count != 0u)
    writeCachedDepthPreviewBlocks(samples, count, shading, output, opaque, minimum, maximum);
  minimumDepth = std::isfinite(minimum) ? minimum : 0.0f;
  maximumDepth = maximum;
}

std::vector<unsigned char> makeDepthPreviewRgba(const std::vector<float>& depth,
  float clipMinimum, float clipMaximum, float& minimumDepth, float& maximumDepth) {
  std::vector<unsigned char> rgba;
  makeDepthPreviewRgba(depth, DepthPreviewShading(clipMinimum, clipMaximum), rgba, minimumDepth,
    maximumDepth);
  return rgba;
}

} // namespace detail

struct SensorRenderer::CachedSensor {
  std::unique_ptr<Camera> camera;
  std::unique_ptr<raisim::RGBCamera> rgbSensor;
  std::unique_ptr<raisim::DepthCamera> depthSensor;
  std::optional<detail::DepthPreviewShading> depthShading;
  std::vector<unsigned char> previewRgba;
  SensorPreviewInfo preview;
  PreviewTexture previewTexture;
};

SensorRenderer::SensorRenderer() = default;

SensorRenderer::~SensorRenderer() {
  clear();
}

bool SensorRenderer::render(RayraiWindow& viewer, std::vector<PendingSensorUpdate>& updates,
                            std::string& status) {
  // Keep the update handshake alive even when a remote sensor advertises an
  // unusable size. The caller sends an empty acknowledgment if all updates
  // are filtered, and the server can continue serving scene frames.
  updates.erase(std::remove_if(updates.begin(), updates.end(),
      [](const PendingSensorUpdate& update) {
        return !validSensorDimensions(update.info.width, update.info.height) ||
               !validSensorClipRange(update.info.clipNear, update.info.clipFar);
      }), updates.end());
  RayraiWindow::RenderOverrides overrides;
  overrides.doShadows = true;
  overrides.drawCoordinateFrames = false;
  overrides.drawPointClouds = false;
  overrides.drawVisualizationObjects = true;
  overrides.drawWeatherEffects = false;
  overrides.postProcess = false;
  overrides.allowViewerMsaa = false;
  overrides.allowTemporalAa = false;
  overrides.allowViewerUpscale = false;

  for (auto& update : updates) {
    const auto& info = update.info;
    if (!isCameraSensor(info.type) || !validSensorDimensions(info.width, info.height)) {
      status = "unsupported or invalid sensor request: " + info.name;
      return false;
    }

    const std::string key = sensorKey(info);
    auto& cached = sensors_[key];
    if (!cached) {
      cached = std::make_unique<CachedSensor>();
      cached->camera = std::make_unique<Camera>();
    }
    configureCamera(*cached->camera, update);

    const auto started = std::chrono::steady_clock::now();
    if (info.type == raisim::Sensor::Type::RGB) {
      viewer.renderWithExternalCamera(*cached->camera, overrides);
      auto& sensor = readbackSensor<raisim::RGBCamera, raisim::RGBCamera::RGBCameraProperties>(
        cached->rgbSensor, info);
      update.colorBgra.resize(static_cast<size_t>(info.width) *
                              static_cast<size_t>(info.height) * 4u);
      cached->camera->getRawImage(sensor, Camera::SensorStorageMode::CUSTOM_BUFFER,
        reinterpret_cast<char*>(update.colorBgra.data()), update.colorBgra.size(), true);
      cached->preview.texture = cached->camera->getFinalTexture();
      cached->preview.topRowFirst = false;
    } else if (info.type == raisim::Sensor::Type::SEGMENTATION) {
      cached->camera->ensureSegmentationRenderTarget();
      viewer.renderSegmentation(*cached->camera, true, true);
      const size_t pixels = static_cast<size_t>(info.width) * static_cast<size_t>(info.height);
      const bool instances = raisim::SegmentationCamera::hasInstanceImages(info.segmentationType);
      const bool labels = raisim::SegmentationCamera::hasLabelImage(info.segmentationType);
      update.objects.resize(instances ? pixels : 0u);
      update.bodies.resize(instances ? pixels : 0u);
      update.labels.resize(labels ? pixels : 0u);
      cached->camera->readSegmentation(instances ? update.objects.data() : nullptr,
        instances ? update.bodies.data() : nullptr, labels ? update.labels.data() : nullptr, pixels);
      uploadSegmentationPreview(cached->previewTexture, update, cached->previewRgba);
      cached->preview.texture = cached->previewTexture.texture;
      cached->preview.topRowFirst = true;
    } else {
      cached->camera->ensureLinearDepthRenderTarget();
      // TCP physical geometry is marked detectable when RemoteScene reconstructs
      // it. Use the same filter as RGB so viewer-only helpers do not leak into
      // either sensor stream.
      viewer.renderDepthPlaneDistance(*cached->camera, nullptr, true, true);
      auto& sensor =
        readbackSensor<raisim::DepthCamera, raisim::DepthCamera::DepthCameraProperties>(
          cached->depthSensor, info);
      update.depth.resize(static_cast<size_t>(info.width) * static_cast<size_t>(info.height));
      cached->camera->getRawImage(sensor, Camera::SensorStorageMode::CUSTOM_BUFFER,
        update.depth.data(), update.depth.size(), true);
      const float clipNear = static_cast<float>(info.clipNear);
      const float clipFar = static_cast<float>(info.clipFar);
      if (!cached->depthShading || !cached->depthShading->matches(clipNear, clipFar)) {
        cached->depthShading.emplace(clipNear, clipFar);
      }
      uploadDepthPreview(cached->previewTexture, info.width, info.height, update.depth,
        *cached->depthShading, cached->previewRgba, cached->preview.minimumDepth,
        cached->preview.maximumDepth);
      cached->preview.texture = cached->previewTexture.texture;
      cached->preview.topRowFirst = true;
    }
    update.renderMilliseconds = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
    cached->preview.parentTag = info.parentTag;
    cached->preview.type = info.type;
    cached->preview.name = info.name;
    cached->preview.width = info.width;
    cached->preview.height = info.height;
    cached->preview.renderMilliseconds = update.renderMilliseconds;
  }
  status = "rendered " + std::to_string(updates.size()) + " sensor request(s)";
  return true;
}

std::vector<SensorPreviewInfo> SensorRenderer::previews() const {
  std::vector<SensorPreviewInfo> result;
  result.reserve(sensors_.size());
  for (const auto& item : sensors_) {
    result.push_back(item.second->preview);
  }
  std::sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.name < rhs.name;
  });
  return result;
}

std::vector<SensorPreviewInfo> SensorRenderer::previewsForTag(uint32_t parentTag) const {
  auto result = previews();
  result.erase(std::remove_if(result.begin(), result.end(), [parentTag](const auto& preview) {
    return preview.parentTag != parentTag;
  }), result.end());
  return result;
}

void SensorRenderer::clear() {
  if (SDL_GL_GetCurrentContext() != nullptr) {
    for (const auto& item : sensors_) {
      if (item.second) deleteTextureAndInvalidateCache(item.second->previewTexture.texture);
    }
  }
  sensors_.clear();
}

} // namespace raisin::tcp_viewer
