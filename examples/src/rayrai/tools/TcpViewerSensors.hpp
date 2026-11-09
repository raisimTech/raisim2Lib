// Copyright (c) 2026 Raion Robotics Inc.
// All rights reserved.

#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

#include "rayrai/TcpRemoteSceneState.hpp"

namespace raisin
{
class Camera;
class RayraiWindow;
}

namespace raisin::tcp_viewer
{

namespace detail
{

/**
 * Log grayscale for depth previews, anchored to a camera clip range.
 *
 * A shade never decreases with depth, so the mapping is stored as the smallest depth that
 * reaches each shade, found with the per-pixel formula itself. A pixel then needs a bucket
 * lookup and an integer bias instead of a logarithm, and gets the formula's shade.
 */
class DepthPreviewShading {
 public:
  /** Read-only view; the owning table must remain alive and unchanged while it is used. */
  class BatchLookup {
   public:
    /** Four finite depths within the effective clip range; returns exact word-sized shades. */
    std::array<uint32_t, 4> shadesWithinRange(const std::array<float, 4>& depth) const {
      std::array<uint32_t, 4> bits, offsets, indices, shades;
      std::memcpy(bits.data(), depth.data(), sizeof(bits));
#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(disable) unroll(disable)
#endif
      for (size_t lane = 0; lane < 4u; ++lane) {
        offsets[lane] = bits[lane] - origin_;
        indices[lane] = offsets[lane] >> shift_;
      }
      for (size_t lane = 0; lane < 4u; ++lane)
        shades[lane] = entries_[indices[lane]];
#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(disable) unroll(disable)
#endif
      for (size_t lane = 0; lane < 4u; ++lane)
        shades[lane] = (offsets[lane] + shades[lane]) >> shift_;
      return shades;
    }

   private:
    friend class DepthPreviewShading;
    BatchLookup(uint32_t layout, const uint32_t* entries)
        : origin_(layout & ~UINT32_C(31)), shift_(layout & 31u), entries_(entries) {}
    uint32_t origin_;
    uint32_t shift_;
    const uint32_t* entries_;
  };

  DepthPreviewShading(float clipMinimum, float clipMaximum);

  /** @return True when this table was built for the given clip range. */
  bool matches(float clipMinimum, float clipMaximum) const {
    return clipMinimum == clipMinimum_ && clipMaximum == clipMaximum_;
  }

  /**
   * @param depth Finite, positive depth.
   * @return Shade of @p depth.
   */
  unsigned char shade(float depth) const;

  /** @param depth Finite depth within the effective clip range. */
  unsigned char shadeWithinRange(float depth) const {
    uint32_t bits;
    std::memcpy(&bits, &depth, sizeof(bits));
    const uint32_t layout = bucketLayout_;
    const uint32_t shift = layout & 31u;
    const uint32_t offset = bits - (layout & ~UINT32_C(31));
    const uint32_t bucket = bucketEntries_[offset >> shift];
    return static_cast<unsigned char>((offset + bucket) >> shift);
  }

  /**
   * @param depth Four finite depths within the effective clip range.
   * @return Exact shades as words to keep the lane arithmetic vectorizable.
   */
  std::array<uint32_t, 4> shadesWithinRange(const std::array<float, 4>& depth) const {
    return batchLookup().shadesWithinRange(depth);
  }

  /** Capture the immutable table pointer and index settings for a conversion run. */
  BatchLookup batchLookup() const { return BatchLookup(bucketLayout_, bucketEntries_.data()); }

  /// Exact interval of raw positive depths producing this shade, including clamped returns.
  std::pair<float, float> interval(unsigned char shade) const {
    return {bounds_[shade], bounds_[size_t(shade) + 1u]};
  }

 private:
  float clipMinimum_;
  float clipMaximum_;
  float scaleMinimum_;
  float scaleMaximum_;
  // A depth has shade k when bounds_[k] <= depth < bounds_[k + 1]. Shades no depth reaches
  // have empty intervals.
  std::array<float, 257> bounds_{};
  // Bucket widths in ordered float words never exceed the smallest shade-boundary
  // gap. A one-float bucket handles repeated boundaries. Each entry biases the
  // input offset so shifting gives the lower-edge shade, with a carry into the
  // next shade exactly at the boundary. Unsigned arithmetic wraps intentionally.
  // The aligned origin's float word occupies the upper 27 bits; the low five
  // bits hold the bucket shift (0-16). Both are read together during lookup.
  uint32_t bucketLayout_ = 0;
  std::vector<uint32_t> bucketEntries_;
};

/**
 * Convert metric depth to RGBA using a log grayscale anchored to the camera clip range.
 * @param rgba Resized to four bytes per depth sample; reusing it across frames avoids an
 *        allocation per frame.
 */
void makeDepthPreviewRgba(const std::vector<float>& depth, const DepthPreviewShading& shading,
  std::vector<unsigned char>& rgba, float& minimumDepth, float& maximumDepth);

/** Same as above, building the shading table and the image for one call. */
std::vector<unsigned char> makeDepthPreviewRgba(const std::vector<float>& depth,
  float clipMinimum, float clipMaximum, float& minimumDepth, float& maximumDepth);

} // namespace detail

struct SensorPreviewInfo {
  uint32_t parentTag = 0;
  raisim::Sensor::Type type = raisim::Sensor::Type::UNKNOWN;
  std::string name;
  int width = 0;
  int height = 0;
  unsigned int texture = 0;
  double renderMilliseconds = 0.0;
  float minimumDepth = 0.0f;
  float maximumDepth = 0.0f;
  /** The texture holds the image's top row first, as the depth and segmentation previews
   *  uploaded from top-down images do; the RGB preview is the camera's GL texture, whose first
   *  row is the bottom of the image. */
  bool topRowFirst = false;
};

/** Renders manual TCP camera requests and retains textures for the diagnostics UI. */
class SensorRenderer {
 public:
  SensorRenderer();
  ~SensorRenderer();
  SensorRenderer(const SensorRenderer&) = delete;
  SensorRenderer& operator=(const SensorRenderer&) = delete;

  bool render(RayraiWindow& viewer, std::vector<PendingSensorUpdate>& updates,
              std::string& status);
  std::vector<SensorPreviewInfo> previews() const;
  std::vector<SensorPreviewInfo> previewsForTag(uint32_t parentTag) const;
  void clear();

 private:
  struct CachedSensor;
  std::unordered_map<std::string, std::unique_ptr<CachedSensor>> sensors_;
};

} // namespace raisin::tcp_viewer
