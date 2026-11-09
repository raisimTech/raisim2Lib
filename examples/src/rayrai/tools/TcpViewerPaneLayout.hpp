// Copyright (c) 2026 Raion Robotics Inc.
// All rights reserved.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace raisin::tcp_viewer
{

/**
 * @brief How an internal split node divides its rectangle.
 *
 * The names follow Terminator's menu wording, which describes the divider, not
 * the stacking direction: a *horizontal* split draws a horizontal divider and
 * stacks its children top/bottom, a *vertical* split draws a vertical divider
 * and places its children side by side.
 */
enum class SplitOrientation { Horizontal, Vertical };

/** @brief Screen rectangle assigned to one pane by PaneLayout::layout(). */
struct PaneRect {
  uint32_t pane = 0;
  float x = 0.0f;
  float y = 0.0f;
  float width = 0.0f;
  float height = 0.0f;
};

/** @brief Draggable divider between the two children of one split node. */
struct SplitterHandle {
  uint32_t node = 0;
  SplitOrientation orientation = SplitOrientation::Vertical;
  float x = 0.0f;
  float y = 0.0f;
  float width = 0.0f;
  float height = 0.0f;
  // Rectangle the divider splits, i.e. both children plus the divider itself.
  // A drag turns a cursor position into a ratio against this, not the screen.
  float regionX = 0.0f;
  float regionY = 0.0f;
  float regionWidth = 0.0f;
  float regionHeight = 0.0f;
};

/**
 * @brief Terminator-style binary split tree over the viewer client area.
 *
 * Leaves are panes and carry the pane id the rest of the viewer keys its
 * per-connection state by; internal nodes are splits with a ratio. Splitting a
 * pane keeps that pane's id on the first child, so ids stay stable for the
 * lifetime of a pane and can be used for ImGui window names and for the
 * settings file.
 *
 * The class is pure geometry and bookkeeping: it holds no GL, ImGui or network
 * state, which keeps it unit-testable without a display.
 */
class PaneLayout
{
 public:
  /** @brief Construct a layout with a single pane whose id is 1. */
  PaneLayout();

  /** @brief Number of panes (tree leaves) currently in the layout. */
  [[nodiscard]] size_t paneCount() const;
  /** @brief Pane ids in layout order (left to right, top to bottom). */
  [[nodiscard]] std::vector<uint32_t> panes() const;
  /** @brief True when @p pane names a live pane. */
  [[nodiscard]] bool contains(uint32_t pane) const;
  /** @brief Id of the pane that currently owns keyboard/viewport focus. */
  [[nodiscard]] uint32_t focused() const { return focused_; }
  /** @brief Focus @p pane; ignored when it is not a live pane. */
  void focus(uint32_t pane);

  /**
   * @brief Split @p pane in two along @p orientation.
   * @return Id of the newly created pane, or 0 when @p pane is unknown.
   *
   * @p pane keeps its id and takes the left/top half; the returned pane takes
   * the right/bottom half.
   */
  uint32_t split(uint32_t pane, SplitOrientation orientation);

  /**
   * @brief Remove @p pane, giving its space back to its sibling.
   * @return False when @p pane is unknown or is the only remaining pane.
   */
  bool close(uint32_t pane);

  /**
   * @brief Assign screen rectangles to every pane and divider.
   * @param x Client-area origin x.
   * @param y Client-area origin y.
   * @param width Client-area width.
   * @param height Client-area height.
   * @param splitterThickness Divider thickness in the same units, subtracted
   *        from the split axis before the ratio is applied.
   * @param panesOut Receives one entry per pane, in layout order.
   * @param splittersOut Receives one entry per divider.
   */
  void layout(float x, float y, float width, float height, float splitterThickness,
              std::vector<PaneRect>& panesOut, std::vector<SplitterHandle>& splittersOut) const;

  /** @brief Current ratio of split node @p node, or 0 when it is not a split. */
  [[nodiscard]] float splitterRatio(uint32_t node) const;
  /** @brief Set the ratio of split node @p node, clamped to [0.05, 0.95]. */
  bool setSplitterRatio(uint32_t node, float ratio);

  /**
   * @brief Encode the tree as a single line, e.g. "V0.5(L1,H0.5(L2,L3))".
   *
   * Leaves are "L<paneId>"; splits are "<H|V><ratio>(<first>,<second>)".
   */
  [[nodiscard]] std::string serialize() const;

  /**
   * @brief Replace the layout with the tree encoded by @p text.
   * @return False when @p text does not parse, leaving the layout unchanged. Pane ids
   *   must be below 0x80000000 and splits may nest at most 64 deep.
   */
  bool deserialize(const std::string& text);

 private:
  struct Node {
    bool leaf = true;
    uint32_t parent = 0;
    SplitOrientation orientation = SplitOrientation::Vertical;
    float ratio = 0.5f;
    uint32_t first = 0;
    uint32_t second = 0;
  };

  uint32_t allocateNode(Node node);
  void collectPanes(uint32_t node, std::vector<uint32_t>& out) const;
  void layoutNode(uint32_t node, float x, float y, float width, float height,
                  float splitterThickness, std::vector<PaneRect>& panesOut,
                  std::vector<SplitterHandle>& splittersOut) const;
  void appendSerialized(uint32_t node, std::string& out) const;
  void eraseSubtree(uint32_t node);
  [[nodiscard]] uint32_t firstPaneOf(uint32_t node) const;
  static bool parseNode(const std::string& text, size_t& pos,
                        std::unordered_map<uint32_t, Node>& nodes, uint32_t parent,
                        uint32_t& nextId, uint32_t& out, int depth);

  std::unordered_map<uint32_t, Node> nodes_;
  uint32_t root_ = 0;
  uint32_t focused_ = 0;
  uint32_t nextId_ = 1;
};

} // namespace raisin::tcp_viewer
