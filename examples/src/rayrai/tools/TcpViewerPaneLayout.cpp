// Copyright (c) 2026 Raion Robotics Inc.
// All rights reserved.

#include "TcpViewerPaneLayout.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <unordered_set>

namespace raisin::tcp_viewer
{

namespace
{
constexpr float kMinSplitRatio = 0.05f;
constexpr float kMaxSplitRatio = 0.95f;
constexpr uint32_t kFirstPlaceholderId = 0x80000000u;
constexpr int kMaxLayoutDepth = 64;

float clampRatio(float ratio) {
  if (!(ratio == ratio)) return 0.5f; // NaN
  return std::clamp(ratio, kMinSplitRatio, kMaxSplitRatio);
}
} // namespace

PaneLayout::PaneLayout() {
  Node leaf;
  leaf.leaf = true;
  root_ = allocateNode(leaf);
  focused_ = root_;
}

uint32_t PaneLayout::allocateNode(Node node) {
  const uint32_t id = nextId_++;
  nodes_.emplace(id, node);
  return id;
}

size_t PaneLayout::paneCount() const {
  size_t count = 0;
  for (const auto& [id, node] : nodes_) {
    (void)id;
    if (node.leaf) ++count;
  }
  return count;
}

void PaneLayout::collectPanes(uint32_t node, std::vector<uint32_t>& out) const {
  const auto it = nodes_.find(node);
  if (it == nodes_.end()) return;
  if (it->second.leaf) {
    out.push_back(node);
    return;
  }
  collectPanes(it->second.first, out);
  collectPanes(it->second.second, out);
}

std::vector<uint32_t> PaneLayout::panes() const {
  std::vector<uint32_t> out;
  out.reserve(nodes_.size());
  collectPanes(root_, out);
  return out;
}

bool PaneLayout::contains(uint32_t pane) const {
  const auto it = nodes_.find(pane);
  return it != nodes_.end() && it->second.leaf;
}

void PaneLayout::focus(uint32_t pane) {
  if (contains(pane)) focused_ = pane;
}

uint32_t PaneLayout::firstPaneOf(uint32_t node) const {
  const auto it = nodes_.find(node);
  if (it == nodes_.end()) return 0;
  if (it->second.leaf) return node;
  return firstPaneOf(it->second.first);
}

uint32_t PaneLayout::split(uint32_t pane, SplitOrientation orientation) {
  const auto paneIt = nodes_.find(pane);
  if (paneIt == nodes_.end() || !paneIt->second.leaf) return 0;

  const uint32_t oldParent = paneIt->second.parent;

  Node created;
  created.leaf = true;
  const uint32_t createdPane = allocateNode(created);

  Node split;
  split.leaf = false;
  split.orientation = orientation;
  split.ratio = 0.5f;
  split.first = pane;
  split.second = createdPane;
  split.parent = oldParent;
  const uint32_t splitNode = allocateNode(split);

  // allocateNode may rehash, so re-find rather than reusing paneIt.
  nodes_[pane].parent = splitNode;
  nodes_[createdPane].parent = splitNode;

  if (oldParent == 0) {
    root_ = splitNode;
  } else {
    Node& parent = nodes_[oldParent];
    if (parent.first == pane) {
      parent.first = splitNode;
    } else {
      parent.second = splitNode;
    }
  }
  return createdPane;
}

void PaneLayout::eraseSubtree(uint32_t node) {
  const auto it = nodes_.find(node);
  if (it == nodes_.end()) return;
  const Node copy = it->second;
  nodes_.erase(it);
  if (!copy.leaf) {
    eraseSubtree(copy.first);
    eraseSubtree(copy.second);
  }
}

bool PaneLayout::close(uint32_t pane) {
  if (!contains(pane) || paneCount() <= 1) return false;

  const uint32_t parent = nodes_[pane].parent;
  if (parent == 0) return false; // the only pane is the root; guarded above

  const Node parentCopy = nodes_[parent];
  const uint32_t sibling = parentCopy.first == pane ? parentCopy.second : parentCopy.first;
  const uint32_t grandParent = parentCopy.parent;

  nodes_[sibling].parent = grandParent;
  if (grandParent == 0) {
    root_ = sibling;
  } else {
    Node& gp = nodes_[grandParent];
    if (gp.first == parent) {
      gp.first = sibling;
    } else {
      gp.second = sibling;
    }
  }
  nodes_.erase(parent);
  eraseSubtree(pane);

  if (!contains(focused_)) focused_ = firstPaneOf(sibling);
  return true;
}

void PaneLayout::layoutNode(uint32_t node, float x, float y, float width, float height,
                            float splitterThickness, std::vector<PaneRect>& panesOut,
                            std::vector<SplitterHandle>& splittersOut) const {
  const auto it = nodes_.find(node);
  if (it == nodes_.end()) return;
  const Node& n = it->second;
  if (n.leaf) {
    PaneRect rect;
    rect.pane = node;
    rect.x = x;
    rect.y = y;
    rect.width = std::max(0.0f, width);
    rect.height = std::max(0.0f, height);
    panesOut.push_back(rect);
    return;
  }

  const float thickness = std::max(0.0f, splitterThickness);
  if (n.orientation == SplitOrientation::Vertical) {
    const float usable = std::max(0.0f, width - thickness);
    const float firstWidth = usable * n.ratio;
    layoutNode(n.first, x, y, firstWidth, height, splitterThickness, panesOut, splittersOut);
    SplitterHandle handle;
    handle.node = node;
    handle.orientation = n.orientation;
    handle.x = x + firstWidth;
    handle.y = y;
    handle.width = thickness;
    handle.height = std::max(0.0f, height);
    handle.regionX = x;
    handle.regionY = y;
    handle.regionWidth = std::max(0.0f, width);
    handle.regionHeight = std::max(0.0f, height);
    splittersOut.push_back(handle);
    layoutNode(n.second, x + firstWidth + thickness, y, usable - firstWidth, height,
               splitterThickness, panesOut, splittersOut);
  } else {
    const float usable = std::max(0.0f, height - thickness);
    const float firstHeight = usable * n.ratio;
    layoutNode(n.first, x, y, width, firstHeight, splitterThickness, panesOut, splittersOut);
    SplitterHandle handle;
    handle.node = node;
    handle.orientation = n.orientation;
    handle.x = x;
    handle.y = y + firstHeight;
    handle.width = std::max(0.0f, width);
    handle.height = thickness;
    handle.regionX = x;
    handle.regionY = y;
    handle.regionWidth = std::max(0.0f, width);
    handle.regionHeight = std::max(0.0f, height);
    splittersOut.push_back(handle);
    layoutNode(n.second, x, y + firstHeight + thickness, width, usable - firstHeight,
               splitterThickness, panesOut, splittersOut);
  }
}

void PaneLayout::layout(float x, float y, float width, float height, float splitterThickness,
                        std::vector<PaneRect>& panesOut,
                        std::vector<SplitterHandle>& splittersOut) const {
  panesOut.clear();
  splittersOut.clear();
  layoutNode(root_, x, y, width, height, splitterThickness, panesOut, splittersOut);
}

float PaneLayout::splitterRatio(uint32_t node) const {
  const auto it = nodes_.find(node);
  if (it == nodes_.end() || it->second.leaf) return 0.0f;
  return it->second.ratio;
}

bool PaneLayout::setSplitterRatio(uint32_t node, float ratio) {
  const auto it = nodes_.find(node);
  if (it == nodes_.end() || it->second.leaf) return false;
  it->second.ratio = clampRatio(ratio);
  return true;
}

void PaneLayout::appendSerialized(uint32_t node, std::string& out) const {
  const auto it = nodes_.find(node);
  if (it == nodes_.end()) return;
  const Node& n = it->second;
  if (n.leaf) {
    out += 'L';
    out += std::to_string(node);
    return;
  }
  std::array<char, 32> ratioText{};
  std::snprintf(ratioText.data(), ratioText.size(), "%.4f", n.ratio);
  out += n.orientation == SplitOrientation::Horizontal ? 'H' : 'V';
  out += ratioText.data();
  out += '(';
  appendSerialized(n.first, out);
  out += ',';
  appendSerialized(n.second, out);
  out += ')';
}

std::string PaneLayout::serialize() const {
  std::string out;
  appendSerialized(root_, out);
  return out;
}

bool PaneLayout::parseNode(const std::string& text, size_t& pos,
                           std::unordered_map<uint32_t, Node>& nodes, uint32_t parent,
                           uint32_t& nextId, uint32_t& out, int depth) {
  if (pos >= text.size() || depth > kMaxLayoutDepth) return false;
  const char kind = text[pos];
  if (kind == 'L') {
    ++pos;
    const size_t start = pos;
    while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) ++pos;
    if (pos == start) return false;
    const unsigned long value = std::strtoul(text.substr(start, pos - start).c_str(), nullptr, 10);
    if (value == 0 || value >= kFirstPlaceholderId) return false;
    const auto id = static_cast<uint32_t>(value);
    if (nodes.count(id) != 0) return false; // duplicate pane id
    Node leaf;
    leaf.leaf = true;
    leaf.parent = parent;
    nodes.emplace(id, leaf);
    nextId = std::max(nextId, id + 1);
    out = id;
    return true;
  }
  if (kind != 'H' && kind != 'V') return false;
  ++pos;
  const size_t ratioStart = pos;
  while (pos < text.size() && text[pos] != '(') ++pos;
  if (pos >= text.size() || pos == ratioStart) return false;
  const std::string ratioText = text.substr(ratioStart, pos - ratioStart);
  char* ratioEnd = nullptr;
  const float ratio = std::strtof(ratioText.c_str(), &ratioEnd);
  if (ratioEnd == nullptr || *ratioEnd != '\0') return false;
  ++pos; // consume '('

  // The split node needs an id before its children so they can point at it, but
  // its id must not collide with a pane id that appears later in the text. Take
  // a placeholder now and renumber once the whole subtree is known.
  const uint32_t placeholder = kFirstPlaceholderId + static_cast<uint32_t>(nodes.size());
  Node split;
  split.leaf = false;
  split.orientation = kind == 'H' ? SplitOrientation::Horizontal : SplitOrientation::Vertical;
  split.ratio = clampRatio(ratio);
  split.parent = parent;
  nodes.emplace(placeholder, split);

  uint32_t firstChild = 0;
  if (!parseNode(text, pos, nodes, placeholder, nextId, firstChild, depth + 1)) return false;
  if (pos >= text.size() || text[pos] != ',') return false;
  ++pos;
  uint32_t secondChild = 0;
  if (!parseNode(text, pos, nodes, placeholder, nextId, secondChild, depth + 1)) return false;
  if (pos >= text.size() || text[pos] != ')') return false;
  ++pos;

  nodes[placeholder].first = firstChild;
  nodes[placeholder].second = secondChild;
  out = placeholder;
  return true;
}

bool PaneLayout::deserialize(const std::string& text) {
  if (text.empty()) return false;
  std::unordered_map<uint32_t, Node> parsed;
  uint32_t nextId = 1;
  size_t pos = 0;
  uint32_t root = 0;
  if (!parseNode(text, pos, parsed, 0, nextId, root, 0)) return false;
  if (pos != text.size()) return false;

  // Renumber the placeholder ids handed to split nodes into the shared id space,
  // now that every pane id in the text is known.
  std::unordered_map<uint32_t, uint32_t> remap;
  for (const auto& [id, node] : parsed) {
    (void)node;
    if (id >= kFirstPlaceholderId) remap.emplace(id, nextId++);
  }
  std::unordered_map<uint32_t, Node> finalNodes;
  finalNodes.reserve(parsed.size());
  const auto resolve = [&remap](uint32_t id) {
    if (id == 0) return 0u;
    const auto it = remap.find(id);
    return it == remap.end() ? id : it->second;
  };
  for (const auto& [id, node] : parsed) {
    Node copy = node;
    copy.parent = resolve(copy.parent);
    copy.first = resolve(copy.first);
    copy.second = resolve(copy.second);
    finalNodes.emplace(resolve(id), copy);
  }

  nodes_ = std::move(finalNodes);
  root_ = resolve(root);
  nextId_ = nextId;
  focused_ = firstPaneOf(root_);
  return focused_ != 0;
}

} // namespace raisin::tcp_viewer
