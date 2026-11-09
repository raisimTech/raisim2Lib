// Copyright (c) 2026 Raion Robotics Inc.
// All rights reserved.

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include "raisim/PathUtils.hpp"

namespace raisin::tcp_viewer
{

/**
 * @brief UTF-8 text of @p path with its native separators, for ImGui and status messages.
 *
 * SDL and ImGui speak UTF-8; turn their text into a path with raisim::pathFromUtf8().
 * Unlike std::filesystem::path::string(), which converts to the Windows code page and
 * throws for a name the code page cannot hold, this never throws: on Windows an unpaired
 * UTF-16 surrogate, which no UTF-8 text can hold, becomes U+FFFD.
 */
inline std::string utf8PathText(const std::filesystem::path& path) {
#if defined(_WIN32)
  const std::wstring& wide = path.native();
  std::string text;
  text.reserve(wide.size());
  for (size_t i = 0; i < wide.size(); ++i) {
    uint32_t code = static_cast<uint16_t>(wide[i]);
    if (code >= 0xD800u && code <= 0xDBFFu && i + 1 < wide.size() &&
        static_cast<uint16_t>(wide[i + 1]) >= 0xDC00u &&
        static_cast<uint16_t>(wide[i + 1]) <= 0xDFFFu) {
      code = 0x10000u + ((code - 0xD800u) << 10) + (static_cast<uint16_t>(wide[++i]) - 0xDC00u);
    } else if (code >= 0xD800u && code <= 0xDFFFu) {
      code = 0xFFFDu;
    }
    if (code < 0x80u) {
      text.push_back(static_cast<char>(code));
    } else if (code < 0x800u) {
      text.push_back(static_cast<char>(0xC0u | (code >> 6)));
      text.push_back(static_cast<char>(0x80u | (code & 0x3Fu)));
    } else if (code < 0x10000u) {
      text.push_back(static_cast<char>(0xE0u | (code >> 12)));
      text.push_back(static_cast<char>(0x80u | ((code >> 6) & 0x3Fu)));
      text.push_back(static_cast<char>(0x80u | (code & 0x3Fu)));
    } else {
      text.push_back(static_cast<char>(0xF0u | (code >> 18)));
      text.push_back(static_cast<char>(0x80u | ((code >> 12) & 0x3Fu)));
      text.push_back(static_cast<char>(0x80u | ((code >> 6) & 0x3Fu)));
      text.push_back(static_cast<char>(0x80u | (code & 0x3Fu)));
    }
  }
  return text;
#else
  return path.string();
#endif
}

} // namespace raisin::tcp_viewer
