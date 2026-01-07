#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>

namespace static_autograd {

inline std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) {
        return "";
    }
    size_t end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

inline std::unordered_map<std::string, std::string> load_kv_file(const std::string& path) {
    std::unordered_map<std::string, std::string> kv;
    std::ifstream file(path);
    if (!file) {
        return kv;
    }

    std::string line;
    while (std::getline(file, line)) {
        auto comment = line.find('#');
        if (comment != std::string::npos) {
            line = line.substr(0, comment);
        }
        line = trim(line);
        if (line.empty()) {
            continue;
        }
        auto colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        auto key = trim(line.substr(0, colon));
        auto value = trim(line.substr(colon + 1));
        if (!key.empty()) {
            kv[key] = value;
        }
    }

    return kv;
}

inline void get_u32(const std::unordered_map<std::string, std::string>& kv,
                    const std::string& key, uint32_t& out) {
    auto it = kv.find(key);
    if (it != kv.end()) {
        out = static_cast<uint32_t>(std::stoul(it->second));
    }
}

inline void get_f32(const std::unordered_map<std::string, std::string>& kv,
                    const std::string& key, float& out) {
    auto it = kv.find(key);
    if (it != kv.end()) {
        out = std::stof(it->second);
    }
}

inline void get_str(const std::unordered_map<std::string, std::string>& kv,
                    const std::string& key, std::string& out) {
    auto it = kv.find(key);
    if (it != kv.end()) {
        out = it->second;
    }
}

} // namespace static_autograd
