// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "tensor.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace tensordiffgradv2 {

enum class OpType {
    Unary,
    Binary,
};

inline std::string op_type_str(OpType t) {
    return t == OpType::Binary ? "binary" : "unary";
}

// Info about a pending composite op (waiting for sub-ops to complete)
struct CompositeInfo {
    std::string name;           // Op name (e.g., "reglu")
    int idx;                    // Reserved idx for this composite
    std::string first_input;    // First sub-op's input path (set on first sub-op)
    std::string first_input_shape;
    std::string first_input_name;
    std::string last_output;    // Last sub-op's output path (updated on each sub-op)
    std::string last_output_shape;
    std::string backend;
    std::string dtype;
};

class OpRecorder {
public:
    void enable(const std::filesystem::path& dir) {
        enabled_ = true;
        dir_ = dir;
        counter_ = 0;
        manifest_initialized_ = false;
        composite_stack_.clear();
        std::filesystem::create_directories(dir_);
    }

    void disable() { enabled_ = false; }
    bool enabled() const { return enabled_; }
    const std::filesystem::path& dir() const { return dir_; }
    size_t counter() const { return counter_; }

    // Get current parent_id (-1 if no composite is active)
    int current_parent_id() const {
        return composite_stack_.empty() ? -1 : composite_stack_.back().idx;
    }

    // Start a composite op - reserves an idx and tracks sub-ops
    // Returns the reserved idx for the composite
    int push_composite(const std::string& name) {
        CompositeInfo info;
        info.name = name;
        info.idx = static_cast<int>(counter_++);  // Reserve idx
        composite_stack_.push_back(info);
        return info.idx;
    }

    // End a composite op - records it with reference to first/last sub-op tensors
    // output_name: name for the composite op output (e.g., "reglu")
    void pop_composite(const std::string& output_name) {
        if (composite_stack_.empty()) return;

        CompositeInfo info = composite_stack_.back();
        composite_stack_.pop_back();

        // Record composite op referencing sub-op tensors
        int parent_id = current_parent_id();  // Parent of this composite (if nested)
        append_manifest(info.name, OpType::Unary, parent_id,
                        {info.first_input}, {info.first_input_shape}, {info.first_input_name},
                        info.last_output, info.last_output_shape, output_name,
                        info.backend, info.dtype, "", info.idx);
    }

    // Record an op with its inputs and output
    // Saves tensors to TDF1 files and appends to ops.tsv
    void record(std::string_view op_name,
                OpType type,
                const std::vector<const Tensor*>& inputs,
                const std::vector<std::string>& input_names,
                const Tensor& output,
                const std::string& output_name,
                std::string_view opts = {}) {
        if (!enabled_) return;

        const std::string safe_op = sanitize(op_name);
        const std::string prefix = make_prefix(safe_op);

        // Save inputs
        std::vector<std::string> input_paths;
        std::vector<std::string> input_shapes;
        input_paths.reserve(inputs.size());
        input_shapes.reserve(inputs.size());

        for (size_t i = 0; i < inputs.size(); ++i) {
            std::ostringstream fname;
            fname << prefix << "_in" << i << ".bin";
            const auto path = (dir_ / fname.str()).string();
            inputs[i]->save(path);
            input_paths.push_back(fname.str());
            input_shapes.push_back(shape_str(inputs[i]->shape()));
        }

        // Save output
        std::ostringstream out_fname;
        out_fname << prefix << "_out.bin";
        const auto out_path = (dir_ / out_fname.str()).string();
        output.save(out_path);

        // Track first/last sub-op for composite
        if (!composite_stack_.empty()) {
            auto& info = composite_stack_.back();
            // First sub-op: capture input info
            if (info.first_input.empty()) {
                info.first_input = input_paths.empty() ? "" : input_paths[0];
                info.first_input_shape = input_shapes.empty() ? "" : input_shapes[0];
                info.first_input_name = input_names.empty() ? "" : input_names[0];
                info.backend = output.backend();
                info.dtype = dtype_name(output.dtype());
            }
            // Always update last output (so composite knows final output)
            info.last_output = out_fname.str();
            info.last_output_shape = shape_str(output.shape());
        }

        // Append to manifest with parent_id
        append_manifest(op_name, type, current_parent_id(),
                        input_paths, input_shapes, input_names,
                        out_fname.str(), shape_str(output.shape()), output_name,
                        output.backend(), dtype_name(output.dtype()), opts);
    }

private:
    bool enabled_ = false;
    bool manifest_initialized_ = false;
    size_t counter_ = 0;
    std::filesystem::path dir_;
    std::vector<CompositeInfo> composite_stack_;  // Stack for nested composites

    static std::string sanitize(std::string_view name) {
        std::string out;
        out.reserve(name.size());
        for (char c : name) {
            if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-') {
                out.push_back(c);
            } else {
                out.push_back('_');
            }
        }
        return out.empty() ? "op" : out;
    }

    std::string make_prefix(const std::string& name) {
        std::ostringstream oss;
        oss << std::setw(4) << std::setfill('0') << counter_++ << "_" << name;
        return oss.str();
    }

    static std::string shape_str(const std::vector<uint32_t>& dims) {
        if (dims.empty()) return "";
        std::ostringstream oss;
        for (size_t i = 0; i < dims.size(); ++i) {
            if (i) oss << "x";
            oss << dims[i];
        }
        return oss.str();
    }

    // Append to manifest with parent_id
    // idx_override: use this idx instead of counter_-1 (for composite ops recorded out-of-order)
    void append_manifest(std::string_view op_name,
                         OpType type,
                         int parent_id,
                         const std::vector<std::string>& input_paths,
                         const std::vector<std::string>& input_shapes,
                         const std::vector<std::string>& input_names,
                         const std::string& output_path,
                         const std::string& output_shape,
                         const std::string& output_name,
                         const std::string& backend,
                         const std::string& dtype,
                         std::string_view opts,
                         int idx_override = -1) {
        const auto path = dir_ / "ops.tsv";
        std::ofstream f(path, std::ios::app);
        if (!f) return;

        if (!manifest_initialized_) {
            f << "idx\top\ttype\tparent_id\tinputs\tinput_shapes\tinput_names\t"
              << "output\toutput_shape\toutput_name\tbackend\tdtype\topts\n";
            manifest_initialized_ = true;
        }

        int idx = (idx_override >= 0) ? idx_override : static_cast<int>(counter_ - 1);
        f << idx << "\t" << op_name << "\t" << op_type_str(type) << "\t" << parent_id << "\t";

        // inputs
        for (size_t i = 0; i < input_paths.size(); ++i) {
            if (i) f << ",";
            f << input_paths[i];
        }
        f << "\t";

        // input_shapes
        for (size_t i = 0; i < input_shapes.size(); ++i) {
            if (i) f << ",";
            f << input_shapes[i];
        }
        f << "\t";

        // input_names
        for (size_t i = 0; i < input_names.size(); ++i) {
            if (i) f << ",";
            f << input_names[i];
        }
        f << "\t";

        f << output_path << "\t" << output_shape << "\t" << output_name << "\t"
          << backend << "\t" << dtype << "\t" << opts << "\n";
    }
};

}  // namespace tensordiffgradv2
