// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cpu_tensor.hpp"
#include "serialize.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <sstream>
#include <cctype>
#include <utility>

namespace tensordiff::core {

#ifndef TENSORDIFF_ENABLE_RECORDING
#define TENSORDIFF_ENABLE_RECORDING 1
#endif

enum class OpType {
    Unary,
    Binary,
};

struct OpRecord {
    std::string name;
    OpType type;
    std::vector<std::string> inputs;
    std::vector<std::string> input_shapes;
    std::vector<std::string> input_origins;
    std::vector<std::string> input_dtypes;
    std::string output;
    std::string output_shape;
    std::string output_origin;
    std::string output_dtype;
    std::string opts;
};

class BaseTensor {
public:
    virtual ~BaseTensor() = default;

    virtual std::vector<uint32_t> shape() const = 0;
    virtual DType dtype() const = 0;
    virtual std::string backend() const = 0;

    std::unique_ptr<BaseTensor> add(const BaseTensor& other) const {
        return handle_binary("add", other, [this, &other]() { return this->add_impl(other); });
    }
    std::unique_ptr<BaseTensor> sub(const BaseTensor& other) const {
        return handle_binary("sub", other, [this, &other]() { return this->sub_impl(other); });
    }
    std::unique_ptr<BaseTensor> mm(const BaseTensor& other) const {
        return handle_binary("mm", other, [this, &other]() { return this->mm_impl(other); });
    }

    virtual CpuTensor to_cpu() const = 0;

protected:
    virtual std::unique_ptr<BaseTensor> add_impl(const BaseTensor& other) const = 0;
    virtual std::unique_ptr<BaseTensor> sub_impl(const BaseTensor& other) const = 0;
    virtual std::unique_ptr<BaseTensor> mm_impl(const BaseTensor& other) const = 0;

    template <typename Fn>
    std::unique_ptr<BaseTensor> handle_op(std::string_view name,
                                          OpType type,
                                          const std::vector<const BaseTensor*>& inputs,
                                          Fn&& fn,
                                          std::string_view opts = {}) const {
        auto out = fn();
        record_if_enabled(name, type, inputs, *out, opts);
        return out;
    }

    template <typename Fn>
    std::unique_ptr<BaseTensor> handle_unary(std::string_view name, Fn&& fn) const {
        return handle_op(name, OpType::Unary, {this}, std::forward<Fn>(fn));
    }

    template <typename Fn>
    std::unique_ptr<BaseTensor> handle_binary(std::string_view name,
                                              const BaseTensor& other,
                                              Fn&& fn) const {
        return handle_op(name, OpType::Binary, {this, &other}, std::forward<Fn>(fn));
    }

    void record_op(std::string_view name,
                   OpType type,
                   const std::vector<const BaseTensor*>& inputs,
                   const BaseTensor& output,
                   std::string_view opts = {}) const {
        record_if_enabled(name, type, inputs, output, opts);
    }

private:
    void record_if_enabled(std::string_view name,
                           OpType type,
                           const std::vector<const BaseTensor*>& inputs,
                           const BaseTensor& output,
                           std::string_view opts) const {
#if TENSORDIFF_ENABLE_RECORDING
        if (op_recorder().enabled()) {
            op_recorder().record(name, type, inputs, output, opts);
        }
#else
        (void)name;
        (void)type;
        (void)inputs;
        (void)output;
        (void)opts;
#endif
    }

public:
    class OpRecorder {
    public:
        void enable(const std::filesystem::path& dir) {
#if TENSORDIFF_ENABLE_RECORDING
            enabled_ = true;
            dir_ = dir;
            counter_ = 0;
            records_.clear();
            manifest_initialized_ = false;
            std::filesystem::create_directories(dir_);
#else
            (void)dir;
#endif
        }

        void disable() {
#if TENSORDIFF_ENABLE_RECORDING
            enabled_ = false;
#endif
        }

        bool enabled() const { return enabled_; }

        void record(std::string_view op_name,
                    OpType type,
                    const std::vector<const BaseTensor*>& inputs,
                    const BaseTensor& output,
                    std::string_view opts = {}) {
#if TENSORDIFF_ENABLE_RECORDING
            if (!enabled_) return;
            const std::string safe = sanitize(op_name);
            const std::string prefix = make_prefix(safe);
            std::vector<std::string> in_paths;
            in_paths.reserve(inputs.size());
            std::vector<std::string> in_shapes;
            in_shapes.reserve(inputs.size());
            std::vector<std::string> in_origins;
            in_origins.reserve(inputs.size());
            std::vector<std::string> in_dtypes;
            in_dtypes.reserve(inputs.size());
            for (size_t i = 0; i < inputs.size(); ++i) {
                const auto& in = *inputs[i];
                std::ostringstream name;
                name << prefix << "_in" << i << ".bin";
                const auto path = (dir_ / name.str()).string();
                save_tensor(in.to_cpu(), path, in.backend());
                in_paths.push_back(path);
                in_shapes.push_back(shape_to_string(in.shape()));
                in_origins.push_back(in.backend());
                in_dtypes.push_back(dtype_name(in.dtype()));
            }
            std::ostringstream out_name;
            out_name << prefix << "_out.bin";
            const auto out_path = (dir_ / out_name.str()).string();
            save_tensor(output.to_cpu(), out_path, output.backend(), op_name);

            OpRecord rec;
            rec.name = std::string(op_name);
            rec.type = type;
            rec.inputs = std::move(in_paths);
            rec.input_shapes = std::move(in_shapes);
            rec.input_origins = std::move(in_origins);
            rec.input_dtypes = std::move(in_dtypes);
            rec.output = out_path;
            rec.output_shape = shape_to_string(output.shape());
            rec.output_origin = output.backend();
            rec.output_dtype = dtype_name(output.dtype());
            rec.opts = sanitize_opts(opts);
            records_.push_back(rec);
            append_manifest(rec);
#else
            (void)op_name;
            (void)type;
            (void)inputs;
            (void)output;
            (void)opts;
#endif
        }

        void record_cpu(std::string_view op_name,
                        OpType type,
                        const std::vector<const CpuTensor*>& inputs,
                        const std::vector<std::string_view>& input_origins,
                        const CpuTensor& output,
                        std::string_view output_origin,
                        std::string_view opts = {}) {
#if TENSORDIFF_ENABLE_RECORDING
            if (!enabled_) return;
            const std::string safe = sanitize(op_name);
            const std::string prefix = make_prefix(safe);
            std::vector<std::string> in_paths;
            in_paths.reserve(inputs.size());
            std::vector<std::string> in_shapes;
            in_shapes.reserve(inputs.size());
            std::vector<std::string> in_origins;
            in_origins.reserve(inputs.size());
            std::vector<std::string> in_dtypes;
            in_dtypes.reserve(inputs.size());
            for (size_t i = 0; i < inputs.size(); ++i) {
                const auto& in = *inputs[i];
                const auto origin = i < input_origins.size() ? input_origins[i] : std::string_view("cpu");
                std::ostringstream name;
                name << prefix << "_in" << i << ".bin";
                const auto path = (dir_ / name.str()).string();
                save_tensor(in, path, origin);
                in_paths.push_back(path);
                in_shapes.push_back(shape_to_string(in.shape()));
                in_origins.push_back(std::string(origin));
                in_dtypes.push_back(dtype_name(in.dtype()));
            }
            std::ostringstream out_name;
            out_name << prefix << "_out.bin";
            const auto out_path = (dir_ / out_name.str()).string();
            save_tensor(output, out_path, output_origin, op_name);

            OpRecord rec;
            rec.name = std::string(op_name);
            rec.type = type;
            rec.inputs = std::move(in_paths);
            rec.input_shapes = std::move(in_shapes);
            rec.input_origins = std::move(in_origins);
            rec.input_dtypes = std::move(in_dtypes);
            rec.output = out_path;
            rec.output_shape = shape_to_string(output.shape());
            rec.output_origin = std::string(output_origin);
            rec.output_dtype = dtype_name(output.dtype());
            rec.opts = sanitize_opts(opts);
            records_.push_back(rec);
            append_manifest(rec);
#else
            (void)op_name;
            (void)type;
            (void)inputs;
            (void)input_origins;
            (void)output;
            (void)output_origin;
            (void)opts;
#endif
        }

    private:
        bool enabled_ = false;
        bool manifest_initialized_ = false;
        size_t counter_ = 0;
        std::filesystem::path dir_;
        std::vector<OpRecord> records_;

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

        static std::string shape_to_string(const std::vector<uint32_t>& dims) {
            if (dims.empty()) {
                return "";
            }
            std::ostringstream oss;
            for (size_t i = 0; i < dims.size(); ++i) {
                if (i) oss << "x";
                oss << dims[i];
            }
            return oss.str();
        }

        static std::string sanitize_opts(std::string_view opts) {
            std::string out;
            out.reserve(opts.size());
            for (char c : opts) {
                if (c == '\t' || c == '\n' || c == '\r') {
                    out.push_back(' ');
                } else {
                    out.push_back(c);
                }
            }
            return out;
        }

        std::string make_prefix(const std::string& name) {
            std::ostringstream oss;
            oss << std::setw(4) << std::setfill('0') << counter_++ << "_" << name;
            return oss.str();
        }

        void append_manifest(const OpRecord& rec) {
            const auto path = dir_ / "ops.tsv";
            std::ofstream f(path, std::ios::app);
            if (!f) return;
            if (!manifest_initialized_) {
                f << "idx\top\ttype\tinputs\tinput_shapes\tinput_origins\tinput_dtypes\toutput\toutput_shape\toutput_origin\toutput_dtype\topts\n";
                manifest_initialized_ = true;
            }
            f << (counter_ - 1) << "\t" << rec.name << "\t"
              << (rec.type == OpType::Binary ? "binary" : "unary") << "\t";
            for (size_t i = 0; i < rec.inputs.size(); ++i) {
                if (i) f << ",";
                f << rec.inputs[i];
            }
            f << "\t";
            for (size_t i = 0; i < rec.input_shapes.size(); ++i) {
                if (i) f << ",";
                f << rec.input_shapes[i];
            }
            f << "\t";
            for (size_t i = 0; i < rec.input_origins.size(); ++i) {
                if (i) f << ",";
                f << rec.input_origins[i];
            }
            f << "\t";
            for (size_t i = 0; i < rec.input_dtypes.size(); ++i) {
                if (i) f << ",";
                f << rec.input_dtypes[i];
            }
            f << "\t" << rec.output << "\t" << rec.output_shape << "\t"
              << rec.output_origin << "\t" << rec.output_dtype << "\t"
              << rec.opts << "\n";
        }
    };

    static OpRecorder& op_recorder() {
        static OpRecorder recorder;
        return recorder;
    }
};

inline BaseTensor::OpRecorder& op_recorder() {
    return BaseTensor::op_recorder();
}

inline void enable_op_recording(const std::filesystem::path& dir) {
#if TENSORDIFF_ENABLE_RECORDING
    op_recorder().enable(dir);
#else
    (void)dir;
#endif
}

inline void disable_op_recording() {
#if TENSORDIFF_ENABLE_RECORDING
    op_recorder().disable();
#endif
}

inline void record_op_cpu(std::string_view op_name,
                          OpType type,
                          const std::vector<const CpuTensor*>& inputs,
                          const std::vector<std::string_view>& input_origins,
                          const CpuTensor& output,
                          std::string_view output_origin,
                          std::string_view opts = {}) {
#if TENSORDIFF_ENABLE_RECORDING
    op_recorder().record_cpu(op_name, type, inputs, input_origins, output, output_origin, opts);
#else
    (void)op_name;
    (void)type;
    (void)inputs;
    (void)input_origins;
    (void)output;
    (void)output_origin;
    (void)opts;
#endif
}

}  // namespace tensordiff::core
