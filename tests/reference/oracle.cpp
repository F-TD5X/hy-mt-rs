// Test-only oracle. Not part of the Rust build or runtime.
#include "llama.h"
#include "ggml-impl.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<char> read(const char * path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(std::string("cannot open ") + path);
    return {std::istreambuf_iterator<char>(f), {}};
}

static void ids(const std::string & path, const std::vector<llama_token> & tokens) {
    std::ofstream out(path);
    out << '[';
    for (size_t i = 0; i < tokens.size(); ++i) { if (i) out << ','; out << tokens[i]; }
    out << "]\n";
}

int main(int argc, char ** argv) {
    try {
        if (argc == 5 && std::string(argv[1]) == "--decode") {
            const auto type = static_cast<ggml_type>(std::stoi(argv[2]));
            const auto bytes = read(argv[3]);
            const size_t count = bytes.size() / ggml_type_size(type) * ggml_blck_size(type);
            if (bytes.size() % ggml_type_size(type)) throw std::runtime_error("partial quant block");
            std::vector<float> values(count);
            ggml_get_type_traits(type)->to_float(bytes.data(), values.data(), count);
            std::ofstream output(argv[4], std::ios::binary);
            output.write(reinterpret_cast<const char *>(values.data()), count * sizeof(float));
            return 0;
        }
        if (argc < 4) throw std::runtime_error("usage: oracle MODEL PROMPT_FILE OUTPUT_PREFIX [STEPS] [THREADS]");
        const int steps = argc > 4 ? std::stoi(argv[4]) : 16;
        const int threads = argc > 5 ? std::stoi(argv[5]) : 4;
        const auto input = read(argv[2]);
        const std::string prefix = argv[3];
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        auto * model = llama_model_load_from_file(argv[1], mp);
        if (!model) throw std::runtime_error("model load failed");
        const auto * vocab = llama_model_get_vocab(model);
        std::vector<llama_token> prompt(input.size() + 64);
        auto n = llama_tokenize(vocab, input.data(), input.size(), prompt.data(), prompt.size(), false, true);
        if (n < 0) { prompt.resize(-n); n = llama_tokenize(vocab, input.data(), input.size(), prompt.data(), prompt.size(), false, true); }
        if (n <= 0) throw std::runtime_error("tokenization failed");
        prompt.resize(n);
        ids(prefix + ".prompt.json", prompt);
        auto cp = llama_context_default_params();
        cp.n_ctx = std::max(512, n + steps + 1);
        cp.n_batch = 256;
        cp.n_ubatch = 256;
        cp.n_threads = threads;
        cp.n_threads_batch = threads;
        cp.type_k = GGML_TYPE_F32;
        cp.type_v = GGML_TYPE_F32;
        auto * ctx = llama_init_from_model(model, cp);
        if (!ctx) throw std::runtime_error("context creation failed");
        for (size_t offset = 0; offset < prompt.size(); offset += 256) {
            auto batch = llama_batch_get_one(prompt.data() + offset, std::min<size_t>(256, prompt.size() - offset));
            if (llama_decode(ctx, batch)) throw std::runtime_error("prefill failed");
        }
        const int vocab_size = llama_vocab_n_tokens(vocab);
        {
            std::ofstream out(prefix + ".logits.f32", std::ios::binary);
            out.write(reinterpret_cast<const char *>(llama_get_logits_ith(ctx, -1)), vocab_size * sizeof(float));
        }
        std::vector<llama_token> generated;
        std::ofstream text(prefix + ".text", std::ios::binary);
        for (int step = 0; step < steps; ++step) {
            auto * logits = llama_get_logits_ith(ctx, -1);
            llama_token token = std::max_element(logits, logits + vocab_size) - logits;
            generated.push_back(token);
            if (llama_vocab_is_eog(vocab, token)) break;
            std::vector<char> piece(256);
            int size = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, false);
            if (size < 0) { piece.resize(-size); size = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, false); }
            text.write(piece.data(), size);
            if (step + 1 < steps && llama_decode(ctx, llama_batch_get_one(&token, 1))) throw std::runtime_error("decode failed");
        }
        ids(prefix + ".generated.json", generated);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
