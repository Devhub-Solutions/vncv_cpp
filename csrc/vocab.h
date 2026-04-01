#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace vncv {

/**
 * Vocabulary loader for VietOCR ONNX inference.
 *
 * Reads a vocab.json produced by the VietOCR export tool.
 * JSON schema:
 *   { "chars": ["a", "b", ...], "total_vocab_size": N }
 *
 * Token indices:
 *   0 = <pad>, 1 = <sos>, 2 = <eos>, 3 = <mask>
 *   4 .. N-1 = characters from the `chars` list
 */
class Vocab {
public:
    static constexpr int PAD  = 0;
    static constexpr int SOS  = 1;
    static constexpr int EOS  = 2;
    static constexpr int MASK = 3;

    explicit Vocab(const std::string& vocab_json_path);

    /** Decode a single sequence of token ids → UTF-8 string. */
    std::string decode(const std::vector<int>& ids) const;

    /** Decode a batch of token id sequences. */
    std::vector<std::string> batch_decode(
        const std::vector<std::vector<int>>& arr) const;

    int size() const { return total_size_; }

private:
    std::unordered_map<int, std::string> i2c_;
    int total_size_{0};
};

} // namespace vncv
