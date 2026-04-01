#pragma once

#include <string>
#include <utility>
#include <vector>

namespace vncv {

/**
 * CTC beam-search / greedy decoder.
 *
 * Character list matches the Python CTCDecoder exactly.
 * Index 0 is the blank token.
 */
class CTCDecoder {
public:
    CTCDecoder();

    /**
     * Decode a single sample.
     *
     * @param logits  (T, num_classes) float array – raw model output for one item
     * @param T       time steps
     * @return        (text, per-character confidences)
     */
    std::pair<std::string, std::vector<float>>
    decode_single(const float* logits, int T) const;

    int num_classes() const { return static_cast<int>(characters_.size()); }

private:
    std::vector<std::string> characters_;
};

} // namespace vncv
