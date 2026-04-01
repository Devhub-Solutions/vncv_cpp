#include "vocab.h"

#include <fstream>
#include <stdexcept>

// nlohmann/json – fetched via CMake FetchContent
#include <nlohmann/json.hpp>

namespace vncv {

Vocab::Vocab(const std::string& vocab_json_path)
{
    std::ifstream f(vocab_json_path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open vocab file: " + vocab_json_path);
    }

    nlohmann::json data = nlohmann::json::parse(f);

    auto chars = data["chars"].get<std::vector<std::string>>();
    total_size_ = data["total_vocab_size"].get<int>();

    i2c_[0] = "<pad>";
    i2c_[1] = "<sos>";
    i2c_[2] = "<eos>";
    i2c_[3] = "<mask>";

    for (int i = 0; i < static_cast<int>(chars.size()); ++i) {
        i2c_[i + 4] = chars[i];
    }
}

std::string Vocab::decode(const std::vector<int>& ids) const
{
    // Find <sos> start and <eos> end
    int first = 0;
    for (int i = 0; i < static_cast<int>(ids.size()); ++i) {
        if (ids[i] == SOS) { first = i + 1; break; }
    }

    int last = static_cast<int>(ids.size());
    for (int i = first; i < static_cast<int>(ids.size()); ++i) {
        if (ids[i] == EOS) { last = i; break; }
    }

    std::string result;
    for (int i = first; i < last; ++i) {
        auto it = i2c_.find(ids[i]);
        if (it != i2c_.end()) {
            const std::string& ch = it->second;
            // Skip special tokens
            if (ch != "<pad>" && ch != "<sos>" && ch != "<eos>" && ch != "<mask>") {
                result += ch;
            }
        }
    }
    return result;
}

std::vector<std::string> Vocab::batch_decode(
    const std::vector<std::vector<int>>& arr) const
{
    std::vector<std::string> results;
    results.reserve(arr.size());
    for (const auto& ids : arr) {
        results.push_back(decode(ids));
    }
    return results;
}

} // namespace vncv
