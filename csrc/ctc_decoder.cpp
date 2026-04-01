#include "ctc_decoder.h"

#include <algorithm>
#include <cmath>

namespace vncv {

CTCDecoder::CTCDecoder()
    : characters_{
          // Index 0 = blank
          "blank",
          "0","1","2","3","4","5","6","7","8","9",
          ":",";","<","=",">","?","@",
          "A","B","C","D","E","F","G","H","I","J","K","L",
          "M","N","O","P","Q","R","S","T","U","V","W","X",
          "Y","Z","[","\\","]","^","_","`",
          "a","b","c","d","e","f","g","h","i","j","k","l",
          "m","n","o","p","q","r","s","t","u","v","w","x",
          "y","z","{","|","}","~","!","\"","#","$","%","&",
          "'","(",")","*","+",",","-",".","/",
          " ", " "   // two spaces at end (matching Python list)
      }
{}

std::pair<std::string, std::vector<float>>
CTCDecoder::decode_single(const float* logits, int T) const
{
    int C = static_cast<int>(characters_.size());

    std::string text;
    std::vector<float> confs;

    int prev_idx = -1;
    for (int t = 0; t < T; ++t) {
        const float* row = logits + t * C;
        int best_idx = static_cast<int>(
            std::max_element(row, row + C) - row);
        float best_val = row[best_idx];

        // Deduplicate consecutive identical tokens, skip blank (idx 0)
        if (best_idx != 0 && best_idx != prev_idx) {
            // Softmax over the row for confidence
            float max_v = best_val;
            float sum   = 0.f;
            for (int c = 0; c < C; ++c) {
                sum += std::exp(row[c] - max_v);
            }
            float conf = std::exp(best_val - max_v) / sum;

            text  += characters_[best_idx];
            confs.push_back(conf);
        }
        prev_idx = best_idx;
    }
    return {text, confs};
}

} // namespace vncv
