#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

#include "ocr_engine.h"

int main(int argc, char** argv) {
    try {
        const std::string image_path =
            (argc > 1) ? argv[1] : "images/raw/image.png";
        const std::string weights_dir =
            (argc > 2) ? argv[2] : "vncv/weights";
        const std::string lang = (argc > 3) ? argv[3] : "en";

        if (!std::filesystem::exists(image_path)) {
            std::cerr << "Image not found: " << image_path << "\n";
            return 2;
        }
        if (!std::filesystem::is_regular_file(image_path)) {
            std::cerr << "Image path is not a file: " << image_path << "\n";
            return 2;
        }
        if (!std::filesystem::exists(weights_dir)) {
            std::cerr << "Weights directory not found: " << weights_dir << "\n";
            return 2;
        }
        if (!std::filesystem::is_directory(weights_dir)) {
            std::cerr << "Weights path is not a directory: " << weights_dir << "\n";
            return 2;
        }

        vncv::OcrEngine engine(weights_dir);
        const auto results = engine.extract(image_path, lang, false, "");

        std::cout << "SDK smoke test completed. Result count: "
                  << results.size() << "\n";
        if (!results.empty()) {
            std::cout << "First text: " << results.front().text << "\n";
            std::cout << "First confidence: " << results.front().confidence << "\n";
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "SDK smoke test failed: " << ex.what() << "\n";
        return 1;
    }
}
