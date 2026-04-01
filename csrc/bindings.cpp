/**
 * pybind11 bindings for the vncv C++ OCR engine.
 *
 * Exposes:
 *   - OcrEngine class  (for re-use across multiple calls)
 *   - extract_text()   (convenience free function using a per-process singleton)
 */

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ocr_engine.h"

namespace py = pybind11;

// ── Process-level singleton engine ──────────────────────────────────────────

static std::unique_ptr<vncv::OcrEngine> g_engine;
static std::string                      g_engine_weights_dir;

static vncv::OcrEngine& get_global_engine(const std::string& weights_dir)
{
    if (!g_engine || g_engine_weights_dir != weights_dir) {
        g_engine = std::make_unique<vncv::OcrEngine>(weights_dir);
        g_engine_weights_dir = weights_dir;
    }
    return *g_engine;
}

// ── Free function matching Python extract_text() signature ──────────────────

static py::object extract_text_cpp(
    const std::string& filepath,
    const std::string& weights_dir,
    bool               save_annotated   = false,
    const std::string& annotated_path   = "",
    const std::string& lang             = "vi",
    bool               return_dict      = false)
{
    auto& engine = get_global_engine(weights_dir);
    auto results = engine.extract(filepath, lang, save_annotated, annotated_path);

    if (return_dict) {
        py::list out;
        for (const auto& r : results) {
            py::dict d;
            d["text"]       = r.text;
            d["confidence"] = r.confidence;
            // box: list of [x, y] pairs
            py::list box;
            for (const auto& pt : r.box) {
                box.append(py::make_tuple(pt[0], pt[1]));
            }
            d["box"] = box;
            out.append(d);
        }
        return out;
    } else {
        py::list out;
        for (const auto& r : results) {
            out.append(r.text);
        }
        return out;
    }
}

// ── Module definition ────────────────────────────────────────────────────────

PYBIND11_MODULE(_vncv_core, m)
{
    m.doc() = "VNCV C++ OCR engine with ONNX Runtime backend";

    // ── OcrEngine class ───────────────────────────────────────────────────
    py::class_<vncv::OcrEngine>(m, "OcrEngine",
        "Unified OCR pipeline: Detection → Classification → Recognition.\n\n"
        "Args:\n"
        "    weights_dir (str): directory containing all .onnx and vocab.json files.\n")
        .def(py::init<const std::string&>(), py::arg("weights_dir"))
        .def("extract", [](vncv::OcrEngine& self,
                           const std::string& filepath,
                           const std::string& lang,
                           bool save_annotated,
                           const std::string& annotated_path,
                           bool return_dict) -> py::object {
                 auto results = self.extract(filepath, lang,
                                             save_annotated, annotated_path);
                 if (return_dict) {
                     py::list out;
                     for (const auto& r : results) {
                         py::dict d;
                         d["text"]       = r.text;
                         d["confidence"] = r.confidence;
                         py::list box;
                         for (const auto& pt : r.box)
                             box.append(py::make_tuple(pt[0], pt[1]));
                         d["box"] = box;
                         out.append(d);
                     }
                     return out;
                 } else {
                     py::list out;
                     for (const auto& r : results) out.append(r.text);
                     return out;
                 }
             },
             py::arg("filepath"),
             py::arg("lang")             = "vi",
             py::arg("save_annotated")   = false,
             py::arg("annotated_path")   = "",
             py::arg("return_dict")      = false,
             "Run the full OCR pipeline on an image file.\n\n"
             "Args:\n"
             "    filepath (str): path to input image.\n"
             "    lang (str): 'vi' for Vietnamese (VietOCR) or 'en' for English (CTC).\n"
             "    save_annotated (bool): write annotated image alongside input.\n"
             "    annotated_path (str): custom path for annotated image.\n"
             "    return_dict (bool): if True return list of dicts with 'text', "
             "'confidence', 'box'; otherwise return list of strings.\n");

    // ── Module-level extract_text convenience function ────────────────────
    m.def("extract_text_cpp", &extract_text_cpp,
          py::arg("filepath"),
          py::arg("weights_dir"),
          py::arg("save_annotated")  = false,
          py::arg("annotated_path")  = "",
          py::arg("lang")            = "vi",
          py::arg("return_dict")     = false,
          "Run OCR using the module-level singleton engine.\n\n"
          "Args:\n"
          "    filepath (str): path to input image.\n"
          "    weights_dir (str): directory containing model weights.\n"
          "    save_annotated (bool): write annotated image.\n"
          "    annotated_path (str): custom annotated image path.\n"
          "    lang (str): 'vi' or 'en'.\n"
          "    return_dict (bool): return structured dicts instead of plain strings.\n");
}
