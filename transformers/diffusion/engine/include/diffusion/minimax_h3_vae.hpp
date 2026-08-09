//
//  minimax_h3_vae.hpp
//
//  MiniMax-H3 video VAE decoder: replays the reference decode plan over a fixed-shape tile graph.
//
#ifndef MNN_MINIMAX_H3_VAE_HPP
#define MNN_MINIMAX_H3_VAE_HPP

#include <memory>
#include <string>
#include <vector>

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Module.hpp>

namespace MNN {
namespace DIFFUSION {

// The decoder is a ViT over latent voxels, and the exported graph decodes exactly one spatial tile of one
// temporal clip. Everything around it is index arithmetic that depends only on the latent shape, so the
// exporter computes it once into `h3_vae_plan.json` and this class replays it:
//
//   * temporal chunks that overlap because the encoder's `token_drop` removed each chunk's tail, cross-faded
//     over `frame_overlap` pixel frames;
//   * spatial tiles with their own overlaps, blended the same way;
//   * the trailing pixel frames that padded latent frames produced and that were never requested.
//
// The decoded video is assembled in host memory as float32 RGB in `[0, 1]` after the VAE's ImageNet
// normalization is reverted.
class MNN_PUBLIC MinimaxH3Vae {
public:
    MinimaxH3Vae(std::string resourcePath, MNNForwardType backendType);
    ~MinimaxH3Vae();

    bool load();

    // Decodes `(latentChannels, numLatentFrames, latentHeight, latentWidth)` latents, laid out channel-major,
    // into `(numFrames, height, width, 3)` float32 RGB in `[0, 1]`.
    bool decode(const std::vector<float>& latents, std::vector<float>* rgb, int* numFrames, int* height,
                int* width);

    // Turns the transformer's patchified, normalized latent rows into the `(C, F, H, W)` latents `decode`
    // takes: drops the conditioning rows the denoising loop never wrote, unpatchifies, and denormalizes with
    // the VAE's per-channel statistics.
    bool unpackLatentRows(const std::vector<float>& rows, int numConditionRows, std::vector<float>* latents) const;

    int numFrames() const;
    int height() const;
    int width() const;

private:
    struct Plan;

    // One tile graph call. `latentFrameStart` indexes the (already padded) latent frames.
    bool decodeTile(const std::vector<float>& latents, int latentFrameStart, int latentYStart, int latentXStart,
                    std::vector<float>* tile);

    std::unique_ptr<Plan> mPlan;
    std::shared_ptr<Express::Module> mModule;
    std::shared_ptr<Express::Executor::RuntimeManager> mRuntime;
    Express::VARP mRopeCos;
    Express::VARP mRopeSin;
    Express::VARP mMask;
    std::string mResourcePath;
    MNNForwardType mBackendType;
    BackendConfig mBackendConfig;
};

} // namespace DIFFUSION
} // namespace MNN

#endif // MNN_MINIMAX_H3_VAE_HPP
