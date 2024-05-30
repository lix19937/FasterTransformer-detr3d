/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-14 18:55:58
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-14 18:56:40
 **************************************************************/

#include "SVTransformerPlugin.h"
#include <map>
#include <thread>

using namespace nvinfer1;

const int NUM_LAYERS = 4;
const std::vector<const char*> base_attri{"max_batch",
                                          "max_seq_len",
                                          "seq_len",
                                          "embed_dim",
                                          "num_heads",
                                          "inter_size",
                                          "layer_num",
                                          "num_cam",
                                          "num_reg_points",
                                          "num_classes",
                                          "l2i_matr_h",
                                          "l2i_matr_w",
                                          "img_shape",
                                          "pc_range"};

const std::vector<const char*> pre_layer_weight_names{
    "posembed__in__query", "posembed__in__query_pos", "reg__in__reference_points"};

const std::vector<const char*> ir_param_names{
    "ir__ca__fs__rfpcat", "ir__ca__attention_weights__out_nobias", "ir__ca__out__inp_res_pos_feat"};

const std::vector<const char*> post_layer_weight_names{"cls_branches__fc1__weight",
                                                       "cls_branches__fc1__bias",
                                                       "cls_branches__ln1__weight",
                                                       "cls_branches__ln1__bias",
                                                       "cls_branches__fc2__weight",
                                                       "cls_branches__fc2__bias",
                                                       "cls_branches__ln2__weight",
                                                       "cls_branches__ln2__bias",
                                                       "cls_branches__fc3__weight",
                                                       "cls_branches__fc3__bias"};

// const std::vector<const char*> layer_weight_names_v0{"block_%d__mh_attention__query__weight",
//                                                      "block_%d__mh_attention__query__bias",
//                                                      "block_%d__mh_attention__key__weight",
//                                                      "block_%d__mh_attention__key__bias",
//                                                      "block_%d__mh_attention__value__weight",
//                                                      "block_%d__mh_attention__value__bias",
//                                                      "block_%d__mh_attention__out__weight",
//                                                      "block_%d__mh_attention__out__bias",
//                                                      "block_%d__mh_attention_norm__ln__weight",
//                                                      "block_%d__mh_attention_norm__ln__bias",
//                                                      "block_%d__cross_attention__attention_weights__fc__weight",
//                                                      "block_%d__cross_attention__attention_weights__fc__bias",
//                                                      "block_%d__cross_attention__output_proj__fc__weight",
//                                                      "block_%d__cross_attention__output_proj__fc__bias",
//                                                      "block_%d__cross_attention__position_encoder__fc1__weight",
//                                                      "block_%d__cross_attention__position_encoder__fc1__bias",
//                                                      "block_%d__cross_attention__position_encoder__ln1__weight",
//                                                      "block_%d__cross_attention__position_encoder__ln1__bias",
//                                                      "block_%d__cross_attention__position_encoder__fc2__weight",
//                                                      "block_%d__cross_attention__position_encoder__fc2__bias",
//                                                      "block_%d__cross_attention__position_encoder__ln2__weight",
//                                                      "block_%d__cross_attention__position_encoder__ln2__bias",
//                                                      "block_%d__cross_attention_norm__ln__weight",
//                                                      "block_%d__cross_attention_norm__ln__bias",
//                                                      "block_%d__ffn__fc1__weight",
//                                                      "block_%d__ffn__fc1__bias",
//                                                      "block_%d__ffn__fc2__weight",
//                                                      "block_%d__ffn__fc2__bias",
//                                                      "block_%d__ffn_norm__ln__weight",
//                                                      "block_%d__ffn_norm__ln__bias",
//                                                      "block_%d__reg_branches__fc1__weight",
//                                                      "block_%d__reg_branches__fc1__bias",
//                                                      "block_%d__reg_branches__fc2__weight",
//                                                      "block_%d__reg_branches__fc2__bias",
//                                                      "block_%d__reg_branches__fc3__weight",
//                                                      "block_%d__reg_branches__fc3__bias"};

const std::vector<const char*> layer_weight_names{"block_1__mh_attention__query__weight",
                                                  "block_1__mh_attention__query__bias",
                                                  "block_1__mh_attention__key__weight",
                                                  "block_1__mh_attention__key__bias",
                                                  "block_1__mh_attention__value__weight",
                                                  "block_1__mh_attention__value__bias",
                                                  "block_1__mh_attention__out__weight",
                                                  "block_1__mh_attention__out__bias",
                                                  "block_1__mh_attention_norm__ln__weight",
                                                  "block_1__mh_attention_norm__ln__bias",
                                                  "block_1__cross_attention__attention_weights__fc__weight",
                                                  "block_1__cross_attention__attention_weights__fc__bias",
                                                  "block_1__cross_attention__output_proj__fc__weight",
                                                  "block_1__cross_attention__output_proj__fc__bias",
                                                  "block_1__cross_attention__position_encoder__fc1__weight",
                                                  "block_1__cross_attention__position_encoder__fc1__bias",
                                                  "block_1__cross_attention__position_encoder__ln1__weight",
                                                  "block_1__cross_attention__position_encoder__ln1__bias",
                                                  "block_1__cross_attention__position_encoder__fc2__weight",
                                                  "block_1__cross_attention__position_encoder__fc2__bias",
                                                  "block_1__cross_attention__position_encoder__ln2__weight",
                                                  "block_1__cross_attention__position_encoder__ln2__bias",
                                                  "block_1__cross_attention_norm__ln__weight",
                                                  "block_1__cross_attention_norm__ln__bias",
                                                  "block_1__ffn__fc1__weight",
                                                  "block_1__ffn__fc1__bias",
                                                  "block_1__ffn__fc2__weight",
                                                  "block_1__ffn__fc2__bias",
                                                  "block_1__ffn_norm__ln__weight",
                                                  "block_1__ffn_norm__ln__bias",
                                                  "block_1__reg_branches__fc1__weight",
                                                  "block_1__reg_branches__fc1__bias",
                                                  "block_1__reg_branches__fc2__weight",
                                                  "block_1__reg_branches__fc2__bias",
                                                  "block_1__reg_branches__fc3__weight",
                                                  "block_1__reg_branches__fc3__bias",
                                                  "block_2__mh_attention__query__weight",
                                                  "block_2__mh_attention__query__bias",
                                                  "block_2__mh_attention__key__weight",
                                                  "block_2__mh_attention__key__bias",
                                                  "block_2__mh_attention__value__weight",
                                                  "block_2__mh_attention__value__bias",
                                                  "block_2__mh_attention__out__weight",
                                                  "block_2__mh_attention__out__bias",
                                                  "block_2__mh_attention_norm__ln__weight",
                                                  "block_2__mh_attention_norm__ln__bias",
                                                  "block_2__cross_attention__attention_weights__fc__weight",
                                                  "block_2__cross_attention__attention_weights__fc__bias",
                                                  "block_2__cross_attention__output_proj__fc__weight",
                                                  "block_2__cross_attention__output_proj__fc__bias",
                                                  "block_2__cross_attention__position_encoder__fc1__weight",
                                                  "block_2__cross_attention__position_encoder__fc1__bias",
                                                  "block_2__cross_attention__position_encoder__ln1__weight",
                                                  "block_2__cross_attention__position_encoder__ln1__bias",
                                                  "block_2__cross_attention__position_encoder__fc2__weight",
                                                  "block_2__cross_attention__position_encoder__fc2__bias",
                                                  "block_2__cross_attention__position_encoder__ln2__weight",
                                                  "block_2__cross_attention__position_encoder__ln2__bias",
                                                  "block_2__cross_attention_norm__ln__weight",
                                                  "block_2__cross_attention_norm__ln__bias",
                                                  "block_2__ffn__fc1__weight",
                                                  "block_2__ffn__fc1__bias",
                                                  "block_2__ffn__fc2__weight",
                                                  "block_2__ffn__fc2__bias",
                                                  "block_2__ffn_norm__ln__weight",
                                                  "block_2__ffn_norm__ln__bias",
                                                  "block_2__reg_branches__fc1__weight",
                                                  "block_2__reg_branches__fc1__bias",
                                                  "block_2__reg_branches__fc2__weight",
                                                  "block_2__reg_branches__fc2__bias",
                                                  "block_2__reg_branches__fc3__weight",
                                                  "block_2__reg_branches__fc3__bias",
                                                  "block_3__mh_attention__query__weight",
                                                  "block_3__mh_attention__query__bias",
                                                  "block_3__mh_attention__key__weight",
                                                  "block_3__mh_attention__key__bias",
                                                  "block_3__mh_attention__value__weight",
                                                  "block_3__mh_attention__value__bias",
                                                  "block_3__mh_attention__out__weight",
                                                  "block_3__mh_attention__out__bias",
                                                  "block_3__mh_attention_norm__ln__weight",
                                                  "block_3__mh_attention_norm__ln__bias",
                                                  "block_3__cross_attention__attention_weights__fc__weight",
                                                  "block_3__cross_attention__attention_weights__fc__bias",
                                                  "block_3__cross_attention__output_proj__fc__weight",
                                                  "block_3__cross_attention__output_proj__fc__bias",
                                                  "block_3__cross_attention__position_encoder__fc1__weight",
                                                  "block_3__cross_attention__position_encoder__fc1__bias",
                                                  "block_3__cross_attention__position_encoder__ln1__weight",
                                                  "block_3__cross_attention__position_encoder__ln1__bias",
                                                  "block_3__cross_attention__position_encoder__fc2__weight",
                                                  "block_3__cross_attention__position_encoder__fc2__bias",
                                                  "block_3__cross_attention__position_encoder__ln2__weight",
                                                  "block_3__cross_attention__position_encoder__ln2__bias",
                                                  "block_3__cross_attention_norm__ln__weight",
                                                  "block_3__cross_attention_norm__ln__bias",
                                                  "block_3__ffn__fc1__weight",
                                                  "block_3__ffn__fc1__bias",
                                                  "block_3__ffn__fc2__weight",
                                                  "block_3__ffn__fc2__bias",
                                                  "block_3__ffn_norm__ln__weight",
                                                  "block_3__ffn_norm__ln__bias",
                                                  "block_3__reg_branches__fc1__weight",
                                                  "block_3__reg_branches__fc1__bias",
                                                  "block_3__reg_branches__fc2__weight",
                                                  "block_3__reg_branches__fc2__bias",
                                                  "block_3__reg_branches__fc3__weight",
                                                  "block_3__reg_branches__fc3__bias",
                                                  "block_4__mh_attention__query__weight",
                                                  "block_4__mh_attention__query__bias",
                                                  "block_4__mh_attention__key__weight",
                                                  "block_4__mh_attention__key__bias",
                                                  "block_4__mh_attention__value__weight",
                                                  "block_4__mh_attention__value__bias",
                                                  "block_4__mh_attention__out__weight",
                                                  "block_4__mh_attention__out__bias",
                                                  "block_4__mh_attention_norm__ln__weight",
                                                  "block_4__mh_attention_norm__ln__bias",
                                                  "block_4__cross_attention__attention_weights__fc__weight",
                                                  "block_4__cross_attention__attention_weights__fc__bias",
                                                  "block_4__cross_attention__output_proj__fc__weight",
                                                  "block_4__cross_attention__output_proj__fc__bias",
                                                  "block_4__cross_attention__position_encoder__fc1__weight",
                                                  "block_4__cross_attention__position_encoder__fc1__bias",
                                                  "block_4__cross_attention__position_encoder__ln1__weight",
                                                  "block_4__cross_attention__position_encoder__ln1__bias",
                                                  "block_4__cross_attention__position_encoder__fc2__weight",
                                                  "block_4__cross_attention__position_encoder__fc2__bias",
                                                  "block_4__cross_attention__position_encoder__ln2__weight",
                                                  "block_4__cross_attention__position_encoder__ln2__bias",
                                                  "block_4__cross_attention_norm__ln__weight",
                                                  "block_4__cross_attention_norm__ln__bias",
                                                  "block_4__ffn__fc1__weight",
                                                  "block_4__ffn__fc1__bias",
                                                  "block_4__ffn__fc2__weight",
                                                  "block_4__ffn__fc2__bias",
                                                  "block_4__ffn_norm__ln__weight",
                                                  "block_4__ffn_norm__ln__bias",
                                                  "block_4__reg_branches__fc1__weight",
                                                  "block_4__reg_branches__fc1__bias",
                                                  "block_4__reg_branches__fc2__weight",
                                                  "block_4__reg_branches__fc2__bias",
                                                  "block_4__reg_branches__fc3__weight",
                                                  "block_4__reg_branches__fc3__bias"};

#define SV_PLUGIN_IN_NUM (4)
#define SV_PLUGIN_OUT_NUM (2)

namespace {
static const char* SV_PLUGIN_VERSION{"1"};
static const char* SV_PLUGIN_NAME{"SvTransformerDecoder"};
}  // namespace

namespace fastertransformer {

PluginFieldCollection SVTransformerPluginCreator::mFC{};
std::vector<PluginField> SVTransformerPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(SVTransformerPluginCreator);

SVTransformerPlugin::SVTransformerPlugin(const std::string& name,
                                         const int max_batch,
                                         const int max_seq_len,
                                         const int seq_len,
                                         const int embed_dim,
                                         const int num_heads,
                                         const int inter_size,
                                         const int layer_num,
                                         const int num_cam,
                                         const int num_reg_points,
                                         const int num_classes,
                                         const int l2i_matr_h,
                                         const int l2i_matr_w,
                                         const int* img_shape,
                                         const float* range,
                                         const float q_scaling,
                                         const bool with_cls_token,
                                         const std::vector<const float*>& w):
    layer_name_(name)
{
    settings_.max_batch_size = max_batch;
    settings_.max_seq_len = max_seq_len;
    settings_.seq_len = seq_len;
    settings_.embed_dim = embed_dim;
    settings_.head_num = num_heads;
    settings_.inter_size = inter_size;
    settings_.num_layer = layer_num;
    settings_.num_cam = num_cam;
    settings_.num_reg_points = num_reg_points;
    settings_.num_classes = num_classes;
    settings_.l2i_matr_h = l2i_matr_h;
    settings_.l2i_matr_w = l2i_matr_w;

    for (int i = 0; i < 2; ++i) {
        settings_.img_shape[i] = img_shape[i];
    }
    for (int i = 0; i < num_cam; ++i) {
        settings_.pc_range[i] = range[i];
    }

    with_cls_token_ = with_cls_token;
    int device{-1};
    check_cuda_error(cudaGetDevice(&device));
    cudaDeviceProp props;
    check_cuda_error(cudaGetDeviceProperties(&props, device));
    sm_ = props.major * 10 + props.minor;
    q_scaling_ = q_scaling;

    if (settings_.max_batch_size != 1) {
        APP_PRINTF("max_batch_size must set 1\n");
        exit(0);
    }

    Init(w, *this, nullptr);
}

SVTransformerPlugin::SVTransformerPlugin(const std::string& name, const void* data /* float buff */, size_t length):
    layer_name_(name)
{
    ::memcpy(&settings_, data, sizeof(settings_));
    const char* w_buffer = static_cast<const char*>(data) + sizeof(settings_);

    std::vector<const float*> dummy;
    Init(dummy, *this, w_buffer);
}

SVTransformerPlugin::SVTransformerPlugin(const SVTransformerPlugin& plugin): layer_name_(plugin.layer_name_)
{
    ::memcpy(&settings_, &plugin.settings_, sizeof(plugin.settings_));
    std::vector<const float*> dummy;
    Init(dummy, plugin, nullptr);

    with_cls_token_ = plugin.with_cls_token_;
    sm_ = plugin.sm_;
    q_scaling_ = plugin.q_scaling_;
    attention_type_ = plugin.attention_type_;
}

void SVTransformerPlugin::Init(const std::vector<const float*>& w, const SVTransformerPlugin& plugin, const char* data)
{
    params_ = new SVWeight<half>(settings_.embed_dim,
                                 settings_.inter_size,
                                 settings_.num_layer,
                                 settings_.seq_len,
                                 settings_.num_classes,
                                 with_cls_token_);

    if (!w.empty()) {
        auto weight_num = params_->GetWeightCount();
        if (weight_num != w.size()) {
            APP_PRINTF("[ERROR][SVTransformerPlugin] weights number %lu does not match expected number %lu!\n",
                       w.size(),
                       weight_num);
            exit(-1);
        }
        const float* const* pp_buf = &w[0];
        params_->CopyWeightsFromHostBuffersFp32ToDeviceHalf(pp_buf);
    }

    if (data != nullptr) {
        params_->deserialize(data);
    }

    if (w.empty() && data == nullptr) {
        *params_ = *plugin.params_;
    }

    ////////////////////////////////////////////////////////////////////// fp //////
    {
        fp_params_ = new SVWeight<float>(settings_.embed_dim,
                                         settings_.inter_size,
                                         settings_.num_layer,
                                         settings_.seq_len,
                                         settings_.num_classes,
                                         with_cls_token_);

        if (!w.empty()) {
            auto weight_num = fp_params_->GetWeightCount();
            if (weight_num != w.size()) {
                APP_PRINTF("[ERROR][SVTransformerPlugin] weights number %lu does not match expected number %lu!\n",
                           w.size(),
                           weight_num);
                exit(-1);
            }
            const float* const* pp_buf = &w[0];
            fp_params_->CopyWeightsFromHostBuffers(pp_buf);
        }

        if (data != nullptr) {
            fp_params_->deserialize(data);
        }

        if (w.empty() && data == nullptr) {
            *fp_params_ = *plugin.fp_params_;
        }
    }

    check_cuda_error(cublasCreate(&cublas_handle_));
    check_cuda_error(cublasLtCreate(&cublaslt_handle_));

    const std::vector<std::string> choices{"1 512 8 32 1 ### 1 256 512 256 6 0 18 0 1 0 0 12 0.008948",
                                           "1 512 8 32 1 ### 1 512 24 4 23 0 5 3 0 1 0 7 0.005013",
                                           "1 512 8 32 1 ### 1 24 512 256 21 0 11 1 0 0 0 13 0.006933",
                                           "1 512 8 32 1 ### 1 256 512 3 0 0 14 0 0 0 0 0 0.007407",
                                           "1 512 8 32 1 ### 1 512 512 256 6 0 20 1 1 0 0 11 0.011434",
                                           "1 512 8 32 1 ### 1 256 512 512 6 0 18 0 0 0 0 12 0.011358",
                                           "1 512 8 32 1 ### 1 8 512 256 21 0 5 1 0 0 0 20 0.006328",
                                           "1 512 8 32 1 ### 1 5 512 256 24 0 11 0 0 0 0 20 0.007329",
                                           "1 512 8 32 1 ### 8 512 512 32 113 -1 -1 -1 -1 -1 -1 -1 0.030930",
                                           "1 512 8 32 1 ### 8 32 512 512 104 -1 -1 -1 -1 -1 -1 -1 0.025400"};
    cublasAlgoMap_ = new cublasAlgoMap(choices);
    cublasWrapperMutex_ = new std::mutex();
    int current_dev_id = 0;
    check_cuda_error(cudaGetDevice(&current_dev_id));
    allocator_ = new Allocator<AllocatorType::CUDA>(current_dev_id);

    cublas_wrapper_ =
        new cublasMMWrapper(cublas_handle_, cublaslt_handle_, nullptr, cublasAlgoMap_, cublasWrapperMutex_, allocator_);

    sv_transformer_ = new SVTransformer<half>(settings_.max_batch_size,
                                              settings_.seq_len, /* be careful here */
                                              settings_.embed_dim,
                                              settings_.head_num,
                                              settings_.inter_size,
                                              settings_.num_layer,
                                              settings_.num_cam,
                                              settings_.l2i_matr_h,
                                              settings_.l2i_matr_w,
                                              settings_.img_shape,
                                              settings_.pc_range,
                                              with_cls_token_,
                                              sm_,
                                              q_scaling_,
                                              params_,
                                              0,
                                              cublas_wrapper_,
                                              allocator_,
                                              false,
                                              attention_type_);

    ////////////////////////////////////////////////////////////////////// fp //////
    {
        check_cuda_error(cublasCreate(&fp_cublas_handle_));
        check_cuda_error(cublasLtCreate(&fp_cublaslt_handle_));

        const std::vector<std::string> choices{};
        fp_cublasAlgoMap_ = new cublasAlgoMap(choices);
        fp_cublasWrapperMutex_ = new std::mutex();
        int current_dev_id = 0;
        check_cuda_error(cudaGetDevice(&current_dev_id));
        fp_allocator_ = new Allocator<AllocatorType::CUDA>(current_dev_id);

        fp_cublas_wrapper_ = new cublasMMWrapper(
            fp_cublas_handle_, fp_cublaslt_handle_, nullptr, fp_cublasAlgoMap_, fp_cublasWrapperMutex_, fp_allocator_);
        fp_sv_transformer_ = new SVTransformer<float>(settings_.max_batch_size,
                                                      settings_.seq_len, /* be careful here */
                                                      settings_.embed_dim,
                                                      settings_.head_num,
                                                      settings_.inter_size,
                                                      settings_.num_layer,
                                                      settings_.num_cam,
                                                      settings_.l2i_matr_h,
                                                      settings_.l2i_matr_w,
                                                      settings_.img_shape,
                                                      settings_.pc_range,
                                                      with_cls_token_,
                                                      sm_,
                                                      q_scaling_,
                                                      fp_params_,
                                                      0,
                                                      fp_cublas_wrapper_,
                                                      fp_allocator_,
                                                      false,
                                                      attention_type_);
    }
}

SVTransformerPlugin::~SVTransformerPlugin() noexcept
{
    check_cuda_error(cublasDestroy(cublas_handle_));
    check_cuda_error(cublasLtDestroy(cublaslt_handle_));

    delete cublasWrapperMutex_;
    delete cublasAlgoMap_;
    delete sv_transformer_;
    delete allocator_;
    delete params_;

    ///////////////////////////////////////////////////////////////////  fp //////
    check_cuda_error(cublasDestroy(fp_cublas_handle_));
    check_cuda_error(cublasLtDestroy(fp_cublaslt_handle_));

    delete fp_cublasWrapperMutex_;
    delete fp_cublasAlgoMap_;
    delete fp_sv_transformer_;
    delete fp_allocator_;
    delete fp_params_;
}

nvinfer1::IPluginV2DynamicExt* SVTransformerPlugin::clone() const noexcept
{
    try {
        SVTransformerPlugin* ret = new SVTransformerPlugin(*this);
        ret->setPluginNamespace(namespace_.c_str());

        ret->with_cls_token_ = with_cls_token_;
        ret->sm_ = sm_;
        ret->q_scaling_ = q_scaling_;
        ret->attention_type_ = attention_type_;
        return ret;
    }
    catch (const std::exception& e) {
        APP_PRINTF("exception: %s\n", e.what());
    }
    return nullptr;
}

void SVTransformerPlugin::attachToContext(cudnnContext* cudnnContext,
                                          cublasContext* cublasContext,
                                          nvinfer1::IGpuAllocator* gpuAllocator) noexcept
{
}

void SVTransformerPlugin::detachFromContext() noexcept {}

DimsExprs SVTransformerPlugin::getOutputDimensions(int outputIndex,
                                                   const DimsExprs* inputs,
                                                   int nbInputs,
                                                   IExprBuilder& exprBuilder) noexcept
{
    assert(outputIndex < SV_PLUGIN_OUT_NUM);
    DimsExprs output;
    output.nbDims = 3;
    if (settings_.max_batch_size != 1) {
        APP_PRINTF("max_batch_size must set 1\n");
        exit(0);
    }
    if (outputIndex == 0) {
        output.d[0] = exprBuilder.constant(settings_.max_batch_size);
        output.d[1] = exprBuilder.constant(settings_.seq_len);
        output.d[2] = exprBuilder.constant(settings_.num_reg_points);
    }
    else {
        output.d[0] = exprBuilder.constant(settings_.max_batch_size);
        output.d[1] = exprBuilder.constant(settings_.seq_len);
        output.d[2] = exprBuilder.constant(settings_.num_classes);
    }
    return output;
}

std::map<nvinfer1::TensorFormat, const char*> format_map{{nvinfer1::TensorFormat::kLINEAR, "kLINEAR"},
                                                         {nvinfer1::TensorFormat::kCHW2, "kCHW2"},
                                                         {nvinfer1::TensorFormat::kHWC8, "kHWC8"},
                                                         {nvinfer1::TensorFormat::kCHW4, "kCHW4"},
                                                         {nvinfer1::TensorFormat::kCHW16, "kCHW16"},
                                                         {nvinfer1::TensorFormat::kCHW32, "kCHW32"},
                                                         {nvinfer1::TensorFormat::kDHWC8, "kDHWC8"},
                                                         {nvinfer1::TensorFormat::kCDHW32, "kCDHW32"},
                                                         {nvinfer1::TensorFormat::kHWC, "kHWC"},
                                                         {nvinfer1::TensorFormat::kDLA_LINEAR, "kDLA_LINEAR"},
                                                         {nvinfer1::TensorFormat::kDLA_HWC4, "kDLA_HWC4"},
                                                         {nvinfer1::TensorFormat::kHWC16, "kHWC16"}};
std::map<nvinfer1::DataType, const char*> type_map{{nvinfer1::DataType::kFLOAT, "kFLOAT"},
                                                   {nvinfer1::DataType::kHALF, "kHALF"},
                                                   {nvinfer1::DataType::kINT8, "kINT8"},
                                                   {nvinfer1::DataType::kINT32, "kINT32"},
                                                   {nvinfer1::DataType::kBOOL, "kBOOL"}};

bool SVTransformerPlugin::supportsFormatCombination(int pos,
                                                    const PluginTensorDesc* inOut,
                                                    int nbInputs,
                                                    int nbOutputs) noexcept
{
    bool res{false};
    assert(pos >= 0 && pos < SV_PLUGIN_IN_NUM + SV_PLUGIN_OUT_NUM);
    /// follow set not work
    // if (pos >= 0 && pos < 3) {
    //     bool cho1 =
    //            (inOut[0].type == nvinfer1::DataType::kINT8 && inOut[0].format == nvinfer1::TensorFormat::kLINEAR)
    //         && (inOut[1].type == nvinfer1::DataType::kHALF && inOut[1].format == nvinfer1::TensorFormat::kLINEAR)
    //         && (inOut[2].type == nvinfer1::DataType::kHALF && inOut[2].format == nvinfer1::TensorFormat::kLINEAR);

    //     bool cho2 =
    //            (inOut[0].type == nvinfer1::DataType::kHALF && inOut[0].format == nvinfer1::TensorFormat::kLINEAR)
    //         && (inOut[1].type == nvinfer1::DataType::kHALF && inOut[1].format == nvinfer1::TensorFormat::kLINEAR)
    //         && (inOut[2].type == nvinfer1::DataType::kHALF && inOut[2].format == nvinfer1::TensorFormat::kLINEAR);

    //     bool cho3 =
    //            (inOut[0].type == nvinfer1::DataType::kFLOAT && inOut[0].format == nvinfer1::TensorFormat::kLINEAR)
    //         && (inOut[1].type == nvinfer1::DataType::kFLOAT && inOut[1].format == nvinfer1::TensorFormat::kLINEAR)
    //         && (inOut[2].type == nvinfer1::DataType::kFLOAT && inOut[2].format == nvinfer1::TensorFormat::kLINEAR);

    //     return cho1 || cho2 || cho3;
    // }

    switch (pos) {
        case 0:
        case 1:
        case 2:
            res = (inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == nvinfer1::TensorFormat::kCHW32);// || 
                 // (inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
            break;

        ////////////////////////////////////////////////////////// fixed follow //////////
        case 3:
            res = inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
            break;

        case 4:
        case 5:
            res = inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
            break;
        default:
            break;
    }

    return res;
}

void SVTransformerPlugin::configurePlugin(const DynamicPluginTensorDesc* in,
                                          int nbInputs,
                                          const DynamicPluginTensorDesc* out,
                                          int nbOutputs) noexcept
{
}

size_t SVTransformerPlugin::getWorkspaceSize(const PluginTensorDesc* inputs,
                                             int nbInputs,
                                             const PluginTensorDesc* outputs,
                                             int nbOutputs) const noexcept
{
    return 0;
}

int SVTransformerPlugin::enqueue(const PluginTensorDesc* inputDesc,
                                 const PluginTensorDesc* outputDesc,
                                 const void* const* inputs,
                                 void* const* outputs,
                                 void* workspace,
                                 cudaStream_t stream) noexcept
{
    size_t batch_size = 1;
    assert(batch_size <= settings_.max_batch_size);
    assert(settings_.seq_len == outputDesc[0].dims.d[1]); /* [1, seq_len, bbox_point] */
    assert(settings_.seq_len == outputDesc[1].dims.d[1]); /* [1, seq_len, num_class] */
    assert(settings_.num_reg_points == outputDesc[0].dims.d[2]);
    assert(settings_.num_classes == outputDesc[1].dims.d[2]);
    assert(settings_.num_cam == inputDesc[0].dims.d[0]); /* [nc, ch, h, w] */
    size_t ch = inputDesc[0].dims.d[1];

    check_cuda_error(cublasSetStream(cublas_handle_, stream));

    if (inputDesc[0].type == nvinfer1::DataType::kINT8 && inputDesc[1].type == nvinfer1::DataType::kINT8
        && inputDesc[2].type == nvinfer1::DataType::kINT8) {
        typedef int8_t DT;
        APP_PRINTF("enter int8 branch ... %s %s %s %s %s %s %f %f %f\n",
                   type_map[inputDesc[0].type],
                   type_map[inputDesc[1].type],
                   type_map[inputDesc[2].type],
                   format_map[inputDesc[0].format],
                   format_map[inputDesc[1].format],
                   format_map[inputDesc[2].format],
                   inputDesc[0].scale,
                   inputDesc[1].scale,
                   inputDesc[2].scale);

        std::vector<Tensor> input_tensors{
            Tensor{MEMORY_GPU,
                   getTensorType<DT>(),
                   {settings_.num_cam, ch, size_t(inputDesc[0].dims.d[2]), size_t(inputDesc[0].dims.d[3])},
                   (const DT*)(inputs[0]),
                   inputDesc[0].scale},
            Tensor{MEMORY_GPU,
                   getTensorType<DT>(),
                   {settings_.num_cam, ch, size_t(inputDesc[1].dims.d[2]), size_t(inputDesc[1].dims.d[3])},
                   (const DT*)(inputs[1]),
                   inputDesc[1].scale},
            Tensor{MEMORY_GPU,
                   getTensorType<DT>(),
                   {settings_.num_cam, ch, size_t(inputDesc[2].dims.d[2]), size_t(inputDesc[2].dims.d[3])},
                   (const DT*)(inputs[2]),
                   inputDesc[2].scale},

            Tensor{MEMORY_GPU,
                   getTensorType<float>(),
                   {settings_.num_cam, settings_.l2i_matr_h, settings_.l2i_matr_w},
                   (const float*)(inputs[3])} /* l2i_matrix */};

        std::vector<Tensor> output_tensors{Tensor{MEMORY_GPU,
                                                  getTensorType<float>(),
                                                  {batch_size, settings_.seq_len, settings_.num_reg_points},
                                                  (float*)(outputs[0])},
                                           Tensor{MEMORY_GPU,
                                                  getTensorType<float>(),
                                                  {batch_size, settings_.seq_len, settings_.num_classes},
                                                  (float*)(outputs[1])}};

        sv_transformer_->forward(&output_tensors, &input_tensors, stream);
    }
    else if (inputDesc[0].type == nvinfer1::DataType::kINT8 && inputDesc[1].type == nvinfer1::DataType::kHALF
             && inputDesc[2].type == nvinfer1::DataType::kHALF) {
        typedef int8_t DT;
        APP_PRINTF("enter int8 v2 branch ... %s %s %s %s %s %s %f %f %f\n",
                   type_map[inputDesc[0].type],
                   type_map[inputDesc[1].type],
                   type_map[inputDesc[2].type],
                   format_map[inputDesc[0].format],
                   format_map[inputDesc[1].format],
                   format_map[inputDesc[2].format],
                   inputDesc[0].scale,
                   inputDesc[1].scale,
                   inputDesc[2].scale);

        std::vector<Tensor> input_tensors{
            Tensor{MEMORY_GPU,
                   getTensorType<DT>(),
                   {settings_.num_cam, ch, size_t(inputDesc[0].dims.d[2]), size_t(inputDesc[0].dims.d[3])},
                   (const DT*)(inputs[0]),
                   inputDesc[0].scale},
            Tensor{MEMORY_GPU,
                   getTensorType<half>(),
                   {settings_.num_cam, ch, size_t(inputDesc[1].dims.d[2]), size_t(inputDesc[1].dims.d[3])},
                   (const half*)(inputs[1]),
                   inputDesc[1].scale},
            Tensor{MEMORY_GPU,
                   getTensorType<half>(),
                   {settings_.num_cam, ch, size_t(inputDesc[2].dims.d[2]), size_t(inputDesc[2].dims.d[3])},
                   (const half*)(inputs[2]),
                   inputDesc[2].scale},

            Tensor{MEMORY_GPU,
                   getTensorType<float>(),
                   {settings_.num_cam, settings_.l2i_matr_h, settings_.l2i_matr_w},
                   (const float*)(inputs[3])} /* l2i_matrix */};

        std::vector<Tensor> output_tensors{Tensor{MEMORY_GPU,
                                                  getTensorType<float>(),
                                                  {batch_size, settings_.seq_len, settings_.num_reg_points},
                                                  (float*)(outputs[0])},
                                           Tensor{MEMORY_GPU,
                                                  getTensorType<float>(),
                                                  {batch_size, settings_.seq_len, settings_.num_classes},
                                                  (float*)(outputs[1])}};

        sv_transformer_->forward(&output_tensors, &input_tensors, stream);
    }
    else if (inputDesc[0].type == nvinfer1::DataType::kHALF) {
        typedef half DT;
        APP_PRINTF("enter fp16 branch ... %s %s %s %s %s %s\n",
                   type_map[inputDesc[0].type],
                   type_map[inputDesc[1].type],
                   type_map[inputDesc[2].type],
                   format_map[inputDesc[0].format],
                   format_map[inputDesc[1].format],
                   format_map[inputDesc[2].format]);

        /// just for debug
        // std::string base_path =
        //     "/home/igs/transformer/FasterTransformer-main/tf_decoder_node_data_model_cnn_in_out/cnns_chw32_from_plug/";

        // cuda_ops::WriteFromDptr(base_path + "ca.in.mvalue_chw32_cnn_48", {6, 256, 48, 184}, (const DT*)(inputs[0]));
        // cuda_ops::WriteFromDptr(base_path + "ca.in.mvalue_chw32_cnn_24", {6, 256, 24, 92}, (const DT*)(inputs[1]));
        // cuda_ops::WriteFromDptr(base_path + "ca.in.mvalue_chw32_cnn_12", {6, 256, 12, 46}, (const DT*)(inputs[2]));
        // exit(0);

        // if (0) {
        //     float max = -65536.f, min = 65536.f;
        //     std::vector<DT> _1179out(6 * 256 * 48 * 184);
        //     cudaD2Hcpy<DT>(_1179out.data(), (const DT*)(inputs[0]), _1179out.size());
        //     for (size_t i = 0; i < _1179out.size(); ++i) {
        //         auto t = float(_1179out[i]);
        //         if (t > max)
        //             max = t;
        //         if (t < min)
        //             min = t;
        //     }
        //     printf("max: %.6f, min: %.6f\n", max, min);
        //     exit(0);
        // }

        std::vector<Tensor> input_tensors{
            Tensor{MEMORY_GPU,
                   getTensorType<DT>(),
                   {settings_.num_cam, ch, size_t(inputDesc[0].dims.d[2]), size_t(inputDesc[0].dims.d[3])},
                   (const DT*)(inputs[0])},
            Tensor{MEMORY_GPU,
                   getTensorType<DT>(),
                   {settings_.num_cam, ch, size_t(inputDesc[1].dims.d[2]), size_t(inputDesc[1].dims.d[3])},
                   (const DT*)(inputs[1])},
            Tensor{MEMORY_GPU,
                   getTensorType<DT>(),
                   {settings_.num_cam, ch, size_t(inputDesc[2].dims.d[2]), size_t(inputDesc[2].dims.d[3])},
                   (const DT*)(inputs[2])},

            Tensor{MEMORY_GPU,
                   getTensorType<float>(),
                   {settings_.num_cam, settings_.l2i_matr_h, settings_.l2i_matr_w},
                   (const float*)(inputs[3])} /* l2i_matrix */};

        std::vector<Tensor> output_tensors{Tensor{MEMORY_GPU,
                                                  getTensorType<float>(),
                                                  {batch_size, settings_.seq_len, settings_.num_reg_points},
                                                  (float*)(outputs[0])},
                                           Tensor{MEMORY_GPU,
                                                  getTensorType<float>(),
                                                  {batch_size, settings_.seq_len, settings_.num_classes},
                                                  (float*)(outputs[1])}};

        sv_transformer_->forward(&output_tensors, &input_tensors, stream);
    }
    else if (inputDesc[0].type == nvinfer1::DataType::kFLOAT && inputDesc[1].type == nvinfer1::DataType::kFLOAT
             && inputDesc[2].type == nvinfer1::DataType::kFLOAT) {
        typedef float DT;
        APP_PRINTF("enter fp32 branch ... %s %s %s %s %s %s\n",
                   type_map[inputDesc[0].type],
                   type_map[inputDesc[1].type],
                   type_map[inputDesc[2].type],
                   format_map[inputDesc[0].format],
                   format_map[inputDesc[1].format],
                   format_map[inputDesc[2].format]);
        // std::string base_path =
        //     "/home/igs/workspace/TensorRT-release-8.2/tools/Polygraphy/examples/api/05_using_tensorrt_network_api/";
        // cuda_ops::WriteFromDptr(base_path + "plugin_cnn_48", {6, 256, 48, 184}, (const DT*)(inputs[0]));
        // cuda_ops::WriteFromDptr(base_path + "plugin_cnn_24", {6, 256, 24, 92}, (const DT*)(inputs[1]));
        // cuda_ops::WriteFromDptr(base_path + "plugin_cnn_12", {6, 256, 24 / 2, 92 / 2}, (const DT*)(inputs[2]));

        std::vector<Tensor> input_tensors{
            Tensor{MEMORY_GPU,
                   getTensorType<DT>(),
                   {settings_.num_cam, ch, size_t(inputDesc[0].dims.d[2]), size_t(inputDesc[0].dims.d[3])},
                   (const DT*)(inputs[0])},
            Tensor{MEMORY_GPU,
                   getTensorType<DT>(),
                   {settings_.num_cam, ch, size_t(inputDesc[1].dims.d[2]), size_t(inputDesc[1].dims.d[3])},
                   (const DT*)(inputs[1])},
            Tensor{MEMORY_GPU,
                   getTensorType<DT>(),
                   {settings_.num_cam, ch, size_t(inputDesc[2].dims.d[2]), size_t(inputDesc[2].dims.d[3])},
                   (const DT*)(inputs[2])},

            Tensor{MEMORY_GPU,
                   getTensorType<float>(),
                   {settings_.num_cam, settings_.l2i_matr_h, settings_.l2i_matr_w},
                   (const float*)(inputs[3])} /* l2i_matrix */
        };

        std::vector<Tensor> output_tensors{Tensor{MEMORY_GPU,
                                                  getTensorType<float>(),
                                                  {batch_size, settings_.seq_len, settings_.num_reg_points},
                                                  (float*)(outputs[0])},
                                           Tensor{MEMORY_GPU,
                                                  getTensorType<float>(),
                                                  {batch_size, settings_.seq_len, settings_.num_classes},
                                                  (float*)(outputs[1])}};

        fp_sv_transformer_->forward(&output_tensors, &input_tensors, stream);
    }
    else {
        /// for debug
        APP_PRINTF("enter unknown branch ... %s %s %s %s %s %s\n",
                   type_map[inputDesc[0].type],
                   type_map[inputDesc[1].type],
                   type_map[inputDesc[2].type],
                   format_map[inputDesc[0].format],
                   format_map[inputDesc[1].format],
                   format_map[inputDesc[2].format]);
        std::this_thread::sleep_for(std::chrono::milliseconds(256));
        check_cuda_error(cudaMemsetAsync(
            (float*)(outputs[0]), 0, batch_size * settings_.seq_len * settings_.num_reg_points, stream));
        check_cuda_error(cudaMemsetAsync(
            (float*)(outputs[1]), 0, batch_size * settings_.seq_len * settings_.num_classes, stream));

        //////////////////////////////////////////////////////////////////////////////////
        // APP_PRINTF("not support, inOut[0].type %s  %s\n", type_map[inputDesc[0].type],
        // format_map[inputDesc[0].format]); return 1;
    }

    return 0;
}

nvinfer1::DataType
SVTransformerPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    assert(index < SV_PLUGIN_OUT_NUM);
    assert(nbInputs == SV_PLUGIN_IN_NUM);
    return nvinfer1::DataType::kFLOAT;
}

const char* SVTransformerPlugin::getPluginType() const noexcept
{
    return SV_PLUGIN_NAME;
}

const char* SVTransformerPlugin::getPluginVersion() const noexcept
{
    return SV_PLUGIN_VERSION;
}

int SVTransformerPlugin::getNbOutputs() const noexcept
{
    return SV_PLUGIN_OUT_NUM;
}

int SVTransformerPlugin::initialize() noexcept
{
    return 0;
}

void SVTransformerPlugin::terminate() noexcept {}

size_t SVTransformerPlugin::getSerializationSize() const noexcept
{
    return sizeof(int) + sizeof(settings_) + params_->GetSerializeSize();
}

void SVTransformerPlugin::serialize(void* buffer) const noexcept
{
    int type_id = 0;
    ::memcpy(buffer, &type_id, sizeof(type_id));
    char* serial_buffer = (char*)buffer + sizeof(type_id);

    ::memcpy(serial_buffer, &settings_, sizeof(settings_));
    serial_buffer += sizeof(settings_);

    params_->serialize(serial_buffer);

    ////////////////////////////////////////////////////////////// fp /////////
    fp_params_->serialize(serial_buffer);
}

void SVTransformerPlugin::destroy() noexcept
{
    delete this;
}

void SVTransformerPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    namespace_ = libNamespace;
}

const char* SVTransformerPlugin::getPluginNamespace() const noexcept
{
    return namespace_.c_str();
}

////////////////////////////////////////////////////////////////////////////////////////
SVTransformerPluginCreator::SVTransformerPluginCreator()
{
    setbuf(stdout, NULL);

    mPluginAttributes.clear();
    APP_PRINTF("base_attri %zd \n", base_attri.size());
    APP_PRINTF("pre_layer_weight_names %zd \n", pre_layer_weight_names.size());
    APP_PRINTF("ir_param_names %zd \n", ir_param_names.size());
    APP_PRINTF("NUM_LAYERS %d \n", NUM_LAYERS);
    APP_PRINTF("layer_weight_names %zd \n", layer_weight_names.size());
    APP_PRINTF("post_layer_weight_names %zd \n", post_layer_weight_names.size());

    for (const auto& it : base_attri) {
        if (strcmp(it, "pc_range") == 0) {
            mPluginAttributes.emplace_back(PluginField(it, nullptr, PluginFieldType::kFLOAT32, 1));
        }
        else {
            mPluginAttributes.emplace_back(PluginField(it, nullptr, PluginFieldType::kINT32, 1));
        }
    }

    for (const auto& it : pre_layer_weight_names) {
        mPluginAttributes.emplace_back(PluginField(it, nullptr, PluginFieldType::kFLOAT32, 1));
    }
    for (const auto& it : ir_param_names) {
        mPluginAttributes.emplace_back(PluginField(it, nullptr, PluginFieldType::kFLOAT32, 1));
    }
    for (const auto& it : post_layer_weight_names) {
        mPluginAttributes.emplace_back(PluginField(it, nullptr, PluginFieldType::kFLOAT32, 1));
    }
    for (const auto& it : layer_weight_names) {
        mPluginAttributes.emplace_back(PluginField(it, nullptr, PluginFieldType::kFLOAT32, 1));
    }

    APP_PRINTF("after all emplace_back, fc nbFields:%zd\n", mPluginAttributes.size());

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* SVTransformerPluginCreator::getPluginName() const noexcept
{
    return SV_PLUGIN_NAME;
}

const char* SVTransformerPluginCreator::getPluginVersion() const noexcept
{
    return SV_PLUGIN_VERSION;
}

const PluginFieldCollection* SVTransformerPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

void loadWeightsPtr(std::vector<const float*>& w,
                    const nvinfer1::PluginFieldCollection* fc,
                    int layer_num,
                    bool with_cls_token = true)
{
    int idx = 0;
    for (const auto& name : pre_layer_weight_names) {
        for (int i = 0; i < fc->nbFields; ++i) {
            std::string field_name(fc->fields[i].name);
            if (field_name.compare(name) == 0) {
                w[idx++] = (const float*)fc->fields[i].data;
            }
        }
    }
    APP_PRINTF("after pre_layer_weight_names idx:%d\n", idx);

    for (const auto& name : ir_param_names) {
        for (int i = 0; i < fc->nbFields; ++i) {
            std::string field_name(fc->fields[i].name);
            if (field_name.compare(name) == 0) {
                w[idx++] = (const float*)fc->fields[i].data;
            }
        }
    }
    APP_PRINTF("after ir_param_names idx:%d\n", idx);

    for (const auto& name : post_layer_weight_names) {
        for (int i = 0; i < fc->nbFields; ++i) {
            std::string field_name(fc->fields[i].name);
            if (field_name.compare(name) == 0) {
                w[idx++] = (const float*)fc->fields[i].data;
            }
        }
    }
    APP_PRINTF("after post_layer_weight_names idx:%d\n", idx);

    for (const auto& name : layer_weight_names) {
        for (int i = 0; i < fc->nbFields; ++i) {
            std::string field_name(fc->fields[i].name);
            if (field_name.compare(name) == 0) {
                w[idx++] = (const float*)fc->fields[i].data;
            }
        }
    }
    APP_PRINTF("after layer_weight_names idx:%d  total weights size:%zd\n", idx, w.size());

    FT_CHECK(idx == int(w.size()));
}

IPluginV2* SVTransformerPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    APP_PRINTF("createPlugin fc->nbFields:%d\n", fc->nbFields);

    int max_batch;
    int max_seq_len;
    int seq_len;
    int embed_dim;
    int num_heads;
    int inter_size;
    int layer_num;
    int num_cam;
    int num_reg_points;
    int num_classes;
    int l2i_matr_h;
    int l2i_matr_w;
    int img_shape[2];
    float pc_range[6];
    int with_cls_token = true;

    std::map<std::string, int*> name2pint = {{"max_batch", &max_batch},
                                             {"max_seq_len", &max_seq_len},
                                             {"seq_len", &seq_len},
                                             {"embed_dim", &embed_dim},
                                             {"num_heads", &num_heads},
                                             {"inter_size", &inter_size},
                                             {"layer_num", &layer_num},
                                             {"num_cam", &num_cam},
                                             {"num_reg_points", &num_reg_points},
                                             {"num_classes", &num_classes},
                                             {"l2i_matr_h", &l2i_matr_h},
                                             {"l2i_matr_w", &l2i_matr_w},
                                             {"img_shape", &img_shape[0]}};

    std::map<std::string, float*> name2pfp32 = {{"pc_range", &pc_range[0]}};

    for (int i = 0; i < fc->nbFields; ++i) {
        auto jter = name2pfp32.find(fc->fields[i].name);
        if (jter != name2pfp32.end()) {
            memcpy(jter->second, fc->fields[i].data, 6 * sizeof(float));  /// be careful
            APP_PRINTF("name=[%s]\n", jter->first.c_str());
            for (int k = 0; k < 6; ++k) {
                APP_PRINTF("value=[%f]\n", jter->second[k]);
            }
            continue;
        }

        auto iter = name2pint.find(fc->fields[i].name);
        if (iter != name2pint.end()) {
            if (iter->first == "img_shape") {
                memcpy(iter->second, fc->fields[i].data, 2 * sizeof(int));  /// be careful
                APP_PRINTF("name=[%s]\n", iter->first.c_str());
                for (int k = 0; k < 2; ++k) {
                    APP_PRINTF("value=[%d]\n", iter->second[k]);
                }
                continue;
            }
            *(iter->second) = *((int*)fc->fields[i].data);
            APP_PRINTF("name=[%s], value=[%d]\n", iter->first.c_str(), *((int*)fc->fields[i].data));
        }
    }

    size_t weights_num = pre_layer_weight_names.size() + ir_param_names.size() + post_layer_weight_names.size()
                         + layer_weight_names.size();
    APP_PRINTF("pre_layer_weight_names:%zd\n", pre_layer_weight_names.size());
    APP_PRINTF("ir_param_names:%zd\n", ir_param_names.size());
    APP_PRINTF("post_layer_weight_names:%zd\n", post_layer_weight_names.size());
    APP_PRINTF("layer_num:%d\n", layer_num);
    APP_PRINTF("layer_weight_names:%zd\n", layer_weight_names.size());
    APP_PRINTF("weights_num:%zd\n", weights_num);

    std::vector<const float*> w_fp32;
    w_fp32.resize(weights_num);
    loadWeightsPtr(w_fp32, fc, layer_num);
    return new SVTransformerPlugin(name,
                                   max_batch,
                                   max_seq_len,
                                   seq_len,
                                   embed_dim,
                                   num_heads,
                                   inter_size,
                                   layer_num,
                                   num_cam,
                                   num_reg_points,
                                   num_classes,
                                   l2i_matr_h,
                                   l2i_matr_w,
                                   img_shape,
                                   pc_range,
                                   1.0f,
                                   with_cls_token,
                                   w_fp32);
}

IPluginV2*
SVTransformerPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    char* modelData = (char*)serialData + sizeof(int);
    return new SVTransformerPlugin(name, modelData, serialLength);
}

void SVTransformerPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    namespace_ = libNamespace;
}

const char* SVTransformerPluginCreator::getPluginNamespace() const noexcept
{
    return namespace_.c_str();
}

}  // namespace fastertransformer
