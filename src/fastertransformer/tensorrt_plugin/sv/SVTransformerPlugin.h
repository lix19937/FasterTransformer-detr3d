/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-14 18:55:58
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-14 18:56:40
 **************************************************************/

#ifndef SV_TRANSFORMER_PLUGIN_H
#define SV_TRANSFORMER_PLUGIN_H

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntime.h"
#include "cublas_v2.h"
#include <cuda.h>

#include <string>
#include <vector>

#include "src/fastertransformer/models/sv/SV.h"
#include "src/fastertransformer/utils/allocator.h"

namespace fastertransformer {

struct SVSettings {
    size_t max_batch_size = 0;
    size_t max_seq_len = 0;
    size_t seq_len = 0;
    size_t embed_dim = 0;
    size_t head_num = 0;
    size_t inter_size = 0;
    size_t num_layer = 0;
    size_t num_cam = 0;
    size_t num_reg_points = 0;
    size_t num_classes = 0;
    size_t l2i_matr_h = 0;
    size_t l2i_matr_w = 0;
    int img_shape[2];
    // nvinfer1::DataType mType = kINT8; /// int8
    float pc_range[6];  ////// here be careful
};

class SVTransformerPlugin: public nvinfer1::IPluginV2DynamicExt {
public:
    SVTransformerPlugin(const std::string& name,
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
                        const int *img_shape,
                        const float* pc_range,
                        const float q_scaling,
                        const bool with_cls_token,
                        const std::vector<const float*>& w);

    SVTransformerPlugin(const std::string& name, const void* data, size_t length);
    SVTransformerPlugin(const SVTransformerPlugin& plugin);
    SVTransformerPlugin() = delete;

    ~SVTransformerPlugin() noexcept override;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex,
                                            const nvinfer1::DimsExprs* inputs,
                                            int nbInputs,
                                            nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int pos,
                                   const nvinfer1::PluginTensorDesc* inOut,
                                   int nbInputs,
                                   int nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                         int nbInputs,
                         const nvinfer1::DynamicPluginTensorDesc* out,
                         int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                            int nbInputs,
                            const nvinfer1::PluginTensorDesc* outputs,
                            int nbOutputs) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                const nvinfer1::PluginTensorDesc* outputDesc,
                const void* const* inputs,
                void* const* outputs,
                void* workspace,
                cudaStream_t stream) noexcept override;

    // IPluginV2Ext Methods
    nvinfer1::DataType
    getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    // IPluginV2 Methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    void attachToContext(cudnnContext* cudnnContext,
                         cublasContext* cublasContext,
                         nvinfer1::IGpuAllocator* gpuAllocator) noexcept override;
    void detachFromContext() noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    const std::string layer_name_;
    std::string namespace_;

    bool with_cls_token_ = true;
    int sm_ = 87; /// for nv-orin
    float q_scaling_ = 1.0f;
    AttentionType attention_type_ = AttentionType::UNFUSED_MHA;

    SVWeight<half>* params_ = nullptr;  /// for int8
    SVWeight<float>* fp_params_ = nullptr;

    SVTransformer<half>* sv_transformer_ = nullptr;  /// for int8
    SVTransformer<float>* fp_sv_transformer_ = nullptr;

    cublasHandle_t cublas_handle_ = nullptr;
    cublasLtHandle_t cublaslt_handle_ = nullptr;
    std::mutex* cublasWrapperMutex_ = nullptr;
    cublasAlgoMap* cublasAlgoMap_ = nullptr;
    fastertransformer::Allocator<AllocatorType::CUDA>* allocator_ = nullptr;
    cublasMMWrapper* cublas_wrapper_ = nullptr;


    ///// fp 
    cublasHandle_t fp_cublas_handle_ = nullptr;
    cublasLtHandle_t fp_cublaslt_handle_ = nullptr;
    std::mutex* fp_cublasWrapperMutex_ = nullptr;
    cublasAlgoMap* fp_cublasAlgoMap_ = nullptr;
    fastertransformer::Allocator<AllocatorType::CUDA>* fp_allocator_ = nullptr;
    cublasMMWrapper* fp_cublas_wrapper_ = nullptr;


    SVSettings settings_;

    void Init(const std::vector<const float*>& w,
              const SVTransformerPlugin& plugin,      
              const char* data = nullptr);
};

class SVTransformerPluginCreator: public nvinfer1::IPluginCreator {
public:
    SVTransformerPluginCreator();

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

    nvinfer1::IPluginV2*
    deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string namespace_;
};

}  // namespace fastertransformer
#endif  // TRT_SV_TRANSFORMER_H
