/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-14 18:55:58
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-14 18:56:40
 **************************************************************/

#include "src/fastertransformer/models/sv/SV.h"
#include "helper_file.h"
#include "helper_macros.h"
#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/kernels/vit_kernels.h"

namespace fastertransformer {

template<typename T>
void SVTransformer<T>::initialize()
{
    APP_PRINTF("\nmax_batch_size :%lu\n"
               "max_seq_len    :%lu\n"
               "embed_dim      :%lu\n"
               "head_num       :%lu\n"
               "head_dim       :%lu\n"
               "inter_size     :%lu\n"
               "num_layer      :%lu\n"
               "num_cam        :%lu\n"
               "l2i_matr_h     :%lu\n"
               "l2i_matr_w     :%lu\n"
               "img_shape      :%d %d\n"
               "pc_range       :%.6f %.6f %.6f %.6f %.6f %.6f\n"
               "att_type       :%d\n"
               "request_seq_len:%lu\n",
               max_batch_size_,
               max_seq_len_,
               embed_dim_,
               head_num_,
               head_dim_,
               inter_size_,
               num_layer_,
               num_cam_,
               l2i_h_,
               l2i_w_,
               img_shape_[0],
               img_shape_[1],
               pc_range_[0],
               pc_range_[1],
               pc_range_[2],
               pc_range_[3],
               pc_range_[4],
               pc_range_[5],
               int(attention_type_),
               request_seq_len_);

    /// pc_range normalize
    pc_range_[3] -= pc_range_[0];
    pc_range_[4] -= pc_range_[1];
    pc_range_[5] -= pc_range_[2];

    APP_PRINTF("\npc_range       :%.6f %.6f %.6f %.6f %.6f %.6f\n",
               pc_range_[0],
               pc_range_[1],
               pc_range_[2],
               pc_range_[3],
               pc_range_[4],
               pc_range_[5]);

    is_allocate_buffer_ = false;

    if (head_num_ * head_dim_ != embed_dim_) {
        std::ostringstream buffer;
        buffer << "[FT][ERROR] Embed size and head number mismatch. Embed_dim=" << embed_dim_
               << "; head_num*head_dim = "
               << "(" << head_num_ << "*" << head_dim_ << ")=" << head_num_ * head_dim_ << std::endl;
        throw std::runtime_error(buffer.str());
    }

    if (request_seq_len_ % 8 != 0 && std::is_same<half, T>::value) {
        max_seq_len_ = (request_seq_len_ + 7) / 8 * 8;
        FT_LOG_DEBUG(
            "Request sequence length(%lu) is odd with unfused mha. Padding to %lu\n", request_seq_len_, max_seq_len_);
    }

    APP_PRINTF(" new  SVMultiHeadAttentionLayer ...\n");
    svmh_attention_layer_ = new SVUnfusedAttentionLayer<T>(max_batch_size_,  //////// #1
                                                           max_seq_len_,
                                                           head_num_,
                                                           head_dim_,
                                                           q_scaling_,
                                                           stream_,
                                                           cublas_wrapper_,
                                                           allocator_,
                                                           is_free_buffer_after_forward_,
                                                           false);

    APP_PRINTF(" new  SVReluFfnLayer ...\n");
    ffn_layer_ = new ReluFfnLayer<T>(max_batch_size_,  //////// #2
                                     max_seq_len_,
                                     head_num_,
                                     head_dim_,
                                     inter_size_,
                                     stream_,
                                     cublas_wrapper_,
                                     allocator_,
                                     is_free_buffer_after_forward_,
                                     false);

    APP_PRINTF("new  SVCrossAttentionLayer ...\n");
    svcross_attention_layer_ = new SVCrossAttentionLayer<T>(max_batch_size_,  //////// #3
                                                            max_seq_len_,
                                                            embed_dim_,
                                                            num_cam_,
                                                            l2i_h_,
                                                            l2i_w_,
                                                            img_shape_,
                                                            pc_range_,
                                                            q_scaling_,
                                                            stream_,
                                                            cublas_wrapper_,
                                                            allocator_,
                                                            is_free_buffer_after_forward_,
                                                            false);
    APP_PRINTF("new  SVRegUpdateLayer ...\n");
    svreg_update_layer_ = new SVRegUpdateLayer<T>(max_batch_size_,  //////// #4
                                                  max_seq_len_,
                                                  embed_dim_,
                                                  pc_range_,
                                                  q_scaling_,
                                                  stream_,
                                                  cublas_wrapper_,
                                                  allocator_,
                                                  is_free_buffer_after_forward_,
                                                  false);
    APP_PRINTF("new  SVClsLayer ...\n");
    svcls_layer_ = new SVClsLayer<T>(max_batch_size_,  //////// #5
                                     max_seq_len_,
                                     embed_dim_,
                                     q_scaling_,
                                     stream_,
                                     cublas_wrapper_,
                                     allocator_,
                                     is_free_buffer_after_forward_,
                                     false);
    APP_PRINTF("new  Buffers ...\n");
    allocateBuffer();
    APP_PRINTF("Init Done %s\n", std::is_same<T, half>::value ? "FP16-int8" : "FP32");
}

template<typename T>
SVTransformer<T>::SVTransformer(size_t max_batch_size,
                                size_t max_seq_len,
                                size_t embed_dim,
                                size_t head_num,
                                size_t inter_size,
                                size_t num_layer,
                                size_t num_cam,
                                size_t l2i_h,
                                size_t l2i_w,
                                int* img_shape,
                                float* pc_range,
                                bool with_cls_token,
                                int sm,
                                float q_scaling,
                                const SVWeight<T>* weights,
                                cudaStream_t stream,
                                cublasMMWrapper* cublas_wrapper,
                                IAllocator* allocator,
                                bool is_free_buffer_after_forward,
                                AttentionType attention_type):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    embed_dim_(embed_dim),
    head_num_(head_num),
    inter_size_(inter_size),
    num_layer_(num_layer),
    num_cam_(num_cam),
    l2i_h_(l2i_h),
    l2i_w_(l2i_w),
    with_cls_token_(with_cls_token),
    sm_(sm),
    q_scaling_(q_scaling),
    weights_(weights),
    attention_type_(attention_type)
{
    ::memcpy(img_shape_, img_shape, 2 * sizeof(int));
    ::memcpy(pc_range_, pc_range, 6 * sizeof(float));

    head_dim_ = embed_dim / head_num;
    request_seq_len_ = max_seq_len;

    if (std::is_same<T, half>::value) {
        cublas_wrapper->setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper->setFP32GemmConfig();
    }

    initialize();
}

template<typename T>
SVTransformer<T>::SVTransformer(SVTransformer<T> const& sv):
    BaseLayer(sv),
    max_batch_size_(sv.max_batch_size_),
    max_seq_len_(sv.max_seq_len_),
    embed_dim_(sv.embed_dim_),
    head_num_(sv.head_num_),
    inter_size_(sv.inter_size_),
    num_layer_(sv.num_layer_),
    num_cam_(sv.num_cam_),
    l2i_h_(sv.l2i_h_),
    l2i_w_(sv.l2i_w_),
    with_cls_token_(sv.with_cls_token_),
    sm_(sv.sm_),
    q_scaling_(sv.q_scaling_),
    weights_(sv.weights_),
    attention_type_(sv.attention_type_),
    head_dim_(sv.head_dim_),
    request_seq_len_(sv.request_seq_len_)
{
    APP_PRINTF("copy constructor  ...\n");

    initialize();
}

template<typename T>
SVTransformer<T>::~SVTransformer()
{
    if (svmh_attention_layer_ != nullptr)
        delete svmh_attention_layer_;

    if (ffn_layer_ != nullptr)
        delete ffn_layer_;

    if (svcross_attention_layer_ != nullptr)
        delete svcross_attention_layer_;

    if (svreg_update_layer_ != nullptr)
        delete svreg_update_layer_;

    if (svcls_layer_ != nullptr)
        delete svcls_layer_;

    freeBuffer();
}

template<typename T>
void SVTransformer<T>::allocateBuffer()
{
    if (!is_allocate_buffer_) {
        query_buf_pair_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * embed_dim_ * 2, false);

        mha_out_buf_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * embed_dim_, false);

        mha_norm_out_buf_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * embed_dim_, false);

        mha_norm_out_with_pos_buf_ =
            (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * embed_dim_, false);

        ca_out_buf_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * embed_dim_, false);
        ca_pos_feat_buf_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * embed_dim_, false);

        ca_norm_out_buf_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * embed_dim_, false);

        ffn_out_buf_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * embed_dim_, false);

        ffn_norm_out_buf_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * embed_dim_, false);
        reg_out_buf_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * 3, false);

        /// for mha
        mask_buf_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * max_seq_len_, false);
        seq_len_vec_ = (int*)allocator_->malloc(sizeof(int) * max_batch_size_, false);

        setSeqLenVec(max_batch_size_);
        setDefaultMask(max_batch_size_);

        check_cuda_error(cudaDeviceSynchronize());
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void SVTransformer<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        allocator_->free(query_buf_pair_);

        allocator_->free(mha_out_buf_);
        allocator_->free(mha_norm_out_buf_);
        allocator_->free(mha_norm_out_with_pos_buf_);

        allocator_->free(ca_out_buf_);
        allocator_->free(ca_pos_feat_buf_);

        allocator_->free(ca_norm_out_buf_);

        allocator_->free(ffn_out_buf_);

        allocator_->free(ffn_norm_out_buf_);
        allocator_->free(reg_out_buf_);

        allocator_->free(mask_buf_);
        allocator_->free(seq_len_vec_);

        is_allocate_buffer_ = false;
    }
}

///------------------ In ---------------------
// value_0                        float32    [nc, ch, _, _]
// value_1                        float32    [nc, ch, _, _]
// value_2                        float32    [nc, ch, _, _]
// value_3                        float32    [nc, ch, _, _]
// lidar2img                      float32    [nc, 4, 4]
//
//
///------------------ Out ---------------------
//  ---output(query_out)          float32    [num_query, 1, embed_dim]---
//  ---reference_points           float32    [1, num_query, 3]----

//  reg_out(out_coord )           float32    [num_query, 1, 8]
//  cls_out                       float32    [1, num_query, 5]
///----------------- Attribute ----------------
//  pc_range                      float32    [1, 6]
//  num_query                     int64      [1]     900 or 512
//  num_cam                       int64      [1]     feats concat
//  l2i_matr_h                    int64      [1]     4
//  l2i_matr_w                    int64      [1]     4

template<typename T>
void SVTransformer<T>::forward(std::vector<Tensor>* output_tensors,
                               const std::vector<Tensor>* input_tensors,
                               cudaStream_t stream)
{
    FT_CHECK(output_tensors->at(0).shape.size() == 3);
    FT_CHECK(output_tensors->at(1).shape.size() == 3);

    const size_t input_batch_size = output_tensors->at(0).shape.at(0);
    const size_t seq_len = output_tensors->at(0).shape.at(1);
    FT_CHECK(seq_len <= max_seq_len_);

    DataType data_type = getTensorType<T>();
    Tensor offset_tensor_ptr(MEMORY_GPU, TYPE_INT32, {0}, nullptr);

    T* from_buf = query_buf_pair_;
    T* query_embed_buf = from_buf + seq_len * embed_dim_;

    const T* query = weights_->pre_transform_embeds.query;                        // update after each loop
    const T* reference_points = weights_->pre_transform_embeds.reference_points;  // update after each loop
    const T* query_pos = weights_->pre_transform_embeds.query_pos; // keep value const 

    cublas_wrapper_->setStream(stream);
    num_layer_ = 4;

    {
        int i = 0;
        //////////////////////// CA //////////////////////
        {
            std::vector<Tensor> in_tensors{
                input_tensors->at(0),  /// feats value
                input_tensors->at(1),
                input_tensors->at(2),

                input_tensors->at(3)  /// lidar2img
            };

            if (num_cam_ == 4) {  /// avod
                in_tensors.emplace_back(input_tensors->at(4));
                in_tensors.emplace_back(input_tensors->at(5));
            }

            std::vector<Tensor> out_tensors{Tensor{MEMORY_GPU, data_type, {seq_len, embed_dim_}, ca_out_buf_}};

            svcross_attention_layer_->forward_magic(&out_tensors,
                                                    &in_tensors,
                                                    &weights_->sv_layer_weights[i].ca_weights,
                                                    &weights_->helper_irparams,
                                                    stream);
        }

        //////////////////////// LN ////////////////////////////////////////////
        {
            //   mha_norm_out_buf_ (redisual)
            // + position_encoder_out (pos_feat)
            // + ca_output_proj_out_without_bias_(ca_out_buf_, svcross_attention_layer_`s out)
            // + bias  ---> ca_norm_out_buf_
            // pos_feat_with_inp_residual = redisual + pos_feat
            invokeGeneralAddBiasResidualPreLayerNorm(ca_out_buf_,
                                                     ca_norm_out_buf_,  /// OUT
                                                     weights_->helper_irparams.pos_feat_with_inp_residual,
                                                     weights_->sv_layer_weights[i].ca_norm_weights.gamma,
                                                     weights_->sv_layer_weights[i].ca_norm_weights.beta,
                                                     weights_->sv_layer_weights[i].ca_weights.output_proj.bias,
                                                     seq_len,
                                                     embed_dim_,
                                                     stream);
        }
        // FT_SAVE<T>("block_1.ca_out_buf_.log", {seq_len, embed_dim_}, (T*)ca_out_buf_);
        // FT_SAVE<T>("block_1.ca_norm_out_buf_.log", {seq_len, embed_dim_}, (T*)ca_norm_out_buf_);
        // FT_SAVE<T>("ir.ca.out.inp_res_pos_feat.log",
        //            {seq_len, embed_dim_},
        //            (T*)weights_->helper_irparams.pos_feat_with_inp_residual);

        //////////////////////// FFN ////////////////////////////////////////////
        {
            std::vector<Tensor> in_tensors{Tensor{MEMORY_GPU, data_type, {seq_len, embed_dim_}, ca_norm_out_buf_}};
            std::vector<Tensor> out_tensors{Tensor{MEMORY_GPU, data_type, {seq_len, embed_dim_}, ffn_out_buf_}};  // OUT
            ffn_layer_->forward(&out_tensors, &in_tensors, &weights_->sv_layer_weights[i].ffn_weights, stream);
        }

        //////////////////////// LN ////////////////////////////////////////////
        {
            invokeGeneralAddBiasResidualPreLayerNorm(ffn_out_buf_,
                                                     ffn_norm_out_buf_,
                                                     ca_norm_out_buf_,
                                                     weights_->sv_layer_weights[i].ffn_norm_weights.gamma,
                                                     weights_->sv_layer_weights[i].ffn_norm_weights.beta,
                                                     weights_->sv_layer_weights[i].ffn_weights.output_weight.bias,
                                                     seq_len,
                                                     embed_dim_,
                                                     stream);
        }

        //////////////////////// reg update ////////////////////////////////////////////
        {
            std::vector<Tensor> in_tensors{
                Tensor{MEMORY_GPU,
                       data_type,
                       {seq_len, input_batch_size, embed_dim_},
                       ffn_norm_out_buf_},  /// IN also LAST OUT, -->> next `query`
                Tensor{MEMORY_GPU, data_type, {seq_len, input_batch_size, 3}, reference_points}};  /// curr fixed

            std::vector<Tensor> out_tensors{Tensor{MEMORY_GPU,
                                                   data_type,
                                                   {seq_len, input_batch_size, 3},
                                                   reg_out_buf_}};  /// LAST OUT, -->> next `reference_points`

            svreg_update_layer_->forward(&out_tensors, &in_tensors, &weights_->sv_layer_weights[i].reg_weights, stream);

            query = ffn_norm_out_buf_;
            reference_points = reg_out_buf_;
        }
    }

    // FT_SAVE<T>("block_1.ffn_norm_out_buf_.log", {seq_len, embed_dim_}, (T *)ffn_norm_out_buf_);

    ////////////////////////////////////////////////////////
    for (uint i = 1; i < num_layer_; ++i) {
        {
            ///  query_embed_buf = query + query_pos
            invokeAddPosEmbed(query_embed_buf,
                              query,      /// update by loop
                              query_pos,  /// FIXED const always
                              seq_len,
                              embed_dim_,
                              stream);
        }

        cudaMemcpyAsync(from_buf, query, seq_len * embed_dim_ * sizeof(T), cudaMemcpyDeviceToDevice, stream);

        //////////////////////// MHA ////////////////////////////////////////////
        {
            std::vector<Tensor> in_tensors{
                Tensor{MEMORY_GPU, data_type, {seq_len, embed_dim_}, from_buf},
                Tensor{MEMORY_GPU, data_type, {input_batch_size, 1, seq_len, seq_len}, mask_buf_},
                offset_tensor_ptr};
            std::vector<Tensor> out_tensors{Tensor{MEMORY_GPU, data_type, {seq_len, embed_dim_}, mha_out_buf_}};

            svmh_attention_layer_->forward(
                &out_tensors, &in_tensors, &weights_->sv_layer_weights[i].mha_weights, stream);
        }

        //////////////////////// MHA`s residual merge in LN //////////////////////
        {
            svexp_pos::invokeGeneralAddBiasResidualPreLayerNorm(
                mha_out_buf_,
                mha_norm_out_buf_,           /// OUT, also IN of svcross_attention is (inp_residual)
                mha_norm_out_with_pos_buf_,  /// IN of svcross_attention
                query,
                query_pos,
                weights_->sv_layer_weights[i].mha_norm_weights.gamma,
                weights_->sv_layer_weights[i].mha_norm_weights.beta,
                weights_->sv_layer_weights[i].mha_weights.attention_output_weight.bias,
                seq_len,
                embed_dim_,
                stream);
        }

        //////////////////////// CA ////////////////////////////////////////////
        {
            std::vector<Tensor> in_tensors{
                Tensor{MEMORY_GPU, data_type, {seq_len, embed_dim_}, mha_norm_out_with_pos_buf_},
                Tensor{MEMORY_GPU, data_type, {seq_len, input_batch_size, 3}, reference_points},

                input_tensors->at(0),  /// feats value
                input_tensors->at(1),
                input_tensors->at(2),

                input_tensors->at(3)  /// lidar2img
            };
            if (num_cam_ == 4) {  /// avod
                in_tensors.emplace_back(input_tensors->at(4));
                in_tensors.emplace_back(input_tensors->at(5));
            }

            std::vector<Tensor> out_tensors{
                Tensor{MEMORY_GPU, data_type, {seq_len, embed_dim_}, ca_out_buf_},  /// OUT with no fc bias
                Tensor{MEMORY_GPU, data_type, {seq_len, embed_dim_}, ca_pos_feat_buf_}};

            svcross_attention_layer_->forward(
                &out_tensors, &in_tensors, &weights_->sv_layer_weights[i].ca_weights, stream);
        }

        //////////////////////// CA`s residual merge in LN //////////////////////
        {
            //  mha_norm_out_buf_ (redisual) + ca_pos_feat_buf_ + ca_out_buf_ + bias  ---> ca_norm_out_buf_
            svexp::invokeGeneralAddBiasResidualPreLayerNorm(ca_out_buf_,
                                                            ca_norm_out_buf_,
                                                            (const T*)mha_norm_out_buf_,
                                                            (const T*)ca_pos_feat_buf_,
                                                            weights_->sv_layer_weights[i].ca_norm_weights.gamma,
                                                            weights_->sv_layer_weights[i].ca_norm_weights.beta,
                                                            weights_->sv_layer_weights[i].ca_weights.output_proj.bias,
                                                            seq_len,
                                                            embed_dim_,
                                                            stream);
        }

        // if (i == 1) {
        //     FT_SAVE<T>("block_2.ca_out_buf_.log", {seq_len, embed_dim_}, (T*)ca_out_buf_);
        //     FT_SAVE<T>("block_2.ca_norm_out_buf_.log", {seq_len, embed_dim_}, (T*)ca_norm_out_buf_);
        // }
        //////////////////////// FFN ////////////////////////////////////////////
        {
            std::vector<Tensor> in_tensors{Tensor{MEMORY_GPU, data_type, {seq_len, embed_dim_}, ca_norm_out_buf_}};
            std::vector<Tensor> out_tensors{Tensor{MEMORY_GPU, data_type, {seq_len, embed_dim_}, ffn_out_buf_}};  // OUT
            ffn_layer_->forward(&out_tensors, &in_tensors, &weights_->sv_layer_weights[i].ffn_weights, stream);
        }

        //////////////////////// FFN`s residual merge in LN //////////////////////
        {
            invokeGeneralAddBiasResidualPreLayerNorm(ffn_out_buf_,
                                                     ffn_norm_out_buf_,
                                                     ca_norm_out_buf_,
                                                     weights_->sv_layer_weights[i].ffn_norm_weights.gamma,
                                                     weights_->sv_layer_weights[i].ffn_norm_weights.beta,
                                                     weights_->sv_layer_weights[i].ffn_weights.output_weight.bias,
                                                     seq_len,
                                                     embed_dim_,
                                                     stream);
        }

        // if (i == 3) {
        //     FT_SAVE<T>("block_4.ffn_norm_out_buf_.log", {seq_len, input_batch_size, embed_dim_},
        //     (T*)ffn_norm_out_buf_);
        // }

        if (i + 1 != num_layer_) {
            std::vector<Tensor> in_tensors{
                Tensor{MEMORY_GPU,
                       data_type,
                       {seq_len, input_batch_size, embed_dim_},
                       ffn_norm_out_buf_},  ///  -->> next `query`
                Tensor{MEMORY_GPU, data_type, {seq_len, input_batch_size, 3}, reference_points}};  /// curr fixed

            std::vector<Tensor> out_tensors{Tensor{MEMORY_GPU,
                                                   data_type,
                                                   {seq_len, input_batch_size, 3},
                                                   reg_out_buf_}};  ///  -->> next `reference_points`
            svreg_update_layer_->forward(&out_tensors, &in_tensors, &weights_->sv_layer_weights[i].reg_weights, stream);

            /// update
            query = ffn_norm_out_buf_;
            reference_points = reg_out_buf_;
        }
        else {
            std::vector<Tensor> in_tensors{
                Tensor{MEMORY_GPU, data_type, {seq_len, input_batch_size, embed_dim_}, ffn_norm_out_buf_},
                Tensor{MEMORY_GPU, data_type, {seq_len, input_batch_size, 3}, reference_points}};
            std::vector<Tensor> out_tensors{output_tensors->at(0)};  /// LAST

            svreg_update_layer_->forward_fused(
                &out_tensors, &in_tensors, &weights_->sv_layer_weights[i].reg_weights, stream);

            // FT_SAVE<T>(
            //     "block_4.reg_branches.in.reference_points.log", {seq_len, input_batch_size, 3},
            //     (T*)reference_points);

            {
                std::vector<Tensor> in_tensors{
                    Tensor{MEMORY_GPU, data_type, {seq_len, input_batch_size, embed_dim_}, ffn_norm_out_buf_}};
                std::vector<Tensor> out_tensors{output_tensors->at(1)};  /// LAST

                svcls_layer_->forward(&out_tensors, &in_tensors, &weights_->post_transform_weights, stream);
            }
        }
    }

    sync_check_cuda_error();
}

template<typename T>
bool SVTransformer<T>::setSeqLenVec(size_t batch_size)
{
    int* seq_len_vec = new int[batch_size];
    for (size_t i = 0; i < batch_size; ++i) {
        seq_len_vec[i] = request_seq_len_;
    }
    check_cuda_error(cudaMemcpy(seq_len_vec_, seq_len_vec, sizeof(int) * batch_size, cudaMemcpyHostToDevice));
    delete[] seq_len_vec;
    return true;
}

template<typename T>
void SVTransformer<T>::setDefaultMask(size_t batch_size)
{
    invokeBuildEncoderAttentionMask(mask_buf_, seq_len_vec_, batch_size, max_seq_len_, stream_);
}

template class SVTransformer<float>;
template class SVTransformer<half>;

}  // namespace fastertransformer
