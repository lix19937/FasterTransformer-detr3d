/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-14 18:55:58
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-14 18:56:40
 **************************************************************/

#pragma once

#include "SVWeight.h"
#include "helper_file.h"
#include <map>
#include <numeric>

namespace fastertransformer {

/// here just for cpp/sv_example 48;   no-timing model use 24;  avod use 16
const static int __dims3 = 16;

const std::map<int, const char*> __layer_weight_names_ca_fc_bias{
    {16, "block_%d.cross_attention.attention_weights.fc.bias.1-16"},
    {24, "block_%d.cross_attention.attention_weights.fc.bias.1-24"},
    {48, "block_%d.cross_attention.attention_weights.fc.bias.1-48"}};

const std::map<int, const char*> __layer_weight_names_ca_fc_weight{
    {16, "block_%d.cross_attention.attention_weights.fc.weight.256-16"},
    {24, "block_%d.cross_attention.attention_weights.fc.weight.256-24"},
    {48, "block_%d.cross_attention.attention_weights.fc.weight.256-48"}};

const std::vector<std::pair<const char*, bool>> layer_weight_names{
    /// mha
    {"block_%d.mh_attention.query.weight.256-256", false},
    {"block_%d.mh_attention.query.bias.1-256", false},

    {"block_%d.mh_attention.key.weight.256-256", false},
    {"block_%d.mh_attention.key.bias.1-256", false},

    {"block_%d.mh_attention.value.weight.256-256", false},
    {"block_%d.mh_attention.value.bias.1-256", false},

    {"block_%d.mh_attention.out.weight.256-256", false},
    {"block_%d.mh_attention.out.bias.1-256", false},

    /// mha-ln
    {"block_%d.mh_attention_norm.ln.weight.1-256", false},
    {"block_%d.mh_attention_norm.ln.bias.1-256", false},

    /// ca
    {__layer_weight_names_ca_fc_weight.at(__dims3), false},
    {__layer_weight_names_ca_fc_bias.at(__dims3), false},

    {"block_%d.cross_attention.output_proj.fc.weight.256-256", false},
    {"block_%d.cross_attention.output_proj.fc.bias.1-256", false},

    {"block_%d.cross_attention.position_encoder.fc1.weight.3-256", false},
    {"block_%d.cross_attention.position_encoder.fc1.bias.1-256", false},

    {"block_%d.cross_attention.position_encoder.ln1.weight.1-256", false},
    {"block_%d.cross_attention.position_encoder.ln1.bias.1-256", false},

    {"block_%d.cross_attention.position_encoder.fc2.weight.256-256", false},
    {"block_%d.cross_attention.position_encoder.fc2.bias.1-256", false},

    {"block_%d.cross_attention.position_encoder.ln2.weight.1-256", false},
    {"block_%d.cross_attention.position_encoder.ln2.bias.1-256", false},

    /// ca-ln
    {"block_%d.cross_attention_norm.ln.weight.1-256", false},
    {"block_%d.cross_attention_norm.ln.bias.1-256", false},

    /// ffn
    {"block_%d.ffn.fc1.weight.256-512", false},
    {"block_%d.ffn.fc1.bias.1-512", false},

    {"block_%d.ffn.fc2.weight.512-256", false},
    {"block_%d.ffn.fc2.bias.1-256", false},

    /// ffn-ln
    {"block_%d.ffn_norm.ln.weight.1-256", false},
    {"block_%d.ffn_norm.ln.bias.1-256", false},

    /// reg_branches
    {"block_%d.reg_branches.fc1.weight.256-256", false},
    {"block_%d.reg_branches.fc1.bias.1-256", false},

    {"block_%d.reg_branches.fc2.weight.256-256", false},
    {"block_%d.reg_branches.fc2.bias.1-256", false},

    {"block_%d.reg_branches.fc3.weight.256-8", false},
    {"block_%d.reg_branches.fc3.bias.1-8", false}};

const std::vector<std::pair<const char*, bool>> pre_weight_names{{"posembed.in.query.512-1-256", false},
                                                                 {"posembed.in.query_pos.512-1-256", false},
                                                                 {"reg.in.reference_points.1-512-3", false}};

const std::map<int, const char*> __helper_irparam_names{{16, "ir.ca.attention_weights.out_nobias.1-512-16"},
                                                        {24, "ir.ca.attention_weights.out_nobias.1-512-24"},
                                                        {48, "ir.ca.attention_weights.out_nobias.1-512-48"}};

const std::vector<std::pair<const char*, bool>> helper_irparam_names{{"ir.ca.fs.rfpcat.1-4-512", false},  /// transpose
                                                                     {__helper_irparam_names.at(__dims3), false},
                                                                     {"ir.ca.out.inp_res_pos_feat.512-1-256", false}};

const std::vector<std::pair<const char*, bool>> post_weight_names{{"cls_branches.fc1.weight.256-256", false},
                                                                  {"cls_branches.fc1.bias.1-256", false},
                                                                  {"cls_branches.ln1.weight.1-256", false},
                                                                  {"cls_branches.ln1.bias.1-256", false},
                                                                  {"cls_branches.fc2.weight.256-256", false},
                                                                  {"cls_branches.fc2.bias.1-256", false},
                                                                  {"cls_branches.ln2.weight.1-256", false},
                                                                  {"cls_branches.ln2.bias.1-256", false},
                                                                  //// be careful here  5 --> 8 !!!
                                                                  {"cls_branches.fc3.weight.256-8", false},
                                                                  {"cls_branches.fc3.bias.1-8", false}};

template<typename T>
int Load(SVWeight<T>& para, const std::string& bpath = "tf_decoder_weights/")
{
    APP_PRINTF("load pre layer ...\n");

    /// pre_wights
    for (size_t j = 0; j < pre_weight_names.size(); ++j) {
        const auto it = pre_weight_names[j];

        ///  get dims
        std::string cur_str(it.first);
        auto dot_pos = cur_str.rfind(".");
        auto dims_str = cur_str.substr(dot_pos + 1);
        std::vector<int> dims;
        cudaacc::split<int>(dims, dims_str, '-');

        APP_PRINTF("%zd, %s \n", j, cur_str.c_str());

        auto len = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
        auto dptr = para.GetPtr(j);
        cudaacc::Read2Dptr(bpath + "pre/" + cur_str, dims, dptr, it.second);
    }

    APP_PRINTF("load helper_ir layer ...\n");

    ///  helper ir
    for (size_t j = 0; j < helper_irparam_names.size(); ++j) {
        const auto it = helper_irparam_names[j];

        ///  get dims
        std::string cur_str(it.first);
        auto dot_pos = cur_str.rfind(".");
        auto dims_str = cur_str.substr(dot_pos + 1);
        std::vector<int> dims;
        cudaacc::split<int>(dims, dims_str, '-');

        APP_PRINTF("%zd, %s \n", j, cur_str.c_str());

        auto len = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
        auto dptr = para.GetPtr(j + pre_weight_names.size());  // !
        cudaacc::Read2Dptr(bpath + "ir/" + cur_str, dims, dptr, it.second);
    }

    APP_PRINTF("load post layer ...\n");

    /// post
    for (size_t j = 0; j < post_weight_names.size(); ++j) {
        const auto it = post_weight_names[j];

        ///  get dims
        std::string cur_str(it.first);
        auto dot_pos = cur_str.rfind(".");
        auto dims_str = cur_str.substr(dot_pos + 1);
        std::vector<int> dims;
        cudaacc::split<int>(dims, dims_str, '-');

        APP_PRINTF("%zd, %s \n", j, cur_str.c_str());

        auto len = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
        auto dptr = para.GetPtr(j + pre_weight_names.size() + helper_irparam_names.size());  // !
        cudaacc::Read2Dptr(bpath + "post/" + cur_str, dims, dptr, it.second);
    }

    APP_PRINTF("load block layer ...\n");

    /// block layer
    const int layer_num = 4;
    for (int i = 0; i < layer_num; ++i) {
        for (size_t j = 0; j < layer_weight_names.size(); ++j) {
            const auto it = layer_weight_names[j];
            char str_buf[1024]{0};
            sprintf(str_buf, it.first, i + 1);

            ///  get dims
            std::string cur_str(str_buf);
            auto dot_pos = cur_str.rfind(".");
            auto dims_str = cur_str.substr(dot_pos + 1);
            std::vector<int> dims;
            cudaacc::split<int>(dims, dims_str, '-');

            auto len = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
            auto dptr = para.sv_layer_weights[i].GetPtr(j);
            cudaacc::Read2Dptr(bpath + cur_str, dims, dptr, it.second);
        }
    }

    return 0;
}

}  // namespace fastertransformer
