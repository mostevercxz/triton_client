// Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <getopt.h>
#include <unistd.h>

#include <iostream>
#include <string>

#include "grpc_client.h"

namespace tc = triton::client;

#define FAIL_IF_ERR(X, MSG)                                        \
  {                                                                \
    tc::Error err = (X);                                           \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }

namespace {

void
ValidateShapeAndDatatype(
    const std::string& name, std::shared_ptr<tc::InferResult> result)
{
  std::vector<int64_t> shape;
  FAIL_IF_ERR(
      result->Shape(name, &shape), "unable to get shape for '" + name + "'");
  // Validate shape
  if ((shape.size() != 2) || (shape[0] != 1) || (shape[1] != 16)) {
    std::cerr << "error: received incorrect shapes for '" << name << "'"
              << std::endl;
    exit(1);
  }
  std::string datatype;
  FAIL_IF_ERR(
      result->Datatype(name, &datatype),
      "unable to get datatype for '" + name + "'");
  // Validate datatype
  if (datatype.compare("INT32") != 0) {
    std::cerr << "error: received incorrect datatype for '" << name
              << "': " << datatype << std::endl;
    exit(1);
  }
}

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << "error: " << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cerr << "\t-v" << std::endl;
  std::cerr << "\t-m <model name>" << std::endl;
  std::cerr << "\t-u <URL for inference service>" << std::endl;
  std::cerr << "\t-t <client timeout in microseconds>" << std::endl;
  std::cerr << "\t-H <HTTP header>" << std::endl;
  std::cerr
      << "\tFor -H, header must be 'Header:Value'. May be given multiple times."
      << std::endl;
  std::cerr << "\t-C <grpc compression algorithm>. \'deflate\', "
               "\'gzip\' and \'none\' are supported"
            << std::endl;
  std::cerr << "\t-c <use_cached_channel>. "
               " Use cached channel when creating new client. "
               " Specify 'true' or 'false'. True by default"
            << std::endl;
  std::cerr << std::endl;

  exit(1);
}

}  // namespace

int
main(int argc, char** argv)
{
  bool verbose = false;
  std::string url("localhost:8001");
  tc::Headers http_headers;
  uint32_t client_timeout = 0;
  bool use_ssl = false;
  std::string root_certificates;
  std::string private_key;
  std::string certificate_chain;
  grpc_compression_algorithm compression_algorithm =
      grpc_compression_algorithm::GRPC_COMPRESS_NONE;
  bool test_use_cached_channel = false;
  bool use_cached_channel = true;
  uint64_t uid = 1;

  // {name, has_arg, *flag, val}
  static struct option long_options[] = {
      {"ssl", 0, 0, 0},
      {"root-certificates", 1, 0, 1},
      {"private-key", 1, 0, 2},
      {"certificate-chain", 1, 0, 3}};

  // Parse commandline...
  int opt;
  while ((opt = getopt_long(argc, argv, "vu:i:t:H:C:c:", long_options, NULL)) !=
         -1) {
    switch (opt) {
      case 0:
        use_ssl = true;
        break;
      case 1:
        root_certificates = optarg;
        break;
      case 2:
        private_key = optarg;
        break;
      case 3:
        certificate_chain = optarg;
        break;
      case 'v':
        verbose = true;
        break;
      case 'u':
        url = optarg;
        break;
      case 'i':
        uid = std::stoul(optarg);
        break;
      case 't':
        client_timeout = std::stoi(optarg);
        break;
      case 'H': {
        std::string arg = optarg;
        std::string header = arg.substr(0, arg.find(":"));
        if (header.size() == arg.size() || header.empty()) {
          Usage(
              argv,
              "HTTP header specified incorrectly. Must be formmated as "
              "'Header:Value'");
        } else {
          http_headers[header] = arg.substr(header.size() + 1);
        }
        break;
      }
      case 'C': {
        std::string algorithm_str{optarg};
        if (algorithm_str.compare("deflate") == 0) {
          compression_algorithm =
              grpc_compression_algorithm::GRPC_COMPRESS_DEFLATE;
        } else if (algorithm_str.compare("gzip") == 0) {
          compression_algorithm =
              grpc_compression_algorithm::GRPC_COMPRESS_GZIP;
        } else if (algorithm_str.compare("none") == 0) {
          compression_algorithm =
              grpc_compression_algorithm::GRPC_COMPRESS_NONE;
        } else {
          Usage(
              argv,
              "unsupported compression algorithm specified... only "
              "\'deflate\', "
              "\'gzip\' and \'none\' are supported.");
        }
        break;
      }
      case 'c': {
        test_use_cached_channel = true;
        std::string arg = optarg;
        if (arg.find("false") != std::string::npos) {
          use_cached_channel = false;
        } else if (arg.find("true") != std::string::npos) {
          use_cached_channel = true;
        } else {
          Usage(argv, "need to specify true or false for use_cached_channel");
        }
        break;
      }
      case '?':
        Usage(argv);
        break;
    }
  }

  // We use a simple model that takes 2 input tensors of 16 integers
  // each and returns 2 output tensors of 16 integers each. One output
  // tensor is the element-wise sum of the inputs and one output is
  // the element-wise difference.
  std::string model_name = "tensorrt_llm";
  std::string model_version = "";

  // Create a InferenceServerGrpcClient instance to communicate with the
  // server using gRPC protocol.
  std::unique_ptr<tc::InferenceServerGrpcClient> client;
  tc::SslOptions ssl_options = tc::SslOptions();
  std::string err;
  if (use_ssl) {
    ssl_options.root_certificates = root_certificates;
    ssl_options.private_key = private_key;
    ssl_options.certificate_chain = certificate_chain;
    err = "unable to create secure grpc client";
  } else {
    err = "unable to create grpc client";
  }
  // Run with the same name to ensure cached channel is not used
  int numRuns = test_use_cached_channel ? 2 : 1;
  for (int i = 0; i < numRuns; ++i) {
    FAIL_IF_ERR(
        tc::InferenceServerGrpcClient::Create(
            &client, url, verbose, use_ssl, ssl_options, tc::KeepAliveOptions(),
            use_cached_channel),
        err);

    // Create the data for the two input tensors. Initialize the first
    // to unique integers and the second to all ones.
    //std::vector<int32_t> input0_data(16);
    std::vector<int32_t> input1_data(16);
    for (size_t i = 0; i < 16; ++i) {
      //input0_data[i] = i;
      input1_data[i] = 1;
    }
    std::vector<int32_t> input0_data{151647, 198, 56568, 99491, 99794, 100717, 103954, 105484, 3837, 100431, 105182, 102889, 100348, 108167, 3837, 35946, 102889, 101622, 57218, 56568, 105051, 198, 100780, 105924, 510, 90476, 100, 62922, 5122, 70108, 198, 74577, 112, 100820, 5122, 17, 17, 198, 90476, 100, 33983, 5122, 118375, 99696, 112425, 5373, 110576, 100668, 5373, 108295, 33108, 17340, 105292, 198, 4891, 244, 250, 99352, 5122, 109157, 11, 100364, 99204, 99629, 107691, 58143, 101622, 93149, 30709, 44793, 64205, 198, 33424, 101, 101738, 5122, 105905, 101070, 3837, 105905, 101070, 3837, 111686, 100397, 107691, 198, 112735, 100470, 106466, 87752, 104787, 104272, 198, 49602, 252, 58364, 30534, 101137, 100780, 105924, 198, 6567, 233, 240, 99631, 104787, 100390, 104013, 5373, 57621, 5373, 99602, 5373, 99599, 5373, 110569, 5373, 73218, 5373, 105358, 5373, 101091, 78556, 108380, 198, 49602, 252, 58364, 43815, 100645, 101137, 105492, 104773, 100376, 3837, 101137, 109157, 198, 6567, 233, 240, 99631, 37029, 105439, 5373, 118711, 5373, 105905, 5373, 112883, 9370, 110376, 715, 220, 113540, 112451, 101622, 17714, 2073, 102557, 16872, 88774, 220, 102762, 112451, 99681, 100474, 99669, 17714, 2073, 99681, 105666, 88774, 220, 103929, 104787, 110098, 36993, 99792, 36556, 53393, 3837, 117242, 99165, 107971, 104336, 8997, 220, 112735, 100470, 106466, 151647, 33108, 151648, 104186, 109504, 198, 88940, 121, 99475, 55338, 151647, 33108, 151648, 102069, 9370, 109504, 198, 56568, 100645, 100372, 101892, 100431, 9370, 102193, 43815, 71817, 105051, 1773, 62244, 101068, 45181, 16872, 110590, 101051, 102349, 37945, 36587, 1036, 102557, 16872, 105073, 3837, 35946, 99744, 99222, 101222, 32945, 28311, 220, 100348, 108167, 115568, 101141, 99816, 103982, 103976, 198, 220, 100348, 108167, 101909, 104130, 59074, 99821, 99417, 3837, 103046, 18493, 104130, 59074, 102829, 1773, 99204, 99681, 100474, 99669, 102857, 3837, 100119, 82894, 104130, 59074, 3837, 99663, 99466, 100638, 103985, 8997, 220, 100348, 108167, 107204, 104158, 99681, 100474, 99669, 105633, 99424, 3837, 99246, 101952, 99445, 30709, 100422, 104989, 26288, 102530, 3837, 99601, 99445, 30709, 100422, 99486, 100348, 31207, 102073, 108001, 8997, 220, 100348, 108167, 18493, 104130, 59074, 116391, 101243, 3837, 99999, 99392, 99555, 104130, 59074, 104006, 44793, 64205, 3837, 100671, 101622, 56007, 3837, 100348, 108167, 36993, 104584, 101883, 30709, 44793, 64205, 89012, 101622, 8997, 220, 100348, 108167, 99392, 99555, 104130, 59074, 104006, 44793, 64205, 8997, 105048, 105267, 87752, 105051, 19793, 26355, 3837, 45912, 100348, 31207, 102073, 104283, 75768, 510, 89982, 63836, 5122, 104044, 104130, 59074, 108432, 101037, 94432, 220, 100348, 108167, 5122, 42140, 99757, 102557, 101373, 104329, 3837, 104044, 104130, 59074, 101109, 100416, 8997, 89982, 63836, 5122, 105043, 100165, 94432, 220, 100348, 108167, 5122, 35946, 99882, 100348, 108167, 3837, 20412, 104130, 59074, 9370, 99821, 99417, 1773, 97611, 104559, 20412, 109270, 100147, 34718, 101318, 90395, 89012, 102464, 99553, 85106, 116211, 8997, 89982, 63836, 5122, 104888, 102328, 198, 220, 100348, 108167, 5122, 102557, 16872, 118271, 117070, 8997, 89982, 63836, 5122, 107733, 100625, 101037, 94432, 220, 100348, 108167, 5122, 100625, 11319, 110985, 114056, 3837, 100707, 65676, 20412, 99569, 44290, 28404, 17992, 94432, 89982, 63836, 5122, 107733, 104730, 59074, 103976, 101037, 198, 220, 100348, 108167, 5122, 49187, 99608, 24562, 97084, 102534, 104730, 59074, 102207, 34187, 52510, 99811, 3837, 99744, 52183, 99601, 100007, 34187, 8997, 89982, 63836, 5122, 56568, 104067, 106428, 198, 220, 100348, 108167, 5122, 104786, 101254, 109270, 15946, 37945, 56007, 104139, 111728, 102557, 101373, 101037, 94432, 89982, 63836, 5122, 99445, 30709, 100422, 198, 220, 100348, 108167, 5122, 102557, 16872, 100720, 99445, 26288, 102530, 101037, 11319, 42411, 103933, 97611, 108001, 6313, 198, 151648};

    std::vector<int64_t> shape{1, 16};

    // Initialize the inputs with the data.
    tc::InferInput* input0;
    tc::InferInput* input1;

    FAIL_IF_ERR(
        tc::InferInput::Create(&input0, "INPUT0", shape, "INT32"),
        "unable to get INPUT0");
    std::shared_ptr<tc::InferInput> input0_ptr;
    input0_ptr.reset(input0);
    FAIL_IF_ERR(
        tc::InferInput::Create(&input1, "INPUT1", shape, "INT32"),
        "unable to get INPUT1");
    std::shared_ptr<tc::InferInput> input1_ptr;
    input1_ptr.reset(input1);

    FAIL_IF_ERR(
        input0_ptr->AppendRaw(
            reinterpret_cast<uint8_t*>(&input0_data[0]),
            input0_data.size() * sizeof(int32_t)),
        "unable to set data for INPUT0");
    FAIL_IF_ERR(
        input1_ptr->AppendRaw(
            reinterpret_cast<uint8_t*>(&input1_data[0]),
            input1_data.size() * sizeof(int32_t)),
        "unable to set data for INPUT1");

    // Generate the outputs to be requested.
    tc::InferRequestedOutput* output0;
    tc::InferRequestedOutput* output1;

    FAIL_IF_ERR(
        tc::InferRequestedOutput::Create(&output0, "OUTPUT0"),
        "unable to get 'OUTPUT0'");
    std::shared_ptr<tc::InferRequestedOutput> output0_ptr;
    output0_ptr.reset(output0);
    FAIL_IF_ERR(
        tc::InferRequestedOutput::Create(&output1, "OUTPUT1"),
        "unable to get 'OUTPUT1'");
    std::shared_ptr<tc::InferRequestedOutput> output1_ptr;
    output1_ptr.reset(output1);


    // The inference settings. Will be using default for now.
    tc::InferOptions options(model_name);
    options.model_version_ = model_version;
    options.client_timeout_ = client_timeout;

    std::vector<tc::InferInput*> inputs = {input0_ptr.get(), input1_ptr.get()};
    std::vector<const tc::InferRequestedOutput*> outputs = {
        output0_ptr.get(), output1_ptr.get()};

    tc::InferResult* results;
    FAIL_IF_ERR(
        client->Infer(
            &results, options, inputs, outputs, http_headers,
            compression_algorithm),
        "unable to run model");
    std::shared_ptr<tc::InferResult> results_ptr;
    results_ptr.reset(results);

    // Validate the results...
    ValidateShapeAndDatatype("OUTPUT0", results_ptr);
    ValidateShapeAndDatatype("OUTPUT1", results_ptr);

    // Get pointers to the result returned...
    int32_t* output0_data;
    size_t output0_byte_size;
    FAIL_IF_ERR(
        results_ptr->RawData(
            "OUTPUT0", (const uint8_t**)&output0_data, &output0_byte_size),
        "unable to get result data for 'OUTPUT0'");
    if (output0_byte_size != 64) {
      std::cerr << "error: received incorrect byte size for 'OUTPUT0': "
                << output0_byte_size << std::endl;
      exit(1);
    }

    int32_t* output1_data;
    size_t output1_byte_size;
    FAIL_IF_ERR(
        results_ptr->RawData(
            "OUTPUT1", (const uint8_t**)&output1_data, &output1_byte_size),
        "unable to get result data for 'OUTPUT1'");
    if (output1_byte_size != 64) {
      std::cerr << "error: received incorrect byte size for 'OUTPUT1': "
                << output1_byte_size << std::endl;
      exit(1);
    }

    for (size_t i = 0; i < 16; ++i) {
      std::cout << input0_data[i] << " + " << input1_data[i] << " = "
                << *(output0_data + i) << std::endl;
      std::cout << input0_data[i] << " - " << input1_data[i] << " = "
                << *(output1_data + i) << std::endl;

      if ((input0_data[i] + input1_data[i]) != *(output0_data + i)) {
        std::cerr << "error: incorrect sum" << std::endl;
        exit(1);
      }
      if ((input0_data[i] - input1_data[i]) != *(output1_data + i)) {
        std::cerr << "error: incorrect difference" << std::endl;
        exit(1);
      }
    }

    // Get full response
    std::cout << results_ptr->DebugString() << std::endl;

    tc::InferStat infer_stat;
    client->ClientInferStat(&infer_stat);
    std::cout << "======Client Statistics======" << std::endl;
    std::cout << "completed_request_count "
              << infer_stat.completed_request_count << std::endl;
    std::cout << "cumulative_total_request_time_ns "
              << infer_stat.cumulative_total_request_time_ns << std::endl;
    std::cout << "cumulative_send_time_ns "
              << infer_stat.cumulative_send_time_ns << std::endl;
    std::cout << "cumulative_receive_time_ns "
              << infer_stat.cumulative_receive_time_ns << std::endl;

    inference::ModelStatisticsResponse model_stat;
    client->ModelInferenceStatistics(&model_stat, model_name);
    std::cout << "======Model Statistics======" << std::endl;
    std::cout << model_stat.DebugString() << std::endl;


    std::cout << "PASS : Infer" << std::endl;
  }
  return 0;
}
