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

#include <unistd.h>

#include <condition_variable>
#include <iostream>
#include <mutex>
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
  if ((shape.size() != 3)) {
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
ValidateResult(
    const std::shared_ptr<tc::InferResult> result,
    std::vector<int32_t>& input0_data, std::vector<int32_t>& input1_data)
{
  // Validate the results...
  ValidateShapeAndDatatype("output_ids", result);
  //ValidateShapeAndDatatype("OUTPUT1", result);

  // Get pointers to the result returned...
  int32_t* output0_data;
  size_t output0_byte_size;
  FAIL_IF_ERR(
      result->RawData(
          "output_ids", (const uint8_t**)&output0_data, &output0_byte_size),
      "unable to get result data for 'OUTPUT0'");
  //if (output0_byte_size != 32) {
    //std::cerr << "error: received incorrect byte size for 'OUTPUT0': "
              //<< output0_byte_size << std::endl;
    //exit(1);
  //}

  //int32_t* output1_data;
  //size_t output1_byte_size;
  //FAIL_IF_ERR(
      //result->RawData(
          //"OUTPUT1", (const uint8_t**)&output1_data, &output1_byte_size),
      //"unable to get result data for 'OUTPUT1'");
  //if (output0_byte_size != 64) {
    //std::cerr << "error: received incorrect byte size for 'OUTPUT1': "
              //<< output0_byte_size << std::endl;
    //exit(1);
  //}

  for (size_t i = 0; i < 16; ++i) {
    std::cout << input0_data[i] << " + " << input1_data[i] << " = "
              << *(output0_data + i) << std::endl;
    //std::cout << input0_data[i] << " - " << input1_data[i] << " = "
              //<< *(output1_data + i) << std::endl;

    //if ((input0_data[i] + input1_data[i]) != *(output0_data + i)) {
      //std::cerr << "error: incorrect sum" << std::endl;
      //exit(1);
    //}
    //if ((input0_data[i] - input1_data[i]) != *(output1_data + i)) {
      //std::cerr << "error: incorrect difference" << std::endl;
      //exit(1);
    //}
  }

  // Get full response
  std::cout << result->DebugString() << std::endl;
}

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << "error: " << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cerr << "\t-v" << std::endl;
  std::cerr << "\t-u <URL for inference service>" << std::endl;
  std::cerr << "\t-t <client timeout in microseconds>" << std::endl;
  std::cerr << "\t-H <HTTP header>" << std::endl;
  std::cerr << std::endl;
  std::cerr
      << "For -H, header must be 'Header:Value'. May be given multiple times."
      << std::endl;

  exit(1);
}

}  // namespace

int
main(int argc, char** argv)
{
  bool verbose = true;
  uint64_t uid = 1;
  std::string url("localhost:8001");
  tc::Headers http_headers;
  uint32_t client_timeout = 0;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vu:i:t:H:")) != -1) {
    switch (opt) {
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
        http_headers[header] = arg.substr(header.size() + 1);
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
  FAIL_IF_ERR(
      tc::InferenceServerGrpcClient::Create(&client, url, verbose),
      "unable to create grpc client");

  // Create the data for the two input tensors. Initialize the first
  // to unique integers and the second to all ones.
  //std::vector<int32_t> input0_data(16);
  //std::vector<int32_t> input0_data(1);
  std::vector<int32_t> input0_data{151644,   8948,    198,   2610,    525,    264,  10950,  17847,
            13, 151645,    198, 151644,  20002,    198,     27,     91,
         15460,     62,     16,     91,    397,  56568,  99491,  99794,
        100717, 103954, 105484,   3837, 100431, 105182, 102889, 103116,
          3837,  35946, 102889, 101622,  57218,  56568, 105051,    198,
        100780, 105924,    510,    262,  90476,    100,  62922,   5122,
         70108,    198,    262,  90476,    100,  33983,   5122, 109739,
        116095,   3837, 116274,  29258,   3837, 118744, 102783,   3837,
         99729,  56007,  86119,    198,    262,   4891,    244,    250,
         99352,   5122, 103153, 102115,  17340,    198,    262,  33424,
           101, 101738,   5122, 115270, 100623,   3837, 100134,    198,
        112735, 100470, 106466,  87752, 104787, 104272,  28311,    262,
         49602,    252,  58364,  30534, 101137, 100780, 105924,    198,
           262,  49602,    252,  58364,  18830, 100645,  18830, 113369,
           198,    262,   6567,    233,    240,  99631, 104787, 100390,
        104013,   5373,  57621,   5373,  99602,   5373,  99599,   5373,
        110569,   5373,  73218,   5373, 105358,  78556, 108380,    198,
           262,  49602,    252,  58364,  11622, 106267, 102783,  33108,
        102744, 105219,   9370, 110098,   8997,    262,  49602,    252,
         58364,  30534, 101447, 110485,  41453,    262,    220,  56568,
        112451, 101622,  17714,   2073, 108207,  88774,    262,    220,
         56568, 112451,  99283,  17714,   2073,  21894, 103954,  99677,
         88774,    262,    220, 112735, 100470, 106466,     27,     91,
         15460,     62,     16,     91,     29,  33108,     27,     91,
         15460,     62,     17,     91,     29, 104186, 109504,    198,
           262,  88940,    121,  99475,  55338,     27,     91,  15460,
            62,     16,     91,     29,  33108,     27,     91,  15460,
            62,     17,     91,     29, 102069,   9370, 109504,    198,
         56568, 100645, 100372, 101892, 100431,   9370, 102193,  43815,
         71817, 105051,   1773,  62244, 101068,  45181,  16872, 110590,
        101051, 102349,  41505, 107314,     11,  21894, 103954,  99677,
        104555, 103950,  33590, 110263,  30868,  66078,   3837, 104787,
        104811,    510,    262,  53599,    228,  99333, 101451,  99677,
        101336,   3837, 112475,  34204, 104974, 101920,   3837,  42192,
         99910,  42192,  99740,   3837, 104585, 101622,  17714, 100005,
        108207,   8997,    262,  53599,    228,  99333, 115568, 101141,
         99816, 103982, 103976,   3407, 105048, 105267,  87752, 105051,
         19793,  26355,   3837,  45912, 103116,   9370, 104283,  75768,
           510,    262,  89982,  63836,   5122, 105043, 100165,    198,
           262,  53599,    228,  99333,   5122, 107314,   6313,  35946,
        101451,  17447,  35727,  17254,  29490, 108944,  53153,   5373,
         99566,  97120,  99609,  70769,  99363, 113176,   9370, 103954,
         99677,   8545, 103116,  20412,  74763,   6313,    198,    262,
         89982,  63836,   5122, 108386, 103924,    198,    262,  53599,
           228,  99333,   5122, 108386, 104256,     93, 108207,  92133,
        103116,  20412, 104139,  29826, 101037,  94432,    262,  89982,
         63836,   5122,  56568, 104104,  26288, 104888, 102328,    198,
           262,  53599,    228,  99333,   5122, 107314,   6313, 101228,
         99521,  70927, 113369,   9370,   3837,  21894, 103954,  99677,
        104359,  33108,  56568, 104283,  34187,  31251,   6313,  30440,
         99695, 100623,  21515,   6313,    198,    262,  89982,  63836,
          5122, 113540, 110648,    198,    262,  53599,    228,  99333,
          5122, 106287,     93,  21894, 103954,  99677,  99729,  26288,
        100655,  26288,  99894,   3837, 111596,  99729, 111383,  94432,
           262,  89982,  63836,   5122,  35946,  52801, 110702,    198,
           262,  53599,    228,  99333,   5122, 108207,  99494,  34187,
         11319, 100165, 111971,  56568,  34187,     30, 103116, 109031,
            11,  56568, 101901, 102313,  99793, 101036,   5267,     27,
            91,  15460,     62,     17,     91,    397, 101622,   5122,
        105043, 100165,    198, 103116,   5122, 107314,   6313,  35946,
        101451,  17447,  35727,  17254,  29490, 108944,  53153,   5373,
         99566,  97120,  99609,  70769,  99363, 113176,   9370, 103954,
         99677,   8545, 103116,  20412,  74763,   6313,    198, 101622,
          5122,  35946, 104044,  99557, 105367, 105209,  34187,   3837,
         35946,  52801, 109384, 104256,   3837, 108965,  61443, 103761,
            17,     15,     15,  18600,   9370, 101821,  99337,  21317,
          3837,  99236,  45861, 108332,   3837, 102570,    198, 103116,
          5122, 151645,    198, 151644,  85254,  35727,    198};
  std::shared_ptr<tc::InferInput> input0_ptr;
  std::shared_ptr<tc::InferInput> input2_ptr;
  std::shared_ptr<tc::InferInput> request_output_len_tensor_ptr;
  std::shared_ptr<tc::InferInput> end_id_tensor_ptr;
  std::shared_ptr<tc::InferInput> pad_id_tensor_ptr;
  std::shared_ptr<tc::InferInput> beam_width_tensor_ptr;
  std::shared_ptr<tc::InferInput> temperature_tensor_ptr;
  {
  std::vector<int64_t> input_ids_shape{1, 1};
  input_ids_shape[1] = input0_data.size();
  tc::InferInput* input0;
  FAIL_IF_ERR(
      tc::InferInput::Create(&input0, "input_ids", input_ids_shape, "INT32"),
      "unable to get INPUT0");
  input0_ptr.reset(input0);
  FAIL_IF_ERR(
      input0_ptr->AppendRaw(
          reinterpret_cast<uint8_t*>(&input0_data[0]),
          input0_data.size() * sizeof(int32_t)),
      "unable to set data for INPUT0");
  }


  {
  std::vector<int32_t> input_lengths_data(1);
  input_lengths_data[0] = input0_data.size();
  std::vector<int64_t> input_lengths_shape{1, 1};
  // lengths
  tc::InferInput* input_lengths_tensor;
  FAIL_IF_ERR(
      tc::InferInput::Create(&input_lengths_tensor, "input_lengths", input_lengths_shape, "INT32"),
      "unable to get INPUT0");
  input2_ptr.reset(input_lengths_tensor);

  FAIL_IF_ERR(
      input2_ptr->AppendRaw(
          reinterpret_cast<uint8_t*>(&input_lengths_data[0]),
          input_lengths_data.size() * sizeof(int32_t)),
      "unable to set data for INPUT0");
  }

  {
  std::vector<uint32_t> request_output_len_data(1);
  request_output_len_data[0] = 512;
  std::vector<int64_t> output_len_shape{1, 1};
  tc::InferInput* request_output_len_tensor;
  FAIL_IF_ERR(
      tc::InferInput::Create(&request_output_len_tensor, "request_output_len", output_len_shape, "UINT32"),
      "unable to get INPUT0");
  request_output_len_tensor_ptr.reset(request_output_len_tensor);
  FAIL_IF_ERR(
      request_output_len_tensor_ptr->AppendRaw(
          reinterpret_cast<uint8_t*>(&request_output_len_data[0]),
          request_output_len_data.size() * sizeof(uint32_t)),
      "unable to set data for INPUT0");
  }

  // end_id
  {
    std::vector<uint32_t> end_id_data{151645};
    std::vector<int64_t> end_id_shape{1,1};
    tc::InferInput* end_id_tensor;
  FAIL_IF_ERR(
      tc::InferInput::Create(&end_id_tensor, "end_id", end_id_shape, "UINT32"),
      "unable to get INPUT0");
  end_id_tensor_ptr.reset(end_id_tensor);
  FAIL_IF_ERR(
      end_id_tensor_ptr->AppendRaw(
          reinterpret_cast<uint8_t*>(&end_id_data[0]),
          end_id_data.size() * sizeof(uint32_t)),
      "unable to set data for INPUT0");
  }

  // pad_id
  {
    std::vector<uint32_t> pad_id_data{151645};
    std::vector<int64_t> pad_id_shape{1,1};
    tc::InferInput* pad_id_tensor;
  FAIL_IF_ERR(
      tc::InferInput::Create(&pad_id_tensor, "pad_id", pad_id_shape, "UINT32"),
      "unable to get INPUT0");
  pad_id_tensor_ptr.reset(pad_id_tensor);
  FAIL_IF_ERR(
      pad_id_tensor_ptr->AppendRaw(
          reinterpret_cast<uint8_t*>(&pad_id_data[0]),
          pad_id_data.size() * sizeof(uint32_t)),
      "unable to set data for INPUT0");
  }

  // beam_width
  {
    std::vector<uint32_t> beam_width_data{1};
    std::vector<int64_t> beam_width_shape{1,1};
    tc::InferInput* beam_width_tensor;
  FAIL_IF_ERR(
      tc::InferInput::Create(&beam_width_tensor, "beam_width", beam_width_shape, "UINT32"),
      "unable to get INPUT0");
  beam_width_tensor_ptr.reset(beam_width_tensor);
  FAIL_IF_ERR(
      beam_width_tensor_ptr->AppendRaw(
          reinterpret_cast<uint8_t*>(&beam_width_data[0]),
          beam_width_data.size() * sizeof(uint32_t)),
      "unable to set data for INPUT0");
  }

  // temperature
  {
    std::vector<float> temperature_data{0.9};
    std::vector<int64_t> temperature_shape{1,1};
    tc::InferInput* temperature_tensor;
  FAIL_IF_ERR(
      tc::InferInput::Create(&temperature_tensor, "temperature", temperature_shape, "FP32"),
      "unable to get INPUT0");
  temperature_tensor_ptr.reset(temperature_tensor);
  FAIL_IF_ERR(
      temperature_tensor_ptr->AppendRaw(
          reinterpret_cast<uint8_t*>(&temperature_data[0]),
          temperature_data.size() * sizeof(float)),
      "unable to set data for INPUT0");
  }


  // Generate the outputs to be requested.
  tc::InferRequestedOutput* output0;
  tc::InferRequestedOutput* output1;

  FAIL_IF_ERR(
      tc::InferRequestedOutput::Create(&output0, "output_ids"),
      "unable to get 'output_ids'");
  std::shared_ptr<tc::InferRequestedOutput> output0_ptr;
  output0_ptr.reset(output0);

  // The inference settings. Will be using default for now.
  tc::InferOptions options(model_name);
  options.model_version_ = model_version;
  options.client_timeout_ = client_timeout;
  options.request_id_ = std::to_string(uid);

  //std::vector<tc::InferInput*> inputs = {input0_ptr.get(), input2_ptr.get(), request_output_len_tensor_ptr.get(), end_id_tensor_ptr.get(), pad_id_tensor_ptr.get(), beam_width_tensor_ptr.get(), temperature_tensor_ptr.get()};
  std::vector<tc::InferInput*> inputs = {input0_ptr.get(), input2_ptr.get(), request_output_len_tensor_ptr.get(), end_id_tensor_ptr.get(), pad_id_tensor_ptr.get()};
  std::vector<const tc::InferRequestedOutput*> outputs = {
      output0_ptr.get()};

  // Send inference request to the inference server.
  std::mutex mtx;
  std::condition_variable cv;
  size_t repeat_cnt = 1;
  size_t done_cnt = 0;
  for (size_t i = 0; i < repeat_cnt; i++) {
    FAIL_IF_ERR(
        client->AsyncInfer(
            [&, i](tc::InferResult* result) {
              {
                std::shared_ptr<tc::InferResult> result_ptr;
                result_ptr.reset(result);
                std::lock_guard<std::mutex> lk(mtx);
                std::cout << "Callback no." << i << " is called" << std::endl;
                done_cnt++;
                if (result_ptr->RequestStatus().IsOk()) {
                  //ValidateResult(result_ptr, input0_data, input1_data);
                } else {
                  std::cerr << "error: Inference failed: "
                            << result_ptr->RequestStatus() << std::endl;
                  exit(1);
                }
              }
              cv.notify_all();
            },
            options, inputs, outputs, http_headers),
        "unable to run model");
  }

  // Wait until all callbacks are invoked
  {
    std::unique_lock<std::mutex> lk(mtx);
    cv.wait(lk, [&]() {
      if (done_cnt >= repeat_cnt) {
        return true;
      } else {
        return false;
      }
    });
  }
  if (done_cnt == repeat_cnt) {
    std::cout << "All done" << std::endl;
  } else {
    std::cerr << "Done cnt: " << done_cnt
              << " does not match repeat cnt: " << repeat_cnt << std::endl;
    exit(1);
  }

  // Send another AsyncInfer whose callback defers the completed request
  // to another thread (main thread) to handle
  bool callback_invoked = false;
  std::shared_ptr<tc::InferResult> result_placeholder;
  FAIL_IF_ERR(
      client->AsyncInfer(
          [&](tc::InferResult* result) {
            {
              std::shared_ptr<tc::InferResult> result_ptr;
              result_ptr.reset(result);
              // Defer the response retrieval to main thread
              std::lock_guard<std::mutex> lk(mtx);
              callback_invoked = true;
              result_placeholder = std::move(result_ptr);
            }
            cv.notify_all();
          },
          options, inputs, outputs, http_headers),
      "unable to run model");

  // Ensure callback is completed
  {
    std::unique_lock<std::mutex> lk(mtx);
    cv.wait(lk, [&]() { return callback_invoked; });
  }

  // Get deferred response
  std::cout << "Getting results from deferred response" << std::endl;
  if (result_placeholder->RequestStatus().IsOk()) {
    //ValidateResult(result_placeholder, input0_data, input1_data);
  } else {
    std::cerr << "error: Inference failed: "
              << result_placeholder->RequestStatus() << std::endl;
    exit(1);
  }

  tc::InferStat infer_stat;
  client->ClientInferStat(&infer_stat);
  std::cout << "completed_request_count " << infer_stat.completed_request_count
            << std::endl;
  std::cout << "cumulative_total_request_time_ns "
            << infer_stat.cumulative_total_request_time_ns << std::endl;
  std::cout << "cumulative_send_time_ns " << infer_stat.cumulative_send_time_ns
            << std::endl;
  std::cout << "cumulative_receive_time_ns "
            << infer_stat.cumulative_receive_time_ns << std::endl;

  std::cout << "PASS : Async Infer" << std::endl;

  return 0;
}
