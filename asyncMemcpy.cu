#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include "helper_timer.h"
#include "helper_cuda.h"
#include <nvToolsExt.h>
#include <random>
#include <thread>
#include <chrono>

const size_t BLOCKSIZE = 512;
#define NUM_ITERATIONS 10
const size_t MEM_SIZE = (1 * 1024 * 1024ull); // Bytes
const size_t CHUNK_SIZE = (32 * 1024ull); // Bytes

StopWatchInterface *timer = NULL;
cudaEvent_t start, stop;
cudaStream_t copyStream[2];


/***************/
/* COPY KERNEL */
/***************/
__global__ void copyKernelDouble(const double * __restrict__ d_in, double * __restrict__ d_out, const int N) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= N) return;

  d_out[tid] = d_in[tid];
}

__global__ void copyKernel(const char * __restrict__ d_in, char * __restrict__ d_out, const int N) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= N) return;

  d_out[tid] = d_in[tid];
}



float chunked_copykernel_htod(const void *src_data, void *dst_data, size_t bytes, size_t chunkSize, int stream_id) 
{
  size_t offset = 0, transferSize = 0;
  nvtxRangePush("CHUNKED_COPYKERNEL_HTOD_STREAM");
  while(offset < bytes) {
    transferSize = std::min(static_cast<size_t>(chunkSize), bytes - offset);
    const void* chunkSrc = static_cast<const void *>(static_cast<const uint8_t*>(src_data) + offset);
    void* chunkDst = static_cast<void *>(static_cast<uint8_t*>(dst_data) + offset);
    if(stream_id >=0) {
      //copyKernelDouble << <(transferSize / BLOCKSIZE), BLOCKSIZE, 0, copyStream[stream_id] >> >((double*)chunkSrc, (double*)chunkDst, transferSize);
      copyKernel << <(transferSize / BLOCKSIZE), BLOCKSIZE, 0, copyStream[stream_id] >> >((char*)chunkSrc, (char*)chunkDst, transferSize);
    }
    else {
      //copyKernelDouble << <(transferSize / BLOCKSIZE), BLOCKSIZE, 0, 0 >> >((double*)chunkSrc, (double*)chunkDst, transferSize);
      copyKernel << <(transferSize / BLOCKSIZE), BLOCKSIZE, 0, 0 >> >((char*)chunkSrc, (char*)chunkDst, transferSize);
    }
    offset += transferSize;
  }
  nvtxRangePop();
  float elapsedTimeInMs = 0.0f;
  return elapsedTimeInMs;
}



void init_memory(unsigned char* memRegion, size_t memSize)
{
  //initialize the memory
  for (size_t i = 0; i < memSize/sizeof(unsigned char); i++)
  {
    memRegion[i] = (unsigned char)(i & 0xff);
  }
}

float multi_chunked_memcpy_htod(const void *src_data_1, const void *src_data_2, void *dst_data_1, void *dst_data_2, size_t bytes, size_t chunkSize) 
{
  //sdkResetTimer(&timer);
  //sdkStartTimer(&timer);
  size_t offset = 0, transferSize = 0;
  nvtxRangePush("CHUNKED_MEMCPY_HTOD_STREAM");
//  checkCudaErrors(cudaEventRecord(start, copyStream[stream_id]));
  
  while(offset < bytes) {
    transferSize = std::min(static_cast<size_t>(chunkSize), bytes - offset);
    const void* chunkSrc_1 = static_cast<const void *>(static_cast<const uint8_t*>(src_data_1) + offset);
    const void* chunkSrc_2 = static_cast<const void *>(static_cast<const uint8_t*>(src_data_2) + offset);
    void* chunkDst_1 = static_cast<void *>(static_cast<uint8_t*>(dst_data_1) + offset);
    void* chunkDst_2 = static_cast<void *>(static_cast<uint8_t*>(dst_data_2) + offset);
    checkCudaErrors(cudaMemcpyAsync(chunkDst_1, chunkSrc_1, transferSize, cudaMemcpyHostToDevice, 0));
    checkCudaErrors(cudaMemcpyAsync(chunkDst_2, chunkSrc_2, transferSize, cudaMemcpyHostToDevice, copyStream[0]));
    offset += transferSize;
  }

/*
  while(bytes > 0) {
    transferSize = std::min(static_cast<size_t>(chunkSize), bytes);
    bytes -= transferSize;
    const void* chunkSrc_1 = static_cast<const void *>(static_cast<const uint8_t*>(src_data_1) + bytes);
    const void* chunkSrc_2 = static_cast<const void *>(static_cast<const uint8_t*>(src_data_2) + offset);
    void* chunkDst_1 = static_cast<void *>(static_cast<uint8_t*>(dst_data_1) + bytes);
    void* chunkDst_2 = static_cast<void *>(static_cast<uint8_t*>(dst_data_2) + offset);
    checkCudaErrors(cudaMemcpyAsync(chunkDst_1, chunkSrc_1, transferSize, cudaMemcpyHostToDevice, copyStream[1]));
    checkCudaErrors(cudaMemcpyAsync(chunkDst_2, chunkSrc_2, transferSize, cudaMemcpyHostToDevice, copyStream[0]));
    offset += transferSize;
  }
  */
/*
size_t ctr = 0;
unsigned int stream0 = 0;
while(offset < bytes) {
    transferSize = std::min(static_cast<size_t>(chunkSize), bytes - offset);
    const void* chunkSrc_1 = static_cast<const void *>(static_cast<const uint8_t*>(src_data_1) + offset);
    const void* chunkSrc_2 = static_cast<const void *>(static_cast<const uint8_t*>(src_data_2) + offset);
    void* chunkDst_1 = static_cast<void *>(static_cast<uint8_t*>(dst_data_1) + offset);
    void* chunkDst_2 = static_cast<void *>(static_cast<uint8_t*>(dst_data_2) + offset);
    checkCudaErrors(cudaMemcpyAsync(chunkDst_1, chunkSrc_1, transferSize, cudaMemcpyHostToDevice, copyStream[stream0]));
    checkCudaErrors(cudaMemcpyAsync(chunkDst_2, chunkSrc_2, transferSize, cudaMemcpyHostToDevice, copyStream[(1-stream0)]));
    offset += transferSize;
    if(++ctr % 20 == 0) {
//      checkCudaErrors(cudaDeviceSynchronize());
      stream0 = (1-stream0);
    }
  }
*/
  /*
  //std::random_device rd;
  std::mt19937 gen(12345);
  std::uniform_int_distribution<> dis(0, bytes - chunkSize - 1);

size_t ctr = 0;
//unsigned int stream0 = 0;
while(++ctr < 4096) {
    transferSize = std::min(static_cast<size_t>(chunkSize), bytes);
    size_t offset = dis(gen);
    const void* chunkSrc_1 = static_cast<const void *>(static_cast<const uint8_t*>(src_data_1) + offset);
    void* chunkDst_1 = static_cast<void *>(static_cast<uint8_t*>(dst_data_1) + offset);
    checkCudaErrors(cudaMemcpyAsync(chunkDst_1, chunkSrc_1, transferSize, cudaMemcpyHostToDevice, 0));
    const void* chunkSrc_2 = static_cast<const void *>(static_cast<const uint8_t*>(src_data_2) + offset);
    void* chunkDst_2 = static_cast<void *>(static_cast<uint8_t*>(dst_data_2) + offset);
    checkCudaErrors(cudaMemcpyAsync(chunkDst_2, chunkSrc_2, transferSize, cudaMemcpyHostToDevice, copyStream[0]));
  }
*/

//  checkCudaErrors(cudaEventRecord(stop, copyStream[stream_id]));
  nvtxRangePop();
  
 // checkCudaErrors(cudaDeviceSynchronize());
  //sdkStopTimer(&timer);
  float elapsedTimeInMs = 0.0f;
//#ifdef CPU_TIMING
//  elapsedTimeInMs = sdkGetTimerValue(&timer);
//#else
//  checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));
//#endif
  return elapsedTimeInMs;
}



float chunked_memcpy_htod(const void *src_data, void *dst_data, size_t bytes, size_t chunkSize, int stream_id) 
{
  //sdkResetTimer(&timer);
  //sdkStartTimer(&timer);
  size_t offset = 0, transferSize = 0;
  nvtxRangePush("CHUNKED_MEMCPY_HTOD_STREAM");
//  checkCudaErrors(cudaEventRecord(start, copyStream[stream_id]));
  while(offset < bytes) {
    transferSize = std::min(static_cast<size_t>(chunkSize), bytes - offset);
    const void* chunkSrc = static_cast<const void *>(static_cast<const uint8_t*>(src_data) + offset);
    void* chunkDst = static_cast<void *>(static_cast<uint8_t*>(dst_data) + offset);
    if(stream_id >=0) {
      checkCudaErrors(cudaMemcpyAsync(chunkDst, chunkSrc, transferSize, cudaMemcpyHostToDevice, copyStream[stream_id]));
    }
    else {
      checkCudaErrors(cudaMemcpyAsync(chunkDst, chunkSrc, transferSize, cudaMemcpyHostToDevice, 0));
    }
    offset += transferSize;
  }
//  checkCudaErrors(cudaEventRecord(stop, copyStream[stream_id]));
  nvtxRangePop();
  
 // checkCudaErrors(cudaDeviceSynchronize());
  //sdkStopTimer(&timer);
  float elapsedTimeInMs = 0.0f;
//#ifdef CPU_TIMING
//  elapsedTimeInMs = sdkGetTimerValue(&timer);
//#else
//  checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));
//#endif
  return elapsedTimeInMs;
}

float async_memcpy_htod(const void *src_data, void *dst_data, size_t bytes, size_t chunkSize) 
{
  float elapsedTimeInMs = 0.0f;
  for(unsigned int i = 0; i < 512; i++) {
//  sdkResetTimer(&timer);
//  sdkStartTimer(&timer);
  nvtxRangePush("MEMCPY_HTOD");
//  checkCudaErrors(cudaEventRecord(start, 0));
  checkCudaErrors(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyHostToDevice, copyStream[0]));
//  checkCudaErrors(cudaEventRecord(stop, 0));
  nvtxRangePop();
//  checkCudaErrors(cudaDeviceSynchronize());
//  sdkStopTimer(&timer);
//#ifdef CPU_TIMING
//  elapsedTimeInMs = sdkGetTimerValue(&timer);
//#else
//  checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));
//#endif
  //std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  return elapsedTimeInMs;
}

float chunked_memcpy_dtoh(const void *src_data, void *dst_data, size_t bytes, size_t chunkSize) 
{
  sdkResetTimer(&timer);
  sdkStartTimer(&timer);
  size_t offset = 0, transferSize = 0;
  nvtxRangePush("CHUNKED_MEMCPY_HTOD");
  checkCudaErrors(cudaEventRecord(start, 0));
  while(offset < bytes) {
    transferSize = std::min(static_cast<size_t>(chunkSize), bytes - offset);
    const void* chunkSrc = static_cast<const void *>(static_cast<const uint8_t*>(src_data) + offset);
    void* chunkDst = static_cast<void *>(static_cast<uint8_t*>(dst_data) + offset);
    checkCudaErrors(cudaMemcpyAsync(chunkDst, chunkSrc, transferSize, cudaMemcpyDeviceToHost, 0));
    offset += transferSize;
  }
  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&timer);
  sdkResetTimer(&timer);
  float elapsedTimeInMs = 0.0f;
#ifdef CPU_TIMING
  elapsedTimeInMs = sdkGetTimerValue(&timer);
#else
  checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));
#endif
  return elapsedTimeInMs;
}

float async_memcpy_dtoh(const void *src_data, void *dst_data, size_t bytes, size_t chunkSize) 
{
  sdkResetTimer(&timer);
  sdkStartTimer(&timer);
  checkCudaErrors(cudaEventRecord(start, 0));
  checkCudaErrors(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToHost, 0));
  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&timer);
  sdkResetTimer(&timer);
  float elapsedTimeInMs = 0.0f;
#ifdef CPU_TIMING
  elapsedTimeInMs = sdkGetTimerValue(&timer);
#else
  checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));
#endif
  return elapsedTimeInMs;
}

void runAndTime(float (*memcpy_function)(const void*, void*, size_t, size_t), const void *src_data, void *dst_data, size_t memSize, size_t chunkSize)
{
  std::vector<float> timings;
  for(unsigned int i = 0; i < NUM_ITERATIONS; i++) {
    float time_memcpy = memcpy_function(src_data, dst_data, memSize, chunkSize);
    timings.push_back(time_memcpy);
  }
  double sum_timing = 0.0f;
  for(auto& timing :timings) {
    sum_timing += timing;
  }
  std::cout<< memSize << ", " << chunkSize << ", "<< static_cast<double>(sum_timing/NUM_ITERATIONS) << "\n";
}

int main(const int argc, const char** argv)
{
  sdkCreateTimer(&timer);
  sdkResetTimer(&timer);
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  unsigned char *h1_data = NULL;
  unsigned char *h2_data = NULL;
  size_t memSize = MEM_SIZE ;
  size_t chunkSize = CHUNK_SIZE;

  bool chunkedTransfer = false;
  bool htod = false;
  bool dtoh = false;

  int lowP=0, highP =0;
  checkCudaErrors(cudaDeviceGetStreamPriorityRange(&lowP, &highP));
  std::cout<<lowP<< " "<< highP << "\n";
  cudaStreamCreateWithPriority(&copyStream[0], cudaStreamNonBlocking, lowP);
  cudaStreamCreateWithPriority(&copyStream[1], cudaStreamNonBlocking, highP);

  if (checkCmdLineFlag(argc, (const char **)argv, "memSize"))
  {
    memSize = getCmdLineArgumentInt(argc, argv, "memSize");
//    std::cout<< "> Buffer size: "<< memSize << "\n";

    if (memSize <= 0)
    {
      printf("Illegal argument - memSize must be greater than zero\n");
      return -4000;
    }
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "chunkSize"))
  {
    chunkSize = getCmdLineArgumentInt(argc, argv, "chunkSize");
//    std::cout<< "> Chunk size: " << chunkSize << "\n";
    chunkedTransfer = true;
    if (chunkSize <= 0)
    {
      printf("Illegal argument - chunkSize must be greater than zero\n");
      return -4000;
    }
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "htod"))
  {
    htod = true;
  }
  if (checkCmdLineFlag(argc, (const char **)argv, "dtoh"))
  {
    dtoh = true;
  }
  
  std::cout<<"> Allocating host data\n";
  cudaHostAlloc((void **)&h1_data, memSize, cudaHostAllocWriteCombined);
  cudaHostAlloc((void **)&h2_data, memSize, cudaHostAllocWriteCombined);
  std::cout<<"> Initializing host data\n";
  init_memory(h1_data, memSize);
  init_memory(h2_data, memSize);

  std::cout<<"> Allocating device data\n";
  unsigned char *d1_data = NULL;
  cudaMalloc((void **) & d1_data, memSize);
  unsigned char *d2_data = NULL;
  cudaMalloc((void **) & d2_data, memSize);
  std::cout<<"> Allocated, warming up!\n";
  // WARMUP
  for(unsigned int i = 0 ; i < 10 ; i++) {
    //async_memcpy_htod(h1_data, d1_data, memSize, 0);
    //async_memcpy_htod(h2_data, d2_data, memSize, 0);
    //chunked_memcpy_htod(h_data, d_data, memSize, chunkSize);
    //async_memcpy_dtoh(d_data, h_data, memSize, 0);
    //chunked_memcpy_dtoh(d_data, h_data, memSize, chunkSize);
  }
  std::cout << "> Warmed up!\n";

  sdkResetTimer(&timer);
  sdkStartTimer(&timer);
  //multi_chunked_memcpy_htod(h1_data, h2_data, d1_data, d2_data, memSize, chunkSize);
  //std::thread t1(chunked_memcpy_htod, h1_data, d1_data, memSize, chunkSize, -1);
  //std::thread t2(async_memcpy_htod, h1_data, d1_data, memSize, chunkSize);
  //chunked_memcpy_htod(h1_data, d1_data, memSize, chunkSize, 0);
  //chunked_memcpy_htod(h1_data, d1_data, memSize, chunkSize, -1);
  //async_memcpy_htod(h1_data, d1_data, memSize, chunkSize);
  //chunked_memcpy_htod(h1_data, d1_data, memSize, chunkSize, 1);
  chunked_copykernel_htod(h1_data, d1_data, memSize, chunkSize, 0);
  chunked_copykernel_htod(h2_data, d2_data, memSize, chunkSize, 1);
  //chunked_memcpy_htod(h1_data, d1_data, memSize, chunkSize, 1);
  //chunked_memcpy_htod(h2_data, d2_data, memSize, chunkSize, -1);
  //t1.join();
  //t2.join();
  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&timer);
  std::cout<< "memcpy Took: " << sdkGetTimerValue(&timer) <<" ms\n";
/*

  if(chunkedTransfer) {
    if(htod) {
      runAndTime(chunked_memcpy_htod, h_data, d_data, memSize, chunkSize);
    }
    if(dtoh) {
      runAndTime(chunked_memcpy_dtoh, d_data, h_data, memSize, chunkSize);
    }
  } else {
    if(htod) {
      runAndTime(async_memcpy_htod, h_data, d_data, memSize, 0);
    }
    if(dtoh) {
      runAndTime(async_memcpy_htod, d_data, h_data, memSize, 0);
    }
  }
  */
}
