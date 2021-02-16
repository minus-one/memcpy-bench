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

//#define CPU_TIMING

const size_t BLOCKSIZE = 256;
const size_t NUM_ITERATIONS=100;
const size_t MEM_SIZE = (1 * 1024 * 1024 * 1024ull); // 1 GiB
const size_t CHUNK_SIZE = (32 * 1024ull); // 32 KiB

StopWatchInterface *timer = NULL;
cudaEvent_t start, stop;
cudaStream_t copyStream[2];

static uint64_t getCurNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  uint64_t t = ts.tv_sec*1000*1000*1000 + ts.tv_nsec;
  return t;
}

struct elapsedTime {
  float cpuTimeMs;
  float gpuTimeMs;
};

/*************************/
/* GPU based COPY KERNEL */
/*************************/
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

// Initialize the memory
void init_memory(unsigned char* memRegion, size_t memSize)
{
  for (size_t i = 0; i < memSize/sizeof(unsigned char); i++) {
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

elapsedTime copykernelint(const void *src_data, void *dst_data, size_t bytes, size_t chunkSize) 
{
  elapsedTime et;
  uint64_t cpuStartNs = getCurNs();
#ifdef NVTX_ON
  nvtxRangePush("KERNELINT_MEMCPY");
#endif
  checkCudaErrors(cudaEventRecord(start, 0));
  copyKernel << <(int)(ceil(static_cast<float>(bytes) / BLOCKSIZE)), BLOCKSIZE, 0, 0 >> >((char*)src_data, (char*)dst_data, bytes);
  checkCudaErrors(cudaEventRecord(stop, 0));
#ifdef NVTX_ON
  nvtxRangePop();
#endif
  et.cpuTimeMs = static_cast<float>((getCurNs() - cpuStartNs) / 1E6);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventElapsedTime(&et.gpuTimeMs, start, stop));
  return et;
}

elapsedTime copykerneldouble(const void *src_data, void *dst_data, size_t bytes, size_t chunkSize) 
{
  elapsedTime et;
  uint64_t cpuStartNs = getCurNs();
#ifdef NVTX_ON
  nvtxRangePush("KERNELDOUBLE_MEMCPY");
#endif
  checkCudaErrors(cudaEventRecord(start, 0));
  copyKernelDouble << <(int)(ceil(static_cast<float>(bytes) / 8 / BLOCKSIZE)), BLOCKSIZE, 0, 0 >> >((double*)src_data, (double*)dst_data, bytes);
  checkCudaErrors(cudaEventRecord(stop, 0));
#ifdef NVTX_ON
  nvtxRangePop();
#endif
  et.cpuTimeMs = static_cast<float>((getCurNs() - cpuStartNs) / 1E6);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventElapsedTime(&et.gpuTimeMs, start, stop));
  return et;
}

elapsedTime async_memcpy_htod(const void *src_data, void *dst_data, size_t bytes, size_t chunkSize) 
{
  elapsedTime et;
  uint64_t cpuStartNs = getCurNs();
#ifdef NVTX_ON
  nvtxRangePush("ASYNC_MEMCPY_HTOD");
#endif
  checkCudaErrors(cudaEventRecord(start, 0));
  checkCudaErrors(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyHostToDevice, 0));
  checkCudaErrors(cudaEventRecord(stop, 0));
#ifdef NVTX_ON
  nvtxRangePop();
#endif
  et.cpuTimeMs = static_cast<float>((getCurNs() - cpuStartNs) / 1E6);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventElapsedTime(&et.gpuTimeMs, start, stop));
  return et;
}

elapsedTime async_memcpy_dtoh(const void *src_data, void *dst_data, size_t bytes, size_t chunkSize) 
{
  elapsedTime et;
  uint64_t cpuStartNs = getCurNs();
#ifdef NVTX_ON
  nvtxRangePush("ASYNC_MEMCPY_HTOD");
#endif
  checkCudaErrors(cudaEventRecord(start, 0));
  checkCudaErrors(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToHost, 0));
  checkCudaErrors(cudaEventRecord(stop, 0));
#ifdef NVTX_ON
  nvtxRangePop();
#endif
  et.cpuTimeMs = static_cast<float>((getCurNs() - cpuStartNs) / 1E6);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventElapsedTime(&et.gpuTimeMs, start, stop));
  return et;
}

elapsedTime sync_memcpy_htod(const void *src_data, void *dst_data, size_t bytes, size_t chunkSize) 
{
  elapsedTime et;
  uint64_t cpuStartNs = getCurNs();
#ifdef NVTX_ON
  nvtxRangePush("SYNC_MEMCPY_HTOD");
#endif
  checkCudaErrors(cudaEventRecord(start, 0));
  checkCudaErrors(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaEventRecord(stop, 0));
#ifdef NVTX_ON
  nvtxRangePop();
#endif
  et.cpuTimeMs = static_cast<float>((getCurNs() - cpuStartNs) / 1E6);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventElapsedTime(&et.gpuTimeMs, start, stop));
  return et;
}

elapsedTime sync_memcpy_dtoh(const void *src_data, void *dst_data, size_t bytes, size_t chunkSize) 
{
  elapsedTime et;
  uint64_t cpuStartNs = getCurNs();
#ifdef NVTX_ON
  nvtxRangePush("ASYNC_MEMCPY_HTOD");
#endif
  checkCudaErrors(cudaEventRecord(start, 0));
  checkCudaErrors(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaEventRecord(stop, 0));
#ifdef NVTX_ON
  nvtxRangePop();
#endif
  et.cpuTimeMs = static_cast<float>((getCurNs() - cpuStartNs) / 1E6);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventElapsedTime(&et.gpuTimeMs, start, stop));
  return et;
}
void runAndTime(std::string dutName, elapsedTime (*memcpy_function)(const void*, void*, size_t, size_t), const void *src_data, void *dst_data, size_t memSize, size_t chunkSize)
{
  std::vector<elapsedTime> timingsInMs;
  for(unsigned int i = 0; i < NUM_ITERATIONS; i++) {
    elapsedTime time_memcpy = memcpy_function(src_data, dst_data, memSize, chunkSize);
    timingsInMs.push_back(time_memcpy);
  }
  double sum_gpu_timing = 0.0f, sum_cpu_timing = 0.0f;
  for(auto& timing :timingsInMs) {
    sum_cpu_timing += timing.cpuTimeMs;
    sum_gpu_timing += timing.gpuTimeMs;
  }
  std::cout<< dutName << ", " << memSize << ", " << chunkSize << ", "<< static_cast<double>(sum_cpu_timing/NUM_ITERATIONS) << ", " << static_cast<double>(sum_gpu_timing/NUM_ITERATIONS) <<"\n";
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

  //int lowP=0, highP =0;
  //checkCudaErrors(cudaDeviceGetStreamPriorityRange(&lowP, &highP));
  //cudaStreamCreateWithPriority(&copyStream[0], cudaStreamNonBlocking, lowP);
  //cudaStreamCreateWithPriority(&copyStream[1], cudaStreamNonBlocking, highP);

  if (checkCmdLineFlag(argc, (const char **)argv, "memSize")) {
    memSize = getCmdLineArgumentInt(argc, argv, "memSize");
    std::cout<< "> Buffer size: "<< memSize << "\n";
    if (memSize <= 0) {
      printf("Illegal argument - memSize must be greater than zero\n");
      return -4000;
    }
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "chunkSize")) {
    chunkSize = getCmdLineArgumentInt(argc, argv, "chunkSize");
    std::cout<< "> Chunk size: " << chunkSize << "\n";
    chunkedTransfer = true;
    if (chunkSize <= 0) {
      printf("Illegal argument - chunkSize must be greater than zero\n");
      return -4000;
    }
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "htod")) {
    htod = true;
  }
  if (checkCmdLineFlag(argc, (const char **)argv, "dtoh")) {
    dtoh = true;
  }
  
  std::cout<<"> Allocating host data\n";
  cudaHostAlloc((void **)&h1_data, MEM_SIZE, cudaHostAllocWriteCombined);
  cudaHostAlloc((void **)&h2_data, MEM_SIZE, cudaHostAllocWriteCombined);
  std::cout<<"> Initializing host data\n";
  init_memory(h1_data, MEM_SIZE);
  init_memory(h2_data, MEM_SIZE);

  std::cout<<"> Allocating device data\n";
  unsigned char *d1_data = NULL;
  cudaMalloc((void **) & d1_data, MEM_SIZE);
  unsigned char *d2_data = NULL;
  cudaMalloc((void **) & d2_data, MEM_SIZE);

  std::cout<<"> Allocated, warming up!\n";
  // WARMUP
  for(unsigned int i = 0 ; i < 10 ; i++) {
    sync_memcpy_htod(h1_data, d1_data, 1024, 0);
    sync_memcpy_dtoh(d1_data, h1_data, 1024, 0);
    async_memcpy_htod(h1_data, d1_data, 1024, 0);
    async_memcpy_dtoh(d1_data, h1_data, 1024, 0);
    copykernelint(h1_data, d1_data, 1024, 0);
    copykerneldouble(h1_data, d1_data, 1024, 0);
    copykernelint(d1_data, h1_data, 1024, 0);
    copykerneldouble(d1_data, h1_data, 1024, 0);
    //chunked_memcpy_htod(h_data, d_data, memSize, chunkSize);
    //chunked_memcpy_dtoh(d_data, h_data, memSize, chunkSize);
  }
  std::cout << "> Warmed up!\n";

  //multi_chunked_memcpy_htod(h1_data, h2_data, d1_data, d2_data, memSize, chunkSize);
  //std::thread t1(chunked_memcpy_htod, h1_data, d1_data, memSize, chunkSize, -1);
  //std::thread t2(async_memcpy_htod, h1_data, d1_data, memSize, chunkSize);
  //chunked_memcpy_htod(h1_data, d1_data, memSize, chunkSize, 0);
  //chunked_memcpy_htod(h1_data, d1_data, memSize, chunkSize, -1);
  //async_memcpy_htod(h1_data, d1_data, memSize, chunkSize);
  //chunked_memcpy_htod(h1_data, d1_data, memSize, chunkSize, 1);
  //chunked_copykernel_htod(h1_data, d1_data, memSize, chunkSize, 0);
  //chunked_copykernel_htod(h2_data, d2_data, memSize, chunkSize, 1);
  //chunked_memcpy_htod(h1_data, d1_data, memSize, chunkSize, 1);
  //chunked_memcpy_htod(h2_data, d2_data, memSize, chunkSize, -1);
  //t1.join();
  //t2.join();
  //std::cout<< "memcpy Took: " << sdkGetTimerValue(&timer) <<" ms\n";

  std::cout<< "DUT" << ", " << "MemSize" << ", " << "ChunkSize" << ", "<< "Mean CPU-Time(ms)" << ", " << "Mean GPU-Time(ms)" << "\n";
  if(chunkedTransfer) {
    if(htod) {
      //runAndTime("CHUNKED_MEMCPY_HTOD", chunked_memcpy_htod, h_data, d_data, memSize, chunkSize);
    }
    if(dtoh) {
      //runAndTime("CHUNKED_MEMCPY_DTOH", chunked_memcpy_dtoh, d_data, h_data, memSize, chunkSize);
    }
  } else {
    if(htod) {
      for(size_t memSize = 1 ; memSize <= MEM_SIZE ; memSize = memSize * 2)
        runAndTime("SYNC_MEMCPY_HTOD", sync_memcpy_htod, h1_data, d1_data, memSize, 0);

      for(size_t memSize = 1 ; memSize <= MEM_SIZE ; memSize = memSize * 2)
        runAndTime("ASYNC_MEMCPY_HTOD", async_memcpy_htod, h1_data, d1_data, memSize, 0);
      
      for(size_t memSize = 1 ; memSize <= MEM_SIZE ; memSize = memSize * 2)
        runAndTime("COPYKERNELINT_MEMCPY_HTOD", copykernelint, h1_data, d1_data, memSize, 0);
      
      for(size_t memSize = 1 ; memSize <= MEM_SIZE ; memSize = memSize * 2)
        runAndTime("COPYKERNELDOUBLE_MEMCPY_HTOD", copykerneldouble, h1_data, d1_data, memSize, 0);
    }
    if(dtoh) {
      for(size_t memSize = 1 ; memSize <= MEM_SIZE ; memSize = memSize * 2)
        runAndTime("SYNC_MEMCPY_DTOH", sync_memcpy_dtoh, d1_data, h1_data, memSize, 0);  for(size_t memSize = 1 ; memSize <= MEM_SIZE ; memSize = memSize * 2);
 
      for(size_t memSize = 1 ; memSize <= MEM_SIZE ; memSize = memSize * 2)
        runAndTime("ASYNC_MEMCPY_DTOH", async_memcpy_dtoh, d1_data, h1_data, memSize, 0);  for(size_t memSize = 1 ; memSize <= MEM_SIZE ; memSize = memSize * 2);
      
      for(size_t memSize = 1 ; memSize <= MEM_SIZE ; memSize = memSize * 2)
        runAndTime("COPYKERNELINT_MEMCPY_DTOH", copykernelint, d1_data, h1_data, memSize, 0);
      
      for(size_t memSize = 1 ; memSize <= MEM_SIZE ; memSize = memSize * 2)
        runAndTime("COPYKERNELDOUBLE_MEMCPY_DTOH", copykerneldouble, d1_data, h1_data, memSize, 0);
    }
  }
  std::cout<<"> Benchmark complete!\n";
}
