"""Independent cumulative-sum oracle for the actual repaired CUDA helper."""
import json,subprocess
from signalforge.runtime import paths,gpu_lease,file_hash
from signalforge.successor_backend import verify_backend

def test_sorted_weight_prefix_cuda_matches_independent_cumulative_sum():
    repo,runtime=paths();backend=verify_backend(repo,runtime)
    assert backend is not None and backend['repair_attempt']==2
    root=(runtime/backend['package_relative_path']).parent
    header=root/'source/lightgbm-4.6.0/include/LightGBM/cuda/cuda_algorithms.hpp'
    changes=[c for c in backend['additional_corrections'] if c['path']=='include/LightGBM/cuda/cuda_algorithms.hpp']
    assert file_hash(header)==changes[-1]['after_sha256']
    folder=runtime/'tools/successor_prefix_oracle';folder.mkdir(exist_ok=True)
    source=folder/'oracle.cu';binary=folder/'oracle'
    source.write_text(r'''
#include <LightGBM/cuda/cuda_algorithms.hpp>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdio>
__global__ void prefix(const float* weights,const int* indices,double* out,int n){
  LightGBM::ShuffleSortedPrefixSumDevice<float,double,int>(weights,indices,out,n);
}
int main(){
 for(int n: {31,32,33,511,512,513,4576}){
  std::vector<float>w(n);std::vector<int>idx(n);std::vector<double>out(n);
  for(int i=0;i<n;i++){w[i]=(i%7+1)*0.125f;idx[i]=n-1-i;}
  float*dw;int*di;double*du;
  if(cudaMalloc(&dw,n*sizeof(float))||cudaMalloc(&di,n*sizeof(int))||cudaMalloc(&du,n*sizeof(double)))return 2;
  cudaMemcpy(dw,w.data(),n*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(di,idx.data(),n*sizeof(int),cudaMemcpyHostToDevice);
  prefix<<<1,512>>>(dw,di,du,n);
  if(cudaDeviceSynchronize()!=cudaSuccess)return 3;
  cudaMemcpy(out.data(),du,n*sizeof(double),cudaMemcpyDeviceToHost);
  double expected=0;
  for(int i=0;i<n;i++){expected+=w[idx[i]];if(std::abs(expected-out[i])>1e-9){printf("n=%d index=%d expected=%f actual=%f\n",n,i,expected,out[i]);return 4;}}
  cudaFree(dw);cudaFree(di);cudaFree(du);
 }
 puts("PASSED_ACTUAL_CUDA_PREFIX_ORACLE");return 0;
}
''')
    includes=[header.parents[2],header.parents[3]/'external_libs/fast_double_parser/include',header.parents[3]/'external_libs/fmt/include',header.parents[3]/'external_libs/eigen']
    compilation=subprocess.run([str(runtime/'env/bin/nvcc'),'-std=c++11','-DUSE_CUDA','-arch=sm_89',*['-I'+str(p) for p in includes],str(source),'-o',str(binary)],capture_output=True,text=True,timeout=90)
    assert compilation.returncode==0,compilation.stderr
    with gpu_lease(runtime):result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
