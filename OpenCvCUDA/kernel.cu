
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <stdio.h>

void* buffers[3];

using namespace cv;


//Try kernel blur image with average 
__global__ void AvgblurImage(uchar* input, uchar* output, int width, int height) {
//__global__ void AvgblurImage(char* input, char* output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int pixelIndex = row * width + col;

        //average blur
        float sum = 0.0;
        int count = 0;

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int x = col + i;
                int y = row + j;

                if (x >= 0 && x < width && y >= 0 && y < height) {
                    sum += input[y * width + x];
                    count++;
                }
            }
        }

        output[pixelIndex] = static_cast<uchar>(sum / count);
    }
}

//Kernel Try for convolution blur
__global__ void blurImageConvolution(uchar* image, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index = y * width + x;
        int numChannels = 3;

        float kernel[9] = { 1, 1, 1,
                           1, 1, 1,
                           1, 1, 1 };

        float sumChannels[3] = { 0, 0, 0 };
        for (int c = 0; c < numChannels; c++)
        {
            for (int r = -1; r <= 1; r++)
            {
                for (int s = -1; s <= 1; s++)
                {
                    int pixelX = x + s;
                    int pixelY = y + r;
                    int pixelIndex = pixelY * width + pixelX;
                    uchar pixelValue = image[pixelIndex * numChannels + c];
                    sumChannels[c] += pixelValue * kernel[(r + 1) * 3 + (s + 1)];
                }
            }
        }

        uchar blurredValue = sumChannels[0] / 9;
        image[index * numChannels] = blurredValue;
        image[index * numChannels + 1] = blurredValue;
        image[index * numChannels + 2] = blurredValue;
    }
}


// CUDA Kernel for image blurring
__global__ void imageBlur(const uchar* src, uchar* dst, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index = y * width + x;

        // Perform simple image convolution
        dst[index] = (src[index] + src[index - 1] + src[index + 1] + src[index - width] + src[index + width]) / 5;
    }
}

int main()
{
    // Read image
    cv::Mat originalImage = cv::imread("C:/Users/ianli/source/repos/CUDABlur/rookie.jpg", cv::IMREAD_GRAYSCALE);


    if (originalImage.empty())
    {
        std::cout << "Could not open or find the image." << std::endl;
        return -1;
    }


    // Copy image to GPU memory
    uchar* gpuSrcImage;
    size_t imageSize = originalImage.cols * originalImage.rows * sizeof(uchar);
    cudaMalloc(&gpuSrcImage, imageSize);
    cudaMemcpy(gpuSrcImage, originalImage.data, imageSize, cudaMemcpyHostToDevice);//CPU to GPU

    //Buffers for Other Kernels of Blur to compare
    buffers[0] = malloc(imageSize);
    cudaMalloc(&buffers[1], imageSize);
    cudaMalloc(&buffers[2], imageSize);

    cudaMemcpy(buffers[1], originalImage.ptr(), imageSize, cudaMemcpyHostToDevice);
    // Copy result from GPU to CPU
    cudaMemcpy(buffers[0], buffers[1], imageSize, cudaMemcpyDeviceToHost);
    cv::Mat salida = cv::Mat(cv::Size(originalImage.cols, originalImage.rows), CV_8U);
    std::memcpy(salida.data, buffers[0], imageSize);

    // Create a destination image on GPU memory
    uchar* gpuDstImage;
    cudaMalloc(&gpuDstImage, imageSize);

    // Define CUDA block and grid dimensions
    dim3 blockDims(16, 16);
    dim3 gridDims((originalImage.cols + blockDims.x - 1) / blockDims.x, (originalImage.rows + blockDims.y - 1) / blockDims.y);
    

    // Call the CUDA kernel to blur the imag
    imageBlur << <gridDims, blockDims >> > (gpuSrcImage, gpuDstImage, originalImage.cols, originalImage.rows);
    //AvgblurImage << <gridDims, blockDims >> > ((uchar*)buffers[1], (uchar*)buffers[2], originalImage.cols, originalImage.rows);
    blurImageConvolution << <gridDims, blockDims >> > ((uchar*)buffers[1], originalImage.cols, originalImage.rows);

    // Copy the result image from GPU to CPU memory
    uchar* resultImage = new uchar[imageSize];
    cv::Mat blurredImage2 = cv::Mat(cv::Size(originalImage.cols, originalImage.rows), CV_8U);
    cudaMemcpy(resultImage, gpuDstImage, imageSize, cudaMemcpyDeviceToHost); //GPU to CPU
    //cudaMemcpy(blurredImage2.data, buffers[2], imageSize, cudaMemcpyDeviceToHost);

    // Save the blurred image locally
    cv::Mat blurredImage(originalImage.rows, originalImage.cols, CV_8UC1, resultImage);
    cv::imwrite("blurred_image.jpg", blurredImage);

    // Show the original and blurred images to compare
    cv::imshow("Original Image", originalImage);
    cv::imshow("Convolution try 1 Blurred Image", blurredImage);
    //cv::imshow("Convolution try 2 Blurred Image", blurredImage2);
    //cv::imshow("Convolution try 2 Blurred Image", blurredImage3);
    cv::waitKey(0);

    // Free GPU memory
    cudaFree(gpuSrcImage);
    cudaFree(gpuDstImage);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    cudaFree(buffers[2]);

    delete[] resultImage;

    return 0;
}

////////////////////////Just 1 blur
// CUDA Kernel for image blurring
/*
__global__ void imageBlur(const uchar* src, uchar* dst, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index = y * width + x;

        // Perform simple image convolution
        dst[index] = (src[index] + src[index - 1] + src[index + 1] + src[index - width] + src[index + width]) / 5;
    }
}

int main()
{
    // Read image
    cv::Mat originalImage = cv::imread("C:/Users/ianli/source/repos/CUDABlur/rookie.jpg", cv::IMREAD_GRAYSCALE);

    if (originalImage.empty())
    {
        std::cout << "Could not open or find the image." << std::endl;
        return -1;
    }

    // Copy image to GPU memory
    uchar* gpuSrcImage;
    size_t imageSize = originalImage.cols * originalImage.rows * sizeof(uchar);
    cudaMalloc(&gpuSrcImage, imageSize);
    cudaMemcpy(gpuSrcImage, originalImage.data, imageSize, cudaMemcpyHostToDevice);

    // Create a destination image on GPU memory
    uchar* gpuDstImage;
    cudaMalloc(&gpuDstImage, imageSize);

    // Define CUDA block and grid dimensions
    dim3 blockDims(16, 16);
    dim3 gridDims((originalImage.cols + blockDims.x - 1) / blockDims.x, (originalImage.rows + blockDims.y - 1) / blockDims.y);

    // Call the CUDA kernel to blur the image
    imageBlur << <gridDims, blockDims >> > (gpuSrcImage, gpuDstImage, originalImage.cols, originalImage.rows);

    // Copy the result image from GPU to CPU memory
    uchar* resultImage = new uchar[imageSize];
    cudaMemcpy(resultImage, gpuDstImage, imageSize, cudaMemcpyDeviceToHost);

    // Save the blurred image locally
    cv::Mat blurredImage(originalImage.rows, originalImage.cols, CV_8UC1, resultImage);
    cv::imwrite("blurred_image.jpg", blurredImage);

    // Show both the original and blurred images
    cv::imshow("Original Image", originalImage);
    cv::imshow("Blurred Image", blurredImage);
    cv::waitKey(0);

    // Free GPU memory
    cudaFree(gpuSrcImage);
    cudaFree(gpuDstImage);
    delete[] resultImage;

    return 0;
}
*/
/////////////////////////