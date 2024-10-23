#pragma once

#include <vector>
#include <string>
#include <filesystem>
#include <functional>

#include "opencv2/opencv.hpp"
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"

#include "Utility.h"

#define NOT_ASCII '`'

const char ASC_MAP[69] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
	, 'A', 'b', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M'
	, 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
	, 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm'
	, 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
	, ':', '#', '@', '(', ')', '-', NOT_ASCII };

class ONNX_Base {
private:
	const std::vector<const char*> kDetectOutputNodeNames = { "output1" };
	const std::vector<const char*> kTmpInputNodeNames = { "input" };

	const int kChannels = 3;
	const int kInputTensorSize = kInputImageSize.width * kInputImageSize.height * kChannels;

	std::vector<const char*> mInputNodeNames;
	std::vector<int64_t> mInputNodeDims;

	void GetModelInfo();

protected:
	const cv::Size kInputImageSize{ 768,768 };
	const cv::Size kOutputImageSize{ 384, 384 };

	std::unique_ptr<Ort::Env> mEnv;
	Ort::SessionOptions mSessionOptions;
	std::unique_ptr<Ort::Session> mSession;
	Ort::AllocatorWithDefaultOptions mAllocator;

	float mTextThreshold{ 0.4 };
	float mLinkThreshold{ 0.4 };

	float mWidthRatio{ 0 };
	float mHeightRatio{ 0 };

	cv::Mat mInputImage;
	cv::Mat mOrgImage;

	std::string mModelPath;

	std::vector<Ort::Value> mOutputTensors;

	void Inference(std::string imagePath);
	void Inference(cv::Mat blob);
	void Inference(std::function<cv::Mat(std::string)> funcPtr, std::string imagePath);

public:
	ONNX_Base(cv::Size inputSZ, cv::Size outputSZ, std::vector<const char*> inNodes, std::vector<const char*> outNodes)
		: kInputImageSize(inputSZ), kOutputImageSize(outputSZ),
		kDetectOutputNodeNames(outNodes), kTmpInputNodeNames(inNodes) {};
		//kDetectOutputNodeNames(inNodes), kTmpInputNodeNames(outNodes){};

	int LoadFromBuffer(std::pair<std::vector<UCHAR>, DWORD>);
	int LoadFromFile(std::string);
};