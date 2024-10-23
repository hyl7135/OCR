#pragma once

#include <vector>
#include <string>
#include <filesystem>

#include "opencv2/opencv.hpp"
//#include "onnxruntime_cxx_api.h"
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"

#include "Utility.h"


//#include "x64/Release/"

class TextDetector_tmp {
	std::string mModelPath;

	const cv::Size kInputImageSize{ 768,768 };
	const cv::Size kOutputImageSize{ 384, 384 };
	const int kChannels = 3;
	const int kInputTensorSize = kInputImageSize.width * kInputImageSize.height * kChannels;
	const std::vector<const char*> mDetectOutputNodeNames = { "output1" };
	const std::vector<const char*> kTmpInputNodeNames = { "input" };

	float mTextThreshold{ 0.4 };
	float mLinkThreshold{ 0.4 };

	float mWidthRatio{ 0 };
	float mHeightRatio{ 0 };

	std::unique_ptr<Ort::Env> mEnv;
	Ort::SessionOptions mSessionOptions;
	std::unique_ptr<Ort::Session> mSession;
	Ort::AllocatorWithDefaultOptions mAllocator;

	cv::Mat mInputImage;
	cv::Mat mOrgImage;

	std::vector<const char*> mInputNodeNames;
	std::vector<int64_t> mInputNodeDims;

	std::vector<Ort::Value> mOutputTensors;
	std::vector<cv::Rect> mCoors;

	//std::vector<std::vector<cv::Rect>> mInferenceROI;

	std::vector<std::pair<cv::Rect, std::vector<cv::Rect>>> mInferenceROI;

public:
	TextDetector_tmp() = delete;
	TextDetector_tmp(std::string);

	void Inference(std::string imagePath);
	void ImageProcessing();
	void ImageProcessing2();
	void ImageProcessing3();

	void test();

	cv::Mat PassRoi(int idx);
	cv::Mat RegularDoc(int idx);

	int GetTextNum();


	cv::Mat GetRoiFromOrg();
	void ROI_Test();
	void ROI_Test2();
};