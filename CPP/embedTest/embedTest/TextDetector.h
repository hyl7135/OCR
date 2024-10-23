#pragma once

#include "ONNX_Base.h"

class TextDetector : ONNX_Base {
private:
	std::vector<std::pair<cv::Rect, std::vector<cv::Rect>>> mInferenceROI;


	cv::Mat Preprocessing(std::string imagePath);
	void ImageProcessing();

public:
	TextDetector(cv::Size inputSZ = cv::Size(768, 768), 
		cv::Size outputSZ = cv::Size(384, 384),
		std::vector<const char*> inNodes = { "input" },
		std::vector<const char*> outNodes = { "output1" })
		: ONNX_Base(inputSZ, outputSZ, inNodes, outNodes) 
	{
		auto tmpEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "TextDetector");
		mEnv = std::move(tmpEnv);

		auto embedData = GetEmbedData(DETECTION, DETECTION_FILE);
		this->LoadFromBuffer(embedData);
	};


	cv::Mat GetOrgImg() { return mOrgImage; }
	cv::Mat GetInputImage() { return mInputImage; }
	cv::Size GetOutputSize() { return kOutputImageSize; }

	void Test(std::string imagePath);

	std::vector<std::pair<cv::Rect, std::vector<cv::Rect>>> GetResult(std::string imagePath);	
};