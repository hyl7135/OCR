#pragma once

#include <numeric>

#include "ONNX_Base.h"

class HanCharRecog : ONNX_Base {
private:
	cv::Mat Preprocessing(std::string imagePath);
	cv::Mat Preprocessing(cv::Mat mat);
	wchar_t GetArgMax();

public:
	HanCharRecog(cv::Size inputSZ = cv::Size(64,64), 
		cv::Size outputSZ = cv::Size(0, 0),
		std::vector<const char*> inNodes = { "posts" },
		std::vector<const char*> outNodes = { "DenseCho2", "DenseJung2", "DenseJong2" })
		: ONNX_Base(inputSZ, outputSZ, inNodes, outNodes) 
	{
		auto tmpEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "HanRecog");
		mEnv = std::move(tmpEnv);
		//HAN_RECOG HAN_RECOG_FILE
		auto embedData = GetEmbedData(HAN_RECOG, HAN_RECOG_FILE);
		this->LoadFromBuffer(embedData);
	};

	void Test(std::string imagePath);
	wchar_t GetResult(std::string);
	wchar_t GetResult(cv::Mat);
};